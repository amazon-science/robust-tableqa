# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import json
import operator
import pickle
from trainers.base_executor import BaseExecutor
import wandb
import logging
logger = logging.getLogger(__name__)

from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

# For TAPEX model
from transformers import TapexTokenizer, BartConfig, BartForConditionalGeneration
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model

from .metrics_processors import MetricsProcessor
from .base_executor import BaseExecutor
from utils.dirs import *
from models.dpr.dpr_retriever import RetrieverDPR


class ITRDPRExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer
        
        ModelClass = globals()[self.config.model_config.ModelClass]

        self.model = ModelClass(config=config)
        self.model.resize_token_embeddings(len(self.tokenizer), len(self.decoder_tokenizer))
        
    
    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """

        def get_parameter_names(model, forbidden_layer_types):
            """
            Returns the names of the model parameters that are not inside a forbidden layer.
            """
            result = []
            for name, child in model.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_parameter_names(child, forbidden_layer_types)
                    if not isinstance(child, tuple(forbidden_layer_types))
                ]
            # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
            result += list(model._parameters.keys())
            return result
        
        weight_decay = self.config.train.additional.get('weight_decay', 0)
        if weight_decay == 0:
            optimization_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters()],
                    'lr': self.config.train.lr,
                    'initial_lr': self.config.train.lr,
                },
            ]
        else:
            # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimization_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": weight_decay,
                    'lr': self.config.train.lr,
                    'initial_lr': self.config.train.lr,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                    'lr': self.config.train.lr,
                    'initial_lr': self.config.train.lr,
                },
            ]

        for group in optimization_parameters:
            logger.info('#params: {}   lr: {}'.format(len(group['params']), group['lr']))
        

        """define optimizer"""
        self.optimizer = torch.optim.AdamW(
            optimization_parameters, lr=self.config.train.lr)

        if self.config.train.scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup
            # Using Linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                last_epoch=self.global_step,
            )
        elif self.config.train.scheduler == 'cosine':
            t_total = self.config.train.epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                            t_total, eta_min=1e-5, last_epoch=-1, verbose=False)
        else:
            from transformers import get_constant_schedule_with_warmup
            # Using constant scheduler
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
                last_epoch=self.global_step,
            )
        
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                # REQUIRED: The scheduler instance
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }
        }


    def training_step(self, sample_batched, batch_idx):
        train_batch = {
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
            'labels': sample_batched['labels'].to(self.device),
            'item_input_ids': sample_batched['decoder_input_ids'].to(self.device),
            'item_attention_mask': sample_batched['decoder_input_attention_mask'].to(self.device),
        }

        forward_results = self.model(**train_batch)
        batch_loss = forward_results.loss

        # if unwrap_model(self.model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #     batch_loss = self.label_smoother(forward_results, train_batch.labels, shift_labels=True)
        # else:
        #     batch_loss = self.label_smoother(forward_results, train_batch.labels)
        
        # log the current learning rate from shedulers
        current_lrs = self.scheduler.get_last_lr()
        for index, current_lr in enumerate(current_lrs):
            self.log(f"train/lr[{index}]", current_lr, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train/loss", batch_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        data_to_return = {
            'loss': batch_loss,
        }
        return data_to_return
    

    def validation_step(self, sample_batched, batch_idx, dataloader_idx=0):
        # print(f'batch_idx {batch_idx}  dataloader_idx {dataloader_idx}')
        return self._compute_query_embeddings_step(sample_batched, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):

        for i in range(len(self.val_dataloader())):
            if len(self.val_dataloader()) == 1:
                validation_step_output = validation_step_outputs
            else:
                validation_step_output = validation_step_outputs[i]
            
            log_dict = self.evaluate_outputs(validation_step_output, self.val_dataloader()[i], self.val_dataloader_names[i])
            self.logging_results(log_dict, prefix=self.val_dataloader_names[i])
        return None
    
    def test_step(self, sample_batched, batch_idx, dataloader_idx=0):
        return self._compute_query_embeddings_step(sample_batched, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        self.save_HF_model()
        for i in range(len(self.test_dataloader())):
            if len(self.test_dataloader()) == 1:
                test_step_output = test_step_outputs
            else:
                test_step_output = test_step_outputs[i]
            
            log_dict = self.evaluate_outputs(test_step_output, self.test_dataloader()[i], self.test_dataloader_names[i])
            self.logging_results(log_dict, prefix=f"{self.config.test.evaluation_name}_{self.test_dataloader_names[i]}")
        
        return None

    def _compute_query_embeddings_step(self, sample_batched, batch_idx):
        """
        This function is shared by both valid and test
        """
        test_batch = {
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
        }
        # batch_size x hidden_states
        query_emb = self.model.generate_query_embeddings(**test_batch)
        
        data_to_return = {
            'btach_idx': batch_idx,
            'query_emb': query_emb,
            'question_ids': sample_batched['question_ids'],
            'answers': sample_batched['answers'],
            'sub_tables': sample_batched['sub_tables'],
            'gold_columns': sample_batched['gold_columns'],
            'gold_rows': sample_batched['gold_rows'],
        }

        return data_to_return
    


    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name, mode='test'):
        # Batching every validation step outputs
        query_embeddings = []
        question_ids = []
        sub_tables = []
        gold_column_dict = {}
        gold_row_dict = {}

        question_id_to_sub_table_index = {}
        question_id_to_query_index = {}

        

        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])

            for question_id, sub_table_list in zip(step_output['question_ids'], step_output['sub_tables']):
                # question_id: (start_index, end_index)
                question_id_to_sub_table_index[question_id] = (len(sub_tables), len(sub_tables)+len(sub_table_list))
                sub_tables += sub_table_list
                question_id_to_query_index[question_id] = len(question_id_to_query_index)
            
            question_ids += step_output['question_ids']
            for question_id, gold_columns, gold_rows in zip(step_output['question_ids'], step_output['gold_columns'], step_output['gold_rows']):
                gold_column_dict[question_id] = gold_columns
                gold_row_dict[question_id] = gold_rows

        # question_ids = [0, 1, 2, ...]
        # sub_tables = [0_0, 0_1, 0_2, ..., 1_0, 1_1, 1_2,...]
        
        query_embeddings = torch.cat(query_embeddings, dim=0)
        
        ##################################
        ##    Generate embeds for items ##
        ##################################
        
        n_queries = query_embeddings.shape[0]
        hidden_size = query_embeddings.shape[1]
        n_items = len(sub_tables)

        i_batch_size = self.config[mode].batch_size
        
        n_item_batchs = n_items // i_batch_size + 1

        rate_batch = np.zeros(shape=(n_queries, n_items))
        # rate_batch = np.random.randint(0, 100, size=(n_queries, n_items))
        logger.info(f'rate_batch shape: {rate_batch.shape}')

        # Create mapping between matrix indice and sub_table ids
        decoder_input_modules = self.config.model_config.decoder_input_modules.module_list
        table_contents = []
        for sub_table in sub_tables:
            sample = EasyDict(table=sub_table)
            parsed_data = current_data_loader.dataset.parse_modules(sample, decoder_input_modules, type='decoder_input')
            table_contents.append(parsed_data.text_sequence)
        

        assert len(table_contents) == len(sub_tables)

        logger.info(f'There are {n_queries} queries.')
        logger.info(f'Generating embeddings for items; there are {n_items} items.')
        i_count = 0
        item_embeddings = []
        for i_batch_id in tqdm(range(n_item_batchs)):
            i_start = i_batch_id * i_batch_size
            i_end = min((i_batch_id + 1) * i_batch_size, n_items)
            if i_end - i_start == 0:
                break
            passage_contents_batch = table_contents[i_start:i_end]
            
            # Encode this batch of data
            item_encoding = self.decoder_tokenizer(passage_contents_batch,
                                padding='longest',
                                max_length=self.config.data_loader.additional.max_decoder_source_length,
                                truncation=True,
                                return_tensors="pt")
            
            item_input_ids, item_attention_mask = item_encoding.input_ids, item_encoding.attention_mask
            
            test_batch = EasyDict({
                'input_ids': item_input_ids.to(self.device),
                'attention_mask': item_attention_mask.to(self.device),
            })

            # batch_size x hidden_states
            item_emb = self.model.generate_item_embeddings(**test_batch)
            item_embeddings.append(item_emb.cpu().detach())

            # n_queries x batch_size
            i_rate_batch = torch.matmul(query_embeddings, item_emb.t()).detach().cpu()

            rate_batch[:, i_start:i_end] = i_rate_batch
            i_count += i_rate_batch.shape[1]

        assert i_count == n_items
        item_embeddings = torch.cat(item_embeddings, dim=0)
        print(item_embeddings.shape)

        Ks = self.config.model_config.Ks
        # Log results
        columns=["question_id"]  \
                    + ['p_{}'.format(i) for i in range(max(Ks))]
        test_table = wandb.Table(columns=columns)

        batch_results = []

        for question_id, start_end_index in tqdm(question_id_to_sub_table_index.items()):
            start, end = start_end_index
            query_index = question_id_to_query_index[question_id]
            query_sub_table_ratings = rate_batch[query_index, start:end] # 1 x end-start
            query_sub_tables = sub_tables[start:end] # length: end-start
            # zip and sort sub_tables wrt scores
            zipped_query_sub_tables = [(query_sub_table, query_sub_table_rating) for query_sub_table, query_sub_table_rating in zip(query_sub_tables, query_sub_table_ratings)]
            # print(zipped_query_sub_tables)
            zipped_query_sub_tables.sort(key=lambda x: x[1], reverse=True)
            # print(zipped_query_sub_tables)
            batch_results.append({
                'question_id': question_id,
                'retrieved_tables_sorted': zipped_query_sub_tables,
                'gold_columns': gold_column_dict[question_id],
                'gold_rows': gold_row_dict[question_id],
            })
            table_entry = [
                question_id,
            ]
            for i in range(max(Ks)):
                if i < len(zipped_query_sub_tables):
                    table, rating = zipped_query_sub_tables[i]
                    table_entry+=[f"{table['id']}, {table['is_gold']}, {rating}"]
                else:
                    table_entry+=[f""]
                
            test_table.add_data(*table_entry)

        # in testing mode, save the generated embeddings
        if self.trainer.state.stage == 'test':
            save_path = os.path.join(self.config.results_path, 'step_{}'.format(self.global_step), dataset_name.replace('/', '.'))
            create_dirs([save_path])
            save_path = os.path.join(save_path, f'static_index_{self.global_rank}.pkl')
            to_save_data = dict(
                query_embeddings=query_embeddings.cpu().detach(),
                item_embeddings=item_embeddings,
                question_ids=question_ids,
                sub_tables=sub_tables,
                question_id_to_sub_table_index=question_id_to_sub_table_index,
                question_id_to_query_index=question_id_to_query_index,
            )
            logger.info(f'saving embedding files into {save_path}')
            
            with open(save_path, 'wb') as f:
                pickle.dump(to_save_data, f)
            
        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_retrieval_results=batch_results,
        )

        log_dict = self.compute_metrics(data_used_for_metrics)
        log_dict.artifacts.test_table = test_table

        return log_dict

    def logging_results(self, log_dict, prefix='test'):
        
        ### Add test results to wandb / tensorboard
        metrics_to_log = EasyDict()
        wandb_artifacts_to_log = dict()
        # Refractor the column names
        for metric, value in log_dict.metrics.items():
            metrics_to_log[f'{prefix}/{metric}'] = value
        
        # include other artifacts / metadata
        metrics_to_log[f'{prefix}/epoch'] = self.current_epoch
        wandb_artifacts_to_log.update({
            f"predictions/step_{self.global_step}_MODE({self.config.mode})_SET({prefix})_rank({self.global_rank})": log_dict.artifacts['test_table']
        })
        pprint(metrics_to_log)
        pprint(wandb_artifacts_to_log)

        logger.info(f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}")
        
        if self.trainer.state.stage in ['sanity_check']:
            logging.warning('Sanity check mode, not saving to loggers.')
            return
        
        # Add to loggers
        for metric, value in metrics_to_log.items():
            if type(value) in [float, int, np.float64]:
                self.log(metric, float(value), logger=True, sync_dist=True)
            else:
                logger.info(f'{metric} is not a type that can be logged, skippped.')
        
        # Call wandb to log artifacts; remember to use commit=False so that the data will be logged
        #       with other metrics later.
        if self.config.args.log_prediction_tables:
            self.wandb_logger.experiment.log(wandb_artifacts_to_log, commit=False)
        
    
    def forward(self, **kwargs):
        return self.model(**kwargs)

    def save_HF_model(self):
        '''
        Save models with the Huggingface built-in save_pretrained() function.
        The checkpoints can be loaded by a RAG-like system.
        '''
        if self.global_rank != 0:
            logger.info('global rank is not 0, skip saving models')
            return
        logger.info('Saving model in the Huggingface format...')
        path_save_model = os.path.join(self.config.saved_model_path, 'step_{}'.format(self.global_step))
        self.model.query_encoder.save_pretrained(os.path.join(path_save_model, 'query_encoder'))
        self.data_loader.tokenizer.save_pretrained(os.path.join(path_save_model, 'query_encoder_tokenizer'))
        self.model.item_encoder.save_pretrained(os.path.join(path_save_model, 'item_encoder'))
        self.data_loader.decoder_tokenizer.save_pretrained(os.path.join(path_save_model, 'item_encoder_tokenizer'))
        logger.info('Model has been saved to {}'.format(path_save_model))
