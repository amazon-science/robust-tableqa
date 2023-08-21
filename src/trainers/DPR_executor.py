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
import pandas as pd
import copy

import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

# For TAPEX model
from transformers import TapexTokenizer, BartConfig, BartForConditionalGeneration
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from datasets import Features, Sequence, Value, load_dataset, Dataset

from .metrics_processors import MetricsProcessor
from .base_executor import BaseExecutor
from utils.dirs import *
from models.dpr.dpr_retriever import RetrieverDPR

# import mkl
# mkl.get_max_threads()

class DPRExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer
        
        ModelClass = globals()[self.config.model_config.ModelClass]

        self.model = ModelClass(config=config)
        self.model.resize_token_embeddings(len(self.tokenizer), len(self.decoder_tokenizer))
        
        self.tmp_table_dataset = None

    
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
        
        # when validation finishes, remove tmp index
        self.tmp_table_dataset = None

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
        # when testing finishes, remove tmp index
        self.tmp_table_dataset = None
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
            'query_emb': query_emb.detach().cpu(),
            'question_ids': sample_batched['question_ids'],
            'answers': sample_batched['answers'],
            'pos_item_ids': sample_batched['pos_item_ids'],
        }

        return data_to_return
    


    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name, mode='test'):
        # Batching every validation step outputs
        query_embeddings = []
        question_ids = []
        pos_item_ids = []

        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])

            for question_id in step_output['question_ids']:
                question_ids.append(question_id)
            pos_item_ids.extend(step_output['pos_item_ids'])

        # question_ids = [0, 1, 2, ...]
        query_embeddings = torch.cat(query_embeddings, dim=0)
        
        ##################################
        ##    Generate embeds for items ##
        ##################################
        
        n_queries = query_embeddings.shape[0]
        hidden_size = query_embeddings.shape[1]
        tables = current_data_loader.dataset.tables

        if self.tmp_table_dataset is None:
            # When item embeddings are not indexed, call the function
            # this will not be called more than once during a validation step
            # which reduces the time needed for validating more than one datasets
            logger.info("No tmp exists, start building indexes...")
            self.prepare_item_embeddings(current_data_loader, mode)
        else:
            logger.info("reusing pre-computed indexes...")
        
        table_dataset = self.tmp_table_dataset

        # # Create dataset instance and add faiss index
        # table_dataset = pd.DataFrame.from_dict(tables, orient='index')
        # table_dataset = Dataset.from_pandas(table_dataset)
        # table_dataset = table_dataset.rename_columns({'__index_level_0__': "table_id"})
        # if self.trainer.state.stage in ['sanity_check']:
        #     # sanity check
        #     logging.warning('Sanity check. Reducing number of items to speed up the sanity check.')
        #     table_dataset = table_dataset.select(range(1000))

        # n_items = len(table_dataset)

        # logger.info(f"n_queries {n_queries}; n_items {n_items}")

        # i_batch_size = self.config[mode].batch_size
        # n_item_batchs = n_items // i_batch_size + 1

        # # rate_batch = np.zeros(shape=(n_queries, n_items))
        # # rate_batch = np.random.randint(0, 100, size=(n_queries, n_items))
        # # logger.info(f'rate_batch shape: {rate_batch.shape}')

        
        
        # # Create mapping between matrix indice and sub_table ids
        # decoder_input_modules = self.config.model_config.decoder_input_modules.module_list
        # table_contents = []
        # for table in tqdm(table_dataset):
        #     sample = EasyDict(table=table)
        #     parsed_data = current_data_loader.dataset.parse_modules(sample, decoder_input_modules, type='decoder_input')
        #     table_contents.append(parsed_data.text_sequence)
        
        # # assert len(table_contents) == len(tables)
        # table_dataset = table_dataset.add_column("table_contents", table_contents)

        # logger.info(f'There are {n_queries} queries.')
        # logger.info(f'Generating embeddings for items; there are {n_items} items.')

        # i_count = 0
        # item_embeddings = []
        # for i_batch_id in tqdm(range(n_item_batchs)):
        #     i_start = i_batch_id * i_batch_size
        #     i_end = min((i_batch_id + 1) * i_batch_size, n_items)
        #     if i_end - i_start == 0:
        #         break
        #     passage_contents_batch = table_contents[i_start:i_end]
            
        #     # Encode this batch of data
        #     item_encoding = self.decoder_tokenizer(passage_contents_batch,
        #                         padding='longest',
        #                         max_length=self.config.data_loader.additional.max_decoder_source_length,
        #                         truncation=True,
        #                         return_tensors="pt")
            
        #     item_input_ids, item_attention_mask = item_encoding.input_ids, item_encoding.attention_mask
            
        #     test_batch = EasyDict({
        #         'input_ids': item_input_ids.to(self.device),
        #         'attention_mask': item_attention_mask.to(self.device),
        #     })

        #     # batch_size x hidden_states
        #     item_emb = self.model.generate_item_embeddings(**test_batch)
        #     for x in item_emb:
        #         item_embeddings.append(x.cpu().detach().numpy())

        #     # n_queries x batch_size
        #     # i_rate_batch = torch.matmul(query_embeddings, item_emb.t()).detach().cpu()

        #     # rate_batch[:, i_start:i_end] = i_rate_batch
        #     i_count += item_emb.shape[0]

        # assert i_count == n_items
        # # item_embeddings = torch.cat(item_embeddings, dim=0)
        # # print(item_embeddings.shape)

        # table_dataset = table_dataset.add_column("embeddings", item_embeddings)

        # index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)
        # table_dataset.add_faiss_index("embeddings", custom_index=index)

        # Search the index and process results
        Ks = self.config.model_config.Ks
        logger.info(f"searching for {query_embeddings.shape} queries")
        query_results = table_dataset.get_nearest_examples_batch(
            index_name="embeddings",
            queries=query_embeddings.numpy(),
            k=max(Ks),
        )
        batch_results = []

        # Log results
        columns=["question_id", "pos_item_ids"]  \
                    + ['p_{}'.format(i) for i in range(max(Ks))]
        test_table = wandb.Table(columns=columns)

        for question_id, pos_item_id, score, retrieved_tables in zip(question_ids, pos_item_ids, query_results.total_scores, query_results.total_examples):
            retrieved_tables_sorted = retrieved_tables['table_id']
            
            res = {
                'question_id': question_id,
                'retrieved_tables_sorted': retrieved_tables_sorted,
                'retrieved_scores': score,
                'pos_item_ids': pos_item_id,
            }
            batch_results.append(res)
            table_entry = [
                question_id,
                pos_item_id,
            ]
            for retrieved_table_id, retrieved_table_score in zip(retrieved_tables_sorted, score):
                table_entry+=[f"{retrieved_table_id}, {retrieved_table_id==pos_item_id}, {retrieved_table_score}"]
            
            test_table.add_data(*table_entry)
        
        
            
        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_retrieval_results=batch_results,
            Ks=Ks,
        )

        log_dict = self.compute_metrics(data_used_for_metrics)
        log_dict.artifacts.test_table = test_table

        return log_dict


    def prepare_item_embeddings(self, current_data_loader, mode):
        """
        This function generates item embeddings for all tables
        """
        tables = current_data_loader.dataset.tables

        # Create dataset instance and add faiss index
        table_dataset = pd.DataFrame.from_dict(tables, orient='index')
        table_dataset = Dataset.from_pandas(table_dataset)
        table_dataset = table_dataset.rename_columns({'__index_level_0__': "table_id"})
        if self.trainer.state.stage in ['sanity_check']:
            # sanity check
            logging.warning('Sanity check. Reducing number of items to speed up the sanity check.')
            table_dataset = table_dataset.select(range(1000))
        
        n_items = len(table_dataset)

        logger.info(f"n_items {n_items}")

        i_batch_size = self.config[mode].batch_size
        n_item_batchs = n_items // i_batch_size + 1

        # rate_batch = np.zeros(shape=(n_queries, n_items))
        # rate_batch = np.random.randint(0, 100, size=(n_queries, n_items))
        # logger.info(f'rate_batch shape: {rate_batch.shape}')

        
        
        # Create mapping between matrix indice and sub_table ids
        decoder_input_modules = self.config.model_config.decoder_input_modules.module_list
        table_contents = []
        for table in tqdm(table_dataset):
            sample = EasyDict(table=table)
            parsed_data = current_data_loader.dataset.parse_modules(sample, decoder_input_modules, type='decoder_input')
            table_contents.append(parsed_data.text_sequence)
        
        # assert len(table_contents) == len(tables)
        table_dataset = table_dataset.add_column("table_contents", table_contents)

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
            for x in item_emb:
                item_embeddings.append(x.cpu().detach().numpy())

            # n_queries x batch_size
            # i_rate_batch = torch.matmul(query_embeddings, item_emb.t()).detach().cpu()

            # rate_batch[:, i_start:i_end] = i_rate_batch
            i_count += item_emb.shape[0]

        assert i_count == n_items
        # item_embeddings = torch.cat(item_embeddings, dim=0)
        # print(item_embeddings.shape)

        table_dataset = table_dataset.add_column("embeddings", item_embeddings)

        if self.trainer.state.stage == 'test' and self.global_rank==0:
            # Save the dataset
            save_path = os.path.join(self.config.results_path, 'step_{}'.format(self.global_step))
            create_dirs([save_path])
            table_dataset_path = os.path.join(save_path, "table_dataset")
            table_dataset.save_to_disk(table_dataset_path)

        hidden_size = item_embeddings[0].shape[-1]
        print("hidden size", hidden_size)

        if "exhaustive_search_in_testing" in self.config.model_config.modules:
            index = faiss.IndexFlatIP(hidden_size)
        else:
            index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)
        
        # in testing mode, save the generated embeddings
        if self.trainer.state.stage == 'test' and self.global_rank==0:
            
            save_path = os.path.join(self.config.results_path, 'step_{}'.format(self.global_step))
            create_dirs([save_path])
            
            index_path = os.path.join(save_path, "table_dataset_hnsw_index.faiss")
            logger.info(f'saving embedding files into {index_path}')
            dataset_copy = copy.deepcopy(table_dataset)
            to_save_index = faiss.IndexHNSWFlat(hidden_size, 128, faiss.METRIC_INNER_PRODUCT)
            dataset_copy.add_faiss_index("embeddings", custom_index=to_save_index)
            dataset_copy.get_index("embeddings").save(index_path)
            
        table_dataset.add_faiss_index("embeddings", custom_index=index)

        # save to tmp variables
        self.tmp_table_dataset = table_dataset


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
