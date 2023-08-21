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

import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

# For ColBERT model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from functools import partial
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from datasets import Features, Sequence, Value, load_dataset, Dataset

from .metrics_processors import MetricsProcessor
from .base_executor import BaseExecutor
from utils.dirs import *
import torch.distributed as dist


class ColBERTExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer
        
        ModelClass = globals()[self.config.model_config.ModelClass]

        if '$' in self.config.model_config.EncoderModelVersion:
            self.config.model_config.EncoderModelVersion = os.path.join(self.config.DATA_FOLDER, self.config.model_config.EncoderModelVersion.replace('$', ''))
        
        colbert_config = ColBERTConfig(
            bsize=None,
            use_ib_negatives=True,
            checkpoint=self.config.model_config.EncoderModelVersion,
            rank=self.global_rank,
        )
        colbert_config.checkpoint = colbert_config.checkpoint or 'bert-base-uncased'
        
        self.model = ColBERT(name=colbert_config.checkpoint, colbert_config=colbert_config)
        self.model.colbert_config.nway = self.config.model_config.num_negative_samples + 1
        self.model.raw_tokenizer = data_loader.decoder_tokenizer.tok

        # Resize the bert embedding space to accommodate special tokens
        logger.info(f'tokenizer lengths = {len(self.tokenizer.tok)} and {len(self.decoder_tokenizer.tok)}')
        self.model.bert.resize_token_embeddings(
            max(len(self.tokenizer.tok), len(self.decoder_tokenizer.tok))
        )
        self.model.skiplist.update({w: True
                             for symbol in self.config.model_config.DECODER_SPECIAL_TOKENS["additional_special_tokens"]
                             for w in [symbol, self.model.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]})
        pprint(self.model.skiplist)
        
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
            'Q': (
                sample_batched['input_ids'].to(self.device),
                sample_batched['attention_mask'].to(self.device)
            ),
            "D": (
                sample_batched['decoder_input_ids'].to(self.device),
                sample_batched['decoder_input_attention_mask'].to(self.device)
            ),
            # 'labels': sample_batched['labels'].to(self.device),
        }

        scores = self.model(**train_batch)

        config = self.model.colbert_config
        if config.use_ib_negatives:
            scores, ib_loss = scores
            loss = ib_loss
        else:
            scores = scores.view(-1, config.nway)
            labels = torch.zeros(sample_batched['input_ids'].shape[0]*dist.get_world_size(), dtype=torch.long, device=self.device)
            loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])
        
        
        batch_loss = loss

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
        self.log("train/ib_loss", ib_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
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
        test_batch = (
            sample_batched['input_ids'].to(self.device),
            sample_batched['attention_mask'].to(self.device)
        )
        # batch_size x hidden_states
        query_emb = self.model.query(*test_batch)

        data_to_return = {
            'btach_idx': batch_idx,
            'query_emb': query_emb.detach().cpu(),
            'question_ids': sample_batched['question_ids'],
            'questions': sample_batched['questions'],
            'answers': sample_batched['answers'],
            'pos_item_ids': sample_batched['pos_item_ids'],
        }

        return data_to_return
    


    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name, mode='test'):
        # Batching every validation step outputs
        query_embeddings = []
        question_ids = []
        questions = []
        pos_item_ids = []

        for step_output in step_outputs:
            query_embeddings.append(step_output['query_emb'])
            questions.extend(step_output['questions'])
            for question_id in step_output['question_ids']:
                question_ids.append(question_id)
            pos_item_ids.extend(step_output['pos_item_ids'])

        # question_ids = [0, 1, 2, ...]
        query_embeddings = torch.cat(query_embeddings, dim=0)
        
        ##################################
        ##    Prepare Index File        ##
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

        # Search the index and process results
        Ks = self.config.model_config.Ks
        logger.info(f"searching for {query_embeddings.shape} queries")

        if "exhaustive_search_in_testing" not in self.config.model_config.modules:
            with Run().context(RunConfig(nranks=1, rank=self.global_rank, root=self.config.saved_model_path, experiment="temp_index")):

                config = ColBERTConfig(
                    total_visible_gpus=0,
                )
                searcher = Searcher(index="temp_index.nbits=2", config=config)
                custom_quries = {question_id: question for question_id, question in zip(question_ids, questions)}
                queries = Queries(data=custom_quries)
                ranking = searcher._search_all_Q(queries, query_embeddings, k=max(Ks))
                
                ranking_dict = ranking.todict()

                torch.distributed.barrier()

                del searcher
        else:
            # exhaustive search
            ranking_dict = {}
            self.model.eval()

            item_embeddings = self.item_embeddings
            item_embedding_mask = self.item_embedding_mask

            n_items = len(item_embeddings)
            logger.info(f"n_items {n_items}")

            i_batch_size = self.config[mode].batch_size
            n_item_batchs = n_items // i_batch_size + 1

            rate_batch = torch.zeros((len(query_embeddings), n_items))
            print("rate_batch", rate_batch.shape)
            for i_batch_id in tqdm(range(n_item_batchs)):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)
                if i_end - i_start == 0:
                    break

                retrieved_item_embeddings = np.stack(item_embeddings[i_start:i_end])
                retrieved_item_embedding_mask = np.stack(item_embedding_mask[i_start:i_end])
                retrieved_item_embeddings = torch.from_numpy(retrieved_item_embeddings).to(self.device)
                retrieved_item_embedding_mask = torch.from_numpy(retrieved_item_embedding_mask).to(self.device)
                current_i_size = len(retrieved_item_embeddings)

                # self.model.colbert_config.nway = current_i_size
                Q_duplicated = query_embeddings.repeat_interleave(current_i_size, dim=0).contiguous().to(self.device)
                retrieved_item_embeddings = retrieved_item_embeddings.repeat(len(query_embeddings), 1, 1)
                retrieved_item_embedding_mask = retrieved_item_embedding_mask.repeat(len(query_embeddings), 1, 1)
                # print("Q_duplicated", Q_duplicated.shape)
                # print("retrieved_item_embeddings", retrieved_item_embeddings.shape)
                scores = self.model.score(Q_duplicated, retrieved_item_embeddings, retrieved_item_embedding_mask)
                scores = scores.reshape(len(query_embeddings), -1)
                rate_batch[:, i_start:i_end] = scores.cpu()
            
            logger.info("sorting...")
            sorted_scores, indices = torch.sort(rate_batch.to(self.device), dim=-1, descending=True)
            sorted_scores = sorted_scores[:, :max(Ks)].cpu()
            indices = indices[:, :max(Ks)].cpu()
            for query_index in range(len(query_embeddings)):
                table_indices = indices[query_index]
                table_scores = sorted_scores[query_index]
                ranking_list = [
                    (table_indices[i].numpy(), i, table_scores[i].numpy()) for i in range(max(Ks))
                ]
                ranking_dict[query_index] = ranking_list

            # Finally, restore the nway
            # self.model.colbert_config.nway = self.config.model_config.num_negative_samples + 1
        
        batch_results = []

        # Log results
        columns=["question_id", "pos_item_ids"]  \
                    + ['p_{}'.format(i) for i in range(max(Ks))]
        test_table = wandb.Table(columns=columns)

        for question_id, pos_item_id, ranking_list in zip(question_ids, pos_item_ids, ranking_dict.values()):

            retrieved_tables_sorted = []
            score = []
            retrieved_table_indices = []
            for entry in ranking_list:
                retrieved_table_index, _, retrieved_table_score =  entry
                retrieved_table_indices.append(retrieved_table_index)
                score.append(retrieved_table_score)
            
            max_K = max(Ks)
            if len(ranking_list) < max_K:
                # normally happens in sanity check
                # the number of documents may be less than max_K
                # this is because the system relies on centroids to retrieve items
                # therefore it is not guaranteed to have enough documents retrieved
                # In this case, we simply replicate the last element to avoid crash
                retrieved_table_indices += [retrieved_table_indices[-1]] * (max_K-len(ranking_list))
                score += [score[-1]] * (max_K-len(ranking_list))
                
            retrieved_tables_sorted = table_dataset.select(retrieved_table_indices)['table_id']

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
                table_entry+=[f"{retrieved_table_id}, {retrieved_table_id in pos_item_id}, {retrieved_table_score}"]
            
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
        exhaustive_search_in_testing = "exhaustive_search_in_testing" in self.config.model_config.modules
        # Move model to cpu to save some memory
        self.model = self.model.to("cpu")
        
        if self.global_rank == 0 and not exhaustive_search_in_testing:
            logger.info(f"Global rank {self.global_rank} starts indexing job...")

            # First, we need to save the model checkpoint so that we can run the index
            # Currently ColBERT does not provide on-the-fly indexer
            if self.trainer.state.stage == 'test':
                tmp_model_path = os.path.join(
                    self.config.saved_model_path,
                    f"test_temp_model",
                )
            else:
                tmp_model_path = os.path.join(
                    self.config.saved_model_path,
                    f"validation_temp_model",
                )
            logger.info(f"saving temp model to {tmp_model_path}...")
            self.model.save(tmp_model_path)
            
            # Launch indexer
            with Run().context(RunConfig(nranks=8, root=self.config.saved_model_path, experiment="temp_index")):
                nbits = self.config.model_config.get("nbits", 2)
                config = ColBERTConfig(
                    nbits=nbits,
                    doc_maxlen=self.config.data_loader.additional.max_decoder_source_length,
                )
                print("indexing with", nbits, "bits")
                custom_collection = table_contents
                indexer = Indexer(checkpoint=tmp_model_path, config=config)
                indexer.index(name="temp_index.nbits=2", collection=custom_collection, overwrite=True)
                index_path = indexer.get_index()
                del indexer
        else:
            logger.info(f"Global rank {self.global_rank} waits for Rank 0...")
        
        # Use barrrier to sync all ranks. Only when Rank 0 finishes indexing, other ranks will move on
        torch.distributed.barrier()
        torch.cuda.empty_cache()

        self.model = self.model.to(self.device)

        if (self.trainer.state.stage == 'test' and self.global_rank==0) or (exhaustive_search_in_testing):
            # Add embeddings
            # For easier implementation, we just rerun the item embedding generation here
            i_count = 0
            item_embeddings = []
            item_embedding_mask = []
            for i_batch_id in tqdm(range(n_item_batchs)):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)
                if i_end - i_start == 0:
                    break
                passage_contents_batch = table_contents[i_start:i_end]
                
                # Encode this batch of data
                item_input_ids, item_attention_mask = self.data_loader.decoder_tokenizer.tensorize(passage_contents_batch)
            
                test_batch = (
                    item_input_ids.to(self.device),
                    item_attention_mask.to(self.device),
                )

                # batch_size x hidden_states
                item_emb, item_emb_mask = self.model.doc(*test_batch, keep_dims='return_mask')
                
                for x in item_emb:
                    item_embeddings.append(x.cpu().detach().numpy())
                for x in item_emb_mask:
                    item_embedding_mask.append(x.cpu().detach().numpy())
                # n_queries x batch_size
                # i_rate_batch = torch.matmul(query_embeddings, item_emb.t()).detach().cpu()

                # rate_batch[:, i_start:i_end] = i_rate_batch
                i_count += item_emb.shape[0]

            assert i_count == n_items
            logger.info("finished generating item embeddings...")
            # table_dataset = table_dataset.add_column("embeddings", item_embeddings)
            # table_dataset = table_dataset.add_column("embedding_mask", item_embedding_mask)

            if self.trainer.state.stage == 'test':
                # Save the dataset
                save_path = os.path.join(self.config.results_path, 'step_{}'.format(self.global_step))
                create_dirs([save_path])
                table_dataset_path = os.path.join(save_path, "table_dataset")
                table_dataset.save_to_disk(table_dataset_path)

                embedding_save_path = os.path.join(save_path, "item_embeddings.pkl")
                logger.info(f"saving embedding files to {embedding_save_path}")
                with open(embedding_save_path, 'wb') as f:
                    pickle.dump({
                        "item_embeddings": item_embeddings,
                        "item_embedding_mask": item_embedding_mask,
                    }, f)
            
            self.item_embeddings = item_embeddings
            self.item_embedding_mask = item_embedding_mask
            

        # Sync all processes. If rank 0 starts saving item embeddings in testing, other processes will wait for it.
        torch.distributed.barrier()

        # save to tmp variables
        self.tmp_table_dataset = table_dataset

        # in testing mode, copy the temp index files
        if self.trainer.state.stage == 'test' and self.global_rank==0 and not exhaustive_search_in_testing:
            save_path = os.path.join(self.config.results_path, 'step_{}'.format(self.global_step))
            create_dirs([save_path])
            
            index_path_from = os.path.join(
                self.config.saved_model_path, "temp_index"
            )
            index_path_to = os.path.join(save_path, "table_dataset_colbert_index")
            logger.info(f'saving index files from {index_path_from} into {index_path_to}')
            import shutil
            if os.path.exists(index_path_to):
                # Delete if the index already exists
                shutil.rmtree(index_path_to)

            shutil.copytree(index_path_from, index_path_to)
        
        # sync all processes
        torch.distributed.barrier()

        

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
        self.model.save(path_save_model)
        logger.info('Model has been saved to {}'.format(path_save_model))
