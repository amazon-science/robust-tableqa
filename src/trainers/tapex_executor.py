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

class TAPEXExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer
        
        ModelClass = globals()[self.config.model_config.ModelClass]

        if self.config.model_config.pretrained == True:
            ConfigClass = globals()[self.config.model_config.ConfigClass]
            model_config = ConfigClass.from_pretrained(self.config.model_config.ModelVersion)
            self.model = ModelClass.from_pretrained(self.config.model_config.ModelVersion,
                                                    config=model_config)
        else:
            ConfigClass = globals()[self.config.model_config.ConfigClass]
            model_config = ConfigClass.from_pretrained(self.config.model_config.ModelVersion)
            self.model = ModelClass(model_config)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    
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
        train_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
            'labels': sample_batched['labels'].to(self.device),
        })

        forward_results = self.model(**train_batch)
        # batch_loss = forward_results.loss

        if self.label_smoother is not None:
            if unwrap_model(self.model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                batch_loss = self.label_smoother(forward_results, train_batch.labels, shift_labels=True)
            else:
                batch_loss = self.label_smoother(forward_results, train_batch.labels)
        else:
            batch_loss = forward_results.loss
        
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
        return self._generative_step(sample_batched, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):

        for i in range(len(self.val_dataloader())):
            if len(self.val_dataloader()) == 1:
                validation_step_output = validation_step_outputs
            else:
                validation_step_output = validation_step_outputs[i]
            
            log_dict = self.evaluate_outputs(validation_step_output)
            self.logging_results(log_dict, prefix=self.val_dataloader_names[i])
        return None
    
    def test_step(self, sample_batched, batch_idx, dataloader_idx=0):
        return self._generative_step(sample_batched, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        for i in range(len(self.test_dataloader())):
            if len(self.test_dataloader()) == 1:
                test_step_output = test_step_outputs
            else:
                test_step_output = test_step_outputs[i]
            
            log_dict = self.evaluate_outputs(test_step_output)
            self.logging_results(log_dict, prefix=f"{self.config.test.evaluation_name}_{self.test_dataloader_names[i]}")

        return None

    def _generative_step(self, sample_batched, batch_idx):
        """
        This function is shared by both valid and test
        """
        predictions = []
        table_entries = []

        labels = sample_batched['labels']
        test_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
            "max_length": self.config.data_loader.additional.max_target_length,
            "num_beams": self.config.model_config.get('num_beams', 1)
        })

        # print(f'running generative step {test_batch.input_ids.shape} {test_batch.attention_mask.shape} {labels.shape}  {sample_batched["question_ids"]}')
        
        outputs = self.model.generate(**test_batch)

        bos_token_id = self.data_loader.decoder_tokenizer.bos_token_id
        for index, i in enumerate(labels):

            cleaned_i = [label if label!=-100 else self.decoder_tokenizer.pad_token_id for label in i]
            cleaned_i = torch.LongTensor(cleaned_i)
            decoded_label = self.decoder_tokenizer.decode(cleaned_i, skip_special_tokens=True)
            # print(self.tokenizer.decode(cleaned_i, skip_special_tokens=True))
            
            output_sequence = outputs[index]
            output_sequence = output_sequence.cpu().numpy().tolist()

            if bos_token_id in output_sequence:
                output_sequence = output_sequence[output_sequence.index(bos_token_id):]

            # print('output_sequence after', output_sequence)
            decoded_output = self.decoder_tokenizer.decode(output_sequence, skip_special_tokens=True)
            actual_output = self.decoder_tokenizer.decode(output_sequence, skip_special_tokens=False)
            # print(self.tokenizer.decode(cleaned_i, skip_special_tokens=True))
            
            if batch_idx < 1:
                print(decoded_label, '<--->', decoded_output, '   ({})'.format(actual_output))
            
            question_id = sample_batched['question_ids'][index]
            predictions.append({
                'question_id': question_id,
                'decoded_output': decoded_output,
                'decoded_label': decoded_label,
            })

            table_entry = [
                question_id,
                sample_batched['questions'][index],
                sample_batched['answers'][index],
                decoded_output,
            ]
            table_entries.append(table_entry)

        
        data_to_return = {
            'predictions': predictions,
            'outputs': outputs,
            'question_ids': sample_batched['question_ids'],
            'answers': sample_batched['answers'],
            'table_entries': table_entries,
        }


        return data_to_return
    
    


    def evaluate_outputs(self, step_outputs, mode='test'):
        # Batching every validation step outputs
        batch_predictions = []
        batch_answers = []

        columns=["question_id", "question", "answers", "prediction"]
        test_table = wandb.Table(columns=columns)

        for step_output in step_outputs:
            batch_predictions += step_output['predictions']
            batch_answers += step_output['answers']
            for table_entry in step_output['table_entries']:
                test_table.add_data(*table_entry)
        
        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_predictions=batch_predictions,
            batch_answers=batch_answers,
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
