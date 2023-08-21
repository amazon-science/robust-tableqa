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

# For TAPAS model
from transformers import TapasTokenizer, TapasConfig, TapasForQuestionAnswering
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model

from .metrics_processors import MetricsProcessor
from .base_executor import BaseExecutor
from utils.dirs import *

class TAPASExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer
        
        ModelClass = globals()[self.config.model_config.ModelClass]
        ConfigClass = globals()[self.config.model_config.ConfigClass]
        model_config = ConfigClass.from_pretrained(self.config.model_config.ConfigModelVersion)
        pprint(model_config)

        if self.config.model_config.pretrained == True:
            self.model = ModelClass.from_pretrained(self.config.model_config.ModelVersion,
                                                    config=model_config)
        else:
            self.model = ModelClass(model_config, config=model_config)
        
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
        self.optimizer = torch.optim.Adam(
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
            'token_type_ids': sample_batched['token_type_ids'].to(self.device),
            'labels': sample_batched['labels'].to(self.device),
            'numeric_values': sample_batched['numeric_values'].to(self.device),
            'numeric_values_scale': sample_batched['numeric_values_scale'].to(self.device),
            'float_answer': sample_batched['float_value'].to(self.device),
        })

        forward_results = self.model(**train_batch)
        
        # if label smoother is defined, pass the logits and labels to the smoother
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
        return self._prediction_step(sample_batched, batch_idx)

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
        return self._prediction_step(sample_batched, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        for i in range(len(self.test_dataloader())):
            if len(self.test_dataloader()) == 1:
                test_step_output = test_step_outputs
            else:
                test_step_output = test_step_outputs[i]
            
            log_dict = self.evaluate_outputs(test_step_output)
            self.logging_results(log_dict, prefix=f"{self.config.test.evaluation_name}_{self.test_dataloader_names[i]}")

        return None

    def _prediction_step(self, sample_batched, batch_idx):
        """
        This function is shared by both valid and test
        """
        predictions = []
        table_entries = []

        test_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].to(self.device),
            'attention_mask': sample_batched['attention_mask'].to(self.device),
            'token_type_ids': sample_batched['token_type_ids'].to(self.device),
        })
        
        outputs = self.model(**test_batch)
        
        convert_batch = EasyDict({
            'input_ids': sample_batched['input_ids'].cpu(),
            'attention_mask': sample_batched['attention_mask'].cpu(),
            'token_type_ids': sample_batched['token_type_ids'].cpu(),
        })
        predicted_answer_coordinates, predicted_aggregation_indices = self.tokenizer.convert_logits_to_predictions(
            convert_batch, outputs.logits.cpu().detach(), outputs.logits_aggregation.cpu().detach()
        )

        # let's print out the results:
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

        tables = sample_batched['tables']
        input_text_sequences = sample_batched['input_text_sequences']
        answers = []
        for index, coordinates in enumerate(predicted_answer_coordinates):
            table = tables[index]
            if len(coordinates) == 1:
                # only a single cell:
                answer = table.iat[coordinates[0]]
                answers.append(answer)
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(tables[index].iat[coordinate])
                answers.append(", ".join(cell_values))
        
        question_ids = sample_batched['question_ids']
        gold_answers = sample_batched['answers']
        valids = sample_batched['valid']

        for question_id, gold_answer, query, answer, predicted_agg, valid in zip(question_ids, gold_answers, input_text_sequences, answers, aggregation_predictions_string, valids):
            # Transform aggregation to a final prediction
            if predicted_agg == "SUM":
                all_cell_values = answer.split(', ')
                all_cell_values = [cell_value.strip() for cell_value in all_cell_values]
                try:
                    all_cell_values = [float(cell_value) for cell_value in all_cell_values]
                    final_pred = sum(all_cell_values)
                except Exception as e:
                    final_pred = 0.0
            elif predicted_agg == 'AVERAGE':
                all_cell_values = answer.split(', ')
                all_cell_values = [cell_value.strip() for cell_value in all_cell_values]
                try:
                    all_cell_values = [float(cell_value) for cell_value in all_cell_values]
                    final_pred = sum(all_cell_values)/len(all_cell_values)
                except Exception as e:
                    final_pred = 0.0
            elif predicted_agg == 'COUNT':
                all_cell_values = answer.split(', ')
                final_pred = len(all_cell_values)
            else:
                final_pred = answer
            
            predictions.append({
                'question_id': question_id,
                'decoded_output': answer,
                'predicted_agg': predicted_agg,
                'decoded_label': ', '.join(gold_answer),
                'final_pred': final_pred,
                'valid': valid,
            })
            
            table_entry = [
                question_id,
                query,
                gold_answer,
                answer,
                predicted_agg,
                str(final_pred),
            ]
            table_entries.append(table_entry)
            # print(query)
            # if predicted_agg == "NONE":
            #     print("Predicted answer: " + answer)
            # else:
            #     print("Predicted answer: " + predicted_agg + " > " + answer)
        


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

        columns=["question_id", "question", "answers", "predicted cells", "predicted aggregation", "final prediction"]
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
