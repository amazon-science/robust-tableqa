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
import wandb
import glob
import tarfile
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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from utils.metrics_log_callback import MetricsHistoryLogger

from .metrics_processors import MetricsProcessor
from utils.dirs import *

class BaseExecutor(pl.LightningModule, MetricsProcessor):
    additional_plugins = []
    
    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.data_loader = data_loader
        self.train_dataloaders = list(self.data_loader.data_loaders['train'].values())
        self.valid_dataloaders = list(self.data_loader.data_loaders['valid'].values())
        self.test_dataloaders = list(self.data_loader.data_loaders['test'].values())

        logger.info(f'Initializing {self.__class__.__name__}...')
        
        # label smoother imported from huggingface transformers
        label_smoothing_factor = self.config.train.additional.get('label_smoothing_factor', 0)
        if label_smoothing_factor != 0:
            from transformers.trainer_pt_utils import LabelSmoother
            self.label_smoother = LabelSmoother(epsilon=label_smoothing_factor)
        else:
            self.label_smoother = None
        
    
    def setup(self, stage):
        """
        set loggers as class attributes for easy access
        """
        for trainer_logger in self.trainer.loggers:
            if type(trainer_logger) == TensorBoardLogger:
                self.tb_logger = trainer_logger
            elif type(trainer_logger) == WandbLogger:
                self.wandb_logger = trainer_logger
                self.wandb_logger.watch(self.model, log_freq=500, log_graph=False)
            elif type(trainer_logger) == MetricsHistoryLogger:
                self.metrics_history_logger = trainer_logger
            else:
                logger.warning(f'Unsupported logger type: {type(trainer_logger)}')

    def log_max_and_min_metrics(self):
        """
        Log the max and min metrics for each metric automatically
        """
        for metric_name, metric_values in self.metrics_history_logger.history.items():
            if metric_name in ['epoch', 'loss_step', 'loss_epoch', 'loss']:
                continue
            if len(metric_values) > 0 and type(metric_values[0]) in [float, int, np.float64]:
                self.log(f'{metric_name}_auto_max', float(max(metric_values)), on_step=False, on_epoch=True)
                self.log(f'{metric_name}_auto_min', float(min(metric_values)), on_step=False, on_epoch=True)
    
    
    def on_train_epoch_end(self):
        if self.global_rank == 0:
            self.log_max_and_min_metrics()

    def on_fit_end(self):
        if self.global_rank == 0:
            pass

    def on_test_end(self):
        if self.global_rank == 0:
            pass
    

    def train_dataloader(self):
        self.train_dataloader_names = list(self.data_loader.data_loaders['train'].keys())
        
        # TODO: we only allow one train data loader at the moment
        return self.train_dataloaders[0]
    
    def val_dataloader(self):
        self.val_dataloader_names = list(self.data_loader.data_loaders['valid'].keys())

        return self.valid_dataloaders
    
    def test_dataloader(self):
        self.test_dataloader_names = list(self.data_loader.data_loaders['test'].keys())
        
        
        return self.test_dataloaders


    def on_exception(self, trainer, pl_module, exception):
        # handle exception
        
        if self.wandb_logger and trainer.is_global_zero:
            if self.wandb_logger.experiment is not None:
                logger.error(f"Attempting to stop the wandb run {self.wandb_logger.experiment}")
                self.wandb_logger.experiment.finish()
        

    def forward(self, **kwargs):
        return self.model(**kwargs)
