# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import base64

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging
logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
from utils.cache_system import save_cached_data, load_cached_data

from data_loader_manager.module_parser import ModuleParser
from .WikiTQ_dataset import WikiTQDataset


class WikiSQLDataset(WikiTQDataset):
    """
    Base WikiSQL dataset class
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)


class ITRRAGWikiSQLDataset(WikiSQLDataset):
    """
    This dataset class is used for RAG-like ITR Generation
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)

    def collate_fn(self, batch):
        '''
        when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
        a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        '''
        batched_data = super().collate_fn(batch)
        sub_tables = []
        gold_columns = [sample.gold_columns for sample in batch]
        gold_rows = [sample.get('gold_rows', []) for sample in batch]
        batched_data['gold_columns'] = gold_columns
        batched_data['gold_rows'] = gold_rows
        batched_data['valid'] = [sample.get("valid", True) for sample in batch]

        for sample in batch:
            sub_table_list = []
            for sub_table in sample.positive_sub_tables+sample.negative_sub_tables:
                sub_table_list.append(dict(sub_table))
            sub_tables.append(sub_table_list)
        
        batched_data['sub_tables'] = sub_tables
        return batched_data


class ITRWikiSQLDataset(WikiSQLDataset):
    """
    Base WikiSQL dataset class for Inner Table Retrieval
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)

        if self.mode == 'train' and self.config.mode == 'train':
            # in training, we ignore the samples without negative items
            self.dataset = self.dataset.filter(lambda x: x["num_positive_sub_tables"] != 0 and x["num_negative_sub_tables"] != 0)
    

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if len(sample['positive_sub_tables']) == 0:
            # we have dropped samples that do not have positive sub tables in the __init__ function for training
            # therefore this line specifically address the bug in row-wise splitting:
            # some questions do not have a gold row, therefore we just randomly pick one to bypass the error (since we need to have all samples in validation/test)
            pos_item = random.sample(sample['negative_sub_tables'], 1)[0]
        else:
            pos_item = random.sample(sample['positive_sub_tables'], 1)[0]
        
        num_neg_samples = self.config.model_config.num_negative_samples

        if len(sample['negative_sub_tables']) == 0:
            # in testing mode, we need to use dummy subtables to create the dataset sample
            # even though we don't use them
            neg_items = [pos_item] * num_neg_samples
            # simply copy pos_item here
        else:
            if num_neg_samples <= len(sample['negative_sub_tables']):
                # sample without replacement
                neg_items = random.sample(sample['negative_sub_tables'], num_neg_samples)
            else:
                # sample with replacement to avoid errors
                neg_items = random.choices(sample['negative_sub_tables'], k=num_neg_samples)
        sample['pos_item'] = pos_item
        sample['neg_items'] = neg_items

        return EasyDict(sample)

    def collate_fn(self, batch):
        '''
        when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
        a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        '''
        # According to the settings in config file, prepare the input and output
        input_modules = self.config.model_config.input_modules.module_list
        decoder_input_modules = self.config.model_config.decoder_input_modules.module_list
        output_modules = self.config.model_config.output_modules.module_list
        
        input_data = EasyDict()
        decoder_input_data = EasyDict()
        pos_item_data = EasyDict()
        neg_item_data = EasyDict()
        output_data = EasyDict()

        #############################
        #  User ModuleParser to parse every sample
        #       according to what modules are selected
        #       modules are parsed in order
        #############################
        for sample in batch:
            parsed_data = self.parse_modules(sample, input_modules, type='input')
            for key, value in parsed_data.items():
                input_data.setdefault(key, []).append(value)
            

            # One positive sample + Multiple negative samples
            ###### For the positive passage, generate input #######
            new_sample = EasyDict(sample.copy())
            new_sample.table = sample.pos_item
            parsed_data = self.parse_modules(new_sample, decoder_input_modules, type='decoder_input')
            for key, value in parsed_data.items():
                decoder_input_data.setdefault(key, []).append(value)
                pos_item_data.setdefault(key, []).append(value)
            
            for neg_item in sample.neg_items:
                ###### For each negative table, generate input #######
                new_sample = EasyDict(sample.copy())
                new_sample.table = neg_item
                
                parsed_data = self.parse_modules(new_sample, decoder_input_modules, type='decoder_input')
                for key, value in parsed_data.items():
                    decoder_input_data.setdefault(key, []).append(value)
                    neg_item_data.setdefault(key, []).append(value)
            

            parsed_data = self.parse_modules(sample, output_modules, type='output')
            for key, value in parsed_data.items():
                output_data.setdefault(key, []).append(value)

        
        input_data = EasyDict(input_data)
        decoder_input_data = EasyDict(decoder_input_data)
        output_data = EasyDict(output_data)
        
        #############################
        #  Postprocessing Features
        #############################
        input_post_modules = self.config.model_config.input_modules.postprocess_module_list
        decoder_input_post_modules = self.config.model_config.decoder_input_modules.postprocess_module_list
        output_post_modules = self.config.model_config.output_modules.postprocess_module_list
        
        input_data = self.post_processing(input_data, input_post_modules)
        decoder_input_data = self.post_processing(decoder_input_data, decoder_input_post_modules)
        output_data = self.post_processing(output_data, output_post_modules)
        
        #############################
        #  Meta Features
        #############################
        question_ids = [sample.question_id for sample in batch]
        questions = [sample.question for sample in batch]
        answers = [sample.answers for sample in batch]
        gold_columns = [sample.gold_columns for sample in batch]
        gold_rows = [sample.get('gold_rows', []) for sample in batch]
        sub_tables = []
        for sample in batch:
            sub_table_list = []
            for sub_table in sample.positive_sub_tables+sample.negative_sub_tables:
                sub_table_list.append(dict(sub_table))
            sub_tables.append(sub_table_list)
        
        batched_data = {
            'question_ids': question_ids,
            'questions': questions,
            'answers': answers,
            'sub_tables': sub_tables,
            'gold_columns': gold_columns,
            'gold_rows': gold_rows,
        }

        batched_data.update(input_data)
        batched_data.update(decoder_input_data)
        batched_data.update(output_data)

        return batched_data
