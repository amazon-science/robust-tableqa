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


class E2EWTQDataset(WikiTQDataset):
    """
    Base NQTables dataset class
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.tables = self.data['table_dataset'].to_pandas().set_index("table_id").to_dict(orient="index")

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
            
            parsed_data = self.parse_modules(sample, decoder_input_modules, type='decoder_input')
            for key, value in parsed_data.items():
                decoder_input_data.setdefault(key, []).append(value)

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
        batched_data = {}

        question_ids = [sample.question_id for sample in batch]
        questions = [sample.question for sample in batch]
        answers = [sample.answers for sample in batch]
        pos_item_ids = [sample.pos_item_id for sample in batch]
        
        batched_data.update({
            'question_ids': question_ids,
            'questions': questions,
            'answers': answers,
            'pos_item_ids': pos_item_ids,
        })

        batched_data.update(input_data)
        batched_data.update(decoder_input_data)
        batched_data.update(output_data)
        
        return batched_data
        
class RAGE2EWTQDataset(E2EWTQDataset):
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

        return batched_data


class DPRE2EWTQDataset(E2EWTQDataset):
    """
    Base NQTables dataset class for Dense Passage Retrieval
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.table_ids = list(self.tables.keys())

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = EasyDict(sample)

        def negative_sampling(sample, num_samples=1):
            """Generate negative samples for a query. ONLY used in training
            Args:
                sample: the target
                num_samples (int, optional): number of samples. Defaults to 1.
            Returns:
                neg_items: list of negative tables.
            """
            neg_items = []
            question_id = sample.question_id
            pos_item = self.tables[sample.pos_item_id]
            answers = sample.answers
            
            while len(neg_items) < num_samples:
                # sample num_samples negative items for the user
                question_id = str(question_id)
                while True:
                    neg_item_id = random.sample(self.table_ids, 1)[0]
                    
                    VALID = True
                    if neg_item_id == sample.pos_item_id:
                        VALID = False
                    
                    if VALID == True:
                        break
                neg_items.append(self.tables[neg_item_id])
            return neg_items
        
        pos_item = self.tables[sample.pos_item_id]

        num_neg_samples = self.config.model_config.num_negative_samples

        # We set a ratio here. With some prob the negative samples are retrieved from all samples
        # With some prob the negative samples are harvested from the BM25-retrieved documents
        bm25_ratio = self.config.model_config.bm25_ratio
        bm25_top_k = self.config.model_config.bm25_top_k

        if random.uniform(0, 1) < bm25_ratio:
            valid_ids = [i for i in sample.bm25_retrieved_item_ids if i != sample.pos_item_id]
            if bm25_top_k > 0:
                valid_ids = valid_ids[:bm25_top_k]
            neg_item_ids = random.sample(valid_ids, num_neg_samples)
            neg_items = [self.tables[i] for i in neg_item_ids]
        else:
            neg_items = negative_sampling(sample, num_neg_samples)
        
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
        pos_item_ids = [[sample.pos_item_id] for sample in batch]

        batched_data = {
            'question_ids': question_ids,
            'questions': questions,
            'answers': answers,
            'pos_item_ids': pos_item_ids,
        }

        batched_data.update(input_data)
        batched_data.update(decoder_input_data)
        batched_data.update(output_data)

        return batched_data
