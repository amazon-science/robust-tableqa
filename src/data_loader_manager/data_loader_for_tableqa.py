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
import gzip

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tfrecord.torch.dataset import TFRecordDataset
from datasets import load_dataset, load_from_disk, Dataset
from itertools import combinations

import logging
logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
from utils.cache_system import save_cached_data, load_cached_data
from utils.tapas_utils import parse_tapas_question, prepare_table_in_sqa_format
from utils.wikisql_utils import _TYPE_CONVERTER, retrieve_wikisql_query_answer_tapas
from utils.tapas_metrics_utils import _collect_cells_from_table, _parse_value
import utils.text_utils as text_utils

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from data_loader_manager.datasets import *

class DataLoaderForTableQA(DataLoaderWrapper):
    '''
    This is the data loader manager for TAPEX model
    Its task is to read data, create torch.utils.data.DataLoader, 
    and return them for the main program to use
    '''

    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)

        if 'TAPAS' in self.config.model_config.base_model:
            # This is a bug in the TAPAS's tokenizer
            # Since the token_type_ids.row_ids can not exceed 512 (defined in the model)
            # We set it here to avoid cuda errors
            self.tokenizer.max_row_id = 256
        

    def LoadWikiTQData(self, module_config):
        '''
        Load WikiTQ Corpus
        {
          "type": "LoadWikiTQData", "option": "default", 
          "config": {
                "preprocess": ["transform_to_sqa_format", "check_tapas_tokenization_compatibility"]
                "path":{
                    "train": "..",
                    "test": "..",
                },
                "tapas_path": {
                    "train": "..",
                    "test": "..",
                }
            },
        },
        '''

        """
        {'id': 'nu-0', 'question': 'which country had the most cyclists finish within the top 10?', 'answers': ['Italy'], 'table': {'header': ['Rank', 'Cyclist', 'Team', 'Time', 'UCI ProTour\\nPoints'], 'rows': [['1', 'Alejandro Valverde\xa0(ESP)', "Caisse d'Epargne", '5h 29\' 10"', '40'], ['2', 'Alexandr Kolobnev\xa0(RUS)', 'Team CSC Saxo Bank', 's.t.', '30'], ['3', 'Davide Rebellin\xa0(ITA)', 'Gerolsteiner', 's.t.', '25'], ['4', 'Paolo Bettini\xa0(ITA)', 'Quick Step', 's.t.', '20'], ['5', 'Franco Pellizotti\xa0(ITA)', 'Liquigas', 's.t.', '15'], ['6', 'Denis Menchov\xa0(RUS)', 'Rabobank', 's.t.', '11'], ['7', 'Samuel Sánchez\xa0(ESP)', 'Euskaltel-Euskadi', 's.t.', '7'], ['8', 'Stéphane Goubert\xa0(FRA)', 'Ag2r-La Mondiale', '+ 2"', '5'], ['9', 'Haimar Zubeldia\xa0(ESP)', 'Euskaltel-Euskadi', '+ 2"', '3'], ['10', 'David Moncoutié\xa0(FRA)', 'Cofidis', '+ 2"', '1']], 'name': 'csv/203-csv/733.tsv'}}
        """

        self.data.wtq_data = EasyDict({})

        datasets = None

        for split, split_file_path in module_config.config.path.items():
            split_file_path = os.path.join(self.config.DATA_FOLDER, split_file_path)

            is_training = (split in ['train'])

            if os.path.exists(split_file_path):
                dataset = load_from_disk(split_file_path)
            else:
                if not datasets:
                    datasets = load_dataset("wikitablequestions")
                column_names = datasets[split].column_names
                dataset = datasets[split]

                if 'use_original_tapas_data' in module_config.config.preprocess:
                    # we try to use the original tapas preprocessed inputs
                    tf_input_path = module_config.config.tapas_path[split]
                    tf_input_path = os.path.join(self.config.DATA_FOLDER, tf_input_path)
                    with open(tf_input_path, 'rb') as f:
                        logger.info(f'Loading tf_input files {tf_input_path}')
                        tf_dataset = pickle.load(f)

                    all_tf_data = {}
                    for example in tqdm(tf_dataset['examples']):
                        question_id = example['question_id'].replace('-0_0', '')
                        all_tf_data[question_id] = example
                    
                    def attach_tf_input_to_dataset(example):
                        question_id = example['id']
                        tf_input_example = all_tf_data.get(question_id, None)
                        if tf_input_example is not None:
                            example['answer_text'] = example['answers']
                            example['float_value'] = tf_input_example['answer']
                            example['aggregation_function'] = tf_input_example['aggregation_function_id']
                            # example['answer_coordinates'] = [(coord[1], coord[0]) for coord in tf_input_example['answer_coordinates']]
                            example['valid'] = True
                        else:
                            example['answer_text'] = example['answers']
                            example['float_value'] = None
                            example['aggregation_function'] = None
                            # example['answer_coordinates'] = None
                            example['valid'] = False
                        return example
                    
                    dataset = dataset.map(attach_tf_input_to_dataset, batched=False, desc="attach_tf_input_to_dataset")
                    
                    conversion_success = len(dataset.filter(lambda x: x["valid"] == True))
                    conversion_error = len(dataset.filter(lambda x: x["valid"] == False))
                    
                    # if is_training:
                    #     dataset = dataset.filter(lambda x: x["valid"] == True)

                    logger.info(f'conversion_success {conversion_success} / conversion_error {conversion_error} --> {conversion_success/(conversion_success+conversion_error)*100}%')


                if "transform_to_sqa_format" in module_config.config.preprocess:
                    # process to get TAPAS-relevant entries
                    def process_to_sqa_format(example):

                        table = example['table']
                        # Transform header/row format into a format that parse_tapas_question can take
                        
                        new_table = prepare_table_in_sqa_format(table)

                        compact_table_pd = pd.DataFrame.from_dict(new_table, orient='index', columns=table['header'])
                        
                        try:
                            question, answer_texts, answer_coordinates, float_value, aggregation_function = parse_tapas_question(table=compact_table_pd, question=example['question'], answer_texts=example['answers'])
                            
                            # print(question)
                            # print(answer_texts)
                            # print("Found coordinates:", answer_coordinates)
                            # print("Found float value:", float_value)

                            example['answer_text'] = answer_texts
                            example['float_value'] = float_value
                            example['aggregation_function'] = aggregation_function

                            # the answer_coordinates must be a single list of one or more tuples. Each tuple must be a (row_index, column_index) pair.
                            example['answer_coordinates'] = [(coord[1], coord[0]) for coord in answer_coordinates]

                            if answer_coordinates is not None:
                                example['valid'] = True
                                example['err_msg'] = ""
                            else:
                                example['valid'] = False
                                example['err_msg'] = "answer_coordinates is empty"

                        except Exception as e:
                            example['answer_text'] = example['answers']
                            example['float_value'] = None
                            example['aggregation_function'] = None
                            example['answer_coordinates'] = None
                            example['valid'] = False
                            example['err_msg'] = str(e)
                        
                        return example

                    dataset = dataset.map(process_to_sqa_format, batched=False, num_proc=4, desc="transform_to_sqa_format")
                    conversion_success = len(dataset.filter(lambda x: x["valid"] == True))
                    conversion_error = len(dataset.filter(lambda x: x["valid"] == False))

                    print(dataset.to_pandas().value_counts(subset=['err_msg']))

                    # keep only data that can be parsed (for only training sets)
                    # we will need to count these dropped samples in evaluation later
                    # so that the comparison is fair!
                    if is_training:
                        dataset = dataset.filter(lambda x: x["valid"] == True)

                    logger.info(f'conversion_success {conversion_success} / conversion_error {conversion_error} --> {conversion_success/(conversion_success+conversion_error)*100}%')
                

                if "check_tapas_tokenization_compatibility" in module_config.config.preprocess:
                    def test_encoding(example):
                        if example['valid'] == False:
                            # we don't check compatibility of invalid examples
                            # since they have been dropped in the WTQ-to-SQA conversion
                            return example
                        
                        table = example['table']
                        new_table = prepare_table_in_sqa_format(table)
                        
                        compact_table_pd = pd.DataFrame.from_dict(new_table, orient='index', columns=table['header'])

                        try:
                            if is_training:
                                # for training set, we have to ensure that all relevant fields are provided
                                # i.e. in supervision mode, if answer_text is provided, answer_coordinates can not be None
                                encoding = self.tokenizer(
                                    table=compact_table_pd,
                                    queries=example['question'],
                                    answer_coordinates=example['answer_coordinates'],
                                    answer_text=example['answer_text'],
                                    truncation=True,
                                    padding="max_length",
                                    return_tensors="pt",
                                )
                            else:
                                # for testing, we allow all data to appear, as long as there is no encoding error
                                encoding = self.tokenizer(
                                    table=compact_table_pd,
                                    queries=example['question'],
                                    truncation=True,
                                    padding="max_length",
                                    return_tensors="pt",
                                )
                        except Exception as e:
                            logger.warning(f'incompatible data: {e}')
                            example['valid'] = False
                        
                        return example

                    if is_training:
                        dataset = dataset.map(test_encoding, batched=False, num_proc=8, desc='check_tapas_tokenization_compatibility')
                    else:
                        logger.info('skip compatibility check for val/test sets...')
                    
                    conversion_success = len(dataset.filter(lambda x: x["valid"] == True))
                    conversion_error = len(dataset.filter(lambda x: x["valid"] == False))

                    # keep only data that can be parsed (for only training sets)
                    # we will need to count these dropped samples in evaluation later
                    # so that the comparison is fair!
                    if is_training:
                        dataset = dataset.filter(lambda x: x["valid"] == True)

                    logger.info(f'conversion_success {conversion_success} / conversion_error {conversion_error} --> {conversion_success/(conversion_success+conversion_error)*100}%')
                    

                if 'split_table_by_column_combination' in module_config.config.preprocess:
                    def split_table_by_column_combination(example):
                        # this function splits column "table" into "sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])

                        # add type to avoid bugs
                        example['table']['types'] = ['unknown']*num_columns
                        example['table']['id'] = example['table'].pop('name')

                        gold_col = []
                        example['gold_columns'] = gold_col

                        sub_tables = []

                        MAX_COLUMNS = 1
                        for max_column in range(1, MAX_COLUMNS+1):
                            for comb in combinations(list(range(num_columns)), max_column):
                                selected_columns = [table['header'][i] for i in comb]
                                selected_rows = [[row[i] for i in comb] for row in table['rows']]
                                selected_types = ['unknown']*len(selected_columns)
                                sub_table = deepcopy(table)
                                sub_table['sub_column_indice'] = comb
                                sub_table['sub_type'] = 'column_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_COLUMN({','.join(str_comb)})"

                                if set(comb).issubset(set(gold_col)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                    
                                sub_tables.append(sub_table)
                        
                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example


                    dataset = dataset.map(split_table_by_column_combination, batched=False, num_proc=4, desc="split_table_by_column_combination")

                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))
                
                if 'split_table_by_row_combination' in module_config.config.preprocess:
                    def split_table_by_row_combination(example):
                        # this function splits "table" into "row-wise sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])

                        # add type to avoid bugs
                        example['table']['types'] = ['unknown']*num_columns
                        example['table']['id'] = example['table'].pop('name')
                        
                        gold_col = []
                        gold_row = []
                        example['gold_columns'] = gold_col
                        example['gold_rows'] = gold_row

                        sub_tables = []

                        MAX_ROWS = 1
                        for max_row in range(1, MAX_ROWS+1):
                            for comb in combinations(list(range(num_rows)), max_row):
                                selected_columns = table['header']
                                selected_rows = [table['rows'][i] for i in comb]
                                selected_types = ['unknown']*len(selected_columns)
                                sub_table = deepcopy(table)
                                sub_table['sub_row_indice'] = comb
                                sub_table['sub_type'] = 'row_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_ROW({','.join(str_comb)})"
                                if set(comb).issubset(set(gold_row)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                
                                sub_tables.append(sub_table)
                        
                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example
                    

                    dataset = dataset.map(split_table_by_row_combination, batched=False, num_proc=4, desc="split_table_by_row_combination")
                    
                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))

                if 'split_table_by_mixed_combination' in module_config.config.preprocess:
                    def split_table_by_mixed_combination(example):
                        # this function splits "table" into "column-wise and row-wise sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])

                        # add type to avoid bugs
                        example['table']['types'] = ['unknown']*num_columns
                        example['table']['id'] = example['table'].pop('name')

                        gold_col = []
                        gold_row = []
                        example['gold_columns'] = gold_col
                        example['gold_rows'] = gold_row

                        sub_tables = []

                        MAX_ROWS = 1
                        for max_row in range(1, MAX_ROWS+1):
                            for comb in combinations(list(range(num_rows)), max_row):
                                selected_columns = table['header']
                                selected_rows = [table['rows'][i] for i in comb]
                                selected_types = table['types']
                                sub_table = deepcopy(table)
                                sub_table['sub_row_indice'] = comb
                                sub_table['sub_type'] = 'row_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_ROW({','.join(str_comb)})"
                                if set(comb).issubset(set(gold_row)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                
                                sub_tables.append(sub_table)
                        
                        MAX_COLUMNS = 1
                        for max_column in range(1, MAX_COLUMNS+1):
                            for comb in combinations(list(range(num_columns)), max_column):
                                selected_columns = [table['header'][i] for i in comb]
                                selected_rows = [[row[i] for i in comb] for row in table['rows']]
                                selected_types = [table['types'][i] for i in comb]
                                sub_table = deepcopy(table)
                                sub_table['sub_column_indice'] = comb
                                sub_table['sub_type'] = 'column_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_COLUMN({','.join(str_comb)})"

                                if set(comb).issubset(set(gold_col)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                sub_tables.append(sub_table)

                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example
                    

                    dataset = dataset.map(split_table_by_mixed_combination, batched=False, num_proc=4, desc="split_table_by_mixed_combination")
                    
                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))

                
                dataset.save_to_disk(split_file_path)
            
            ###### END OF PREPROCESSING ########

            dataset = dataset.rename_columns({'id': 'question_id'})
            
            if 'valid' not in dataset.column_names:
                conversion_success = len(dataset)
                conversion_error = 0
            else:
                conversion_success = len(dataset.filter(lambda x: x["valid"] == True))
                conversion_error = len(dataset.filter(lambda x: x["valid"] == False))

            self.data.wtq_data[split] = EasyDict({
                'dataset': dataset,
                'column_names': dataset.column_names,
                'conversion_success': conversion_success,
                'conversion_error': conversion_error,
            })

    def LoadWikiSQLData(self, module_config):
        '''
        Load WikiSQL Corpus
        {
          "type": "LoadWikiSQLData", "option": "default", 
          "config": {
                "preprocess": []
                "path":{
                    "train": "..",
                    "test": "..",
                },
                "tapas_path": {
                    "train": "..",
                    "test": "..",
                }
            },
        },
        '''
        self.data.wikisql_data = EasyDict({})

        datasets = None

        current_question_id = 0

        for split, split_file_path in module_config.config.path.items():
            split_file_path = os.path.join(self.config.DATA_FOLDER, split_file_path)
            
            is_training = (split in ['train'])

            if os.path.exists(split_file_path):
                dataset = load_from_disk(split_file_path)
            else:
                if not datasets:
                    datasets = load_dataset("wikisql")
                column_names = datasets[split].column_names
                dataset = datasets[split]

                def preprocess_tableqa_function(example):
                    """
                    Preprocess SQL data
                    """

                    # this function is specific for WikiSQL since the util function need the data structure
                    # to retrieve the WikiSQL answer for each question
                    def _convert_table_types(_table):
                        """Runs the type converter over the table cells."""
                        ret_table = deepcopy(_table)
                        types = ret_table["types"]
                        ret_table["real_rows"] = ret_table["rows"]
                        typed_rows = []
                        for row in ret_table["rows"]:
                            typed_row = []
                            for column, cell_value in enumerate(row):
                                typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
                            typed_rows.append(typed_row)
                        ret_table["rows"] = typed_rows
                        return ret_table

                    example_table = example["table"]
                    example_sql = example["sql"]
                    table = pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])

                    # using tapas utils to obtain wikisql answer
                    answers = []
                    
                    tapas_table = _convert_table_types(example_table)
                    answer_list, answer_coordinates, aggregation_op, conditions = retrieve_wikisql_query_answer_tapas(tapas_table, example_sql)

                    example['conditioned_columns'] = [condition.column for condition in conditions]
                    example['num_conditioned_columns'] = len(example['conditioned_columns'])

                    # you can choose other delimiters to split each answer
                    answers = answer_list
                    example['answer_coordinates'] = answer_coordinates
                    example['aggregation_function'] = aggregation_op.value
                    example['answers'] = answers

                    return example
                
                dataset = dataset.map(preprocess_tableqa_function, batched=False, num_proc=4, desc="preprocess_tableqa_function")
                dataset = dataset.add_column("question_id", list(range(current_question_id, current_question_id+len(dataset))))
                current_question_id += len(dataset)

                print(dataset.to_pandas().value_counts(subset=['aggregation_function']))
                
                if 'use_original_tapas_data' in module_config.config.preprocess:
                    # we try to add information from the original tapas data
                    tf_input_path = module_config.config.tapas_path[split]
                    tf_input_path = os.path.join(self.config.DATA_FOLDER, tf_input_path)
                    with open(tf_input_path, 'rb') as f:
                        logger.info(f'Loading tf_input files {tf_input_path}')
                        tf_dataset = pickle.load(f)

                    all_tf_data = {}
                    for example in tqdm(tf_dataset['examples']):
                        table_id = example["table_id"].replace("table_csv/", "").replace(".csv", "")
                        question = example["question"].lower()
                        all_tf_data[table_id + ":" + question] = example
                    
                    def attach_tf_input_to_dataset(example):
                        # for example in dataset:
                        table_id = example['table']['id']
                        question = example['question'].lower()
                        tf_input_example = all_tf_data.get(table_id + ":" + question, None)
                        if tf_input_example is not None:
                            # print(tf_input_example)
                            # example_table = example["table"]
                            # table = pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
                            # cell_coos = tf_input_example['answer_coordinates']
                            # aggregation_function_id = tf_input_example['aggregation_function_id']
                            
                            # values = _collect_cells_from_table(cell_coos, table)
                            
                            # values_parsed = [_parse_value(value) for value in values]
                            # values_parsed = tuple(values_parsed)
                            
                            # id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
                            # aggregation_type = id2aggregation[aggregation_function_id]
                            # if aggregation_type == "NONE":
                            #     # In this case there is no aggregation
                            #     pass
                            # else:  # Should perform aggregation.
                            #     if not values and (aggregation_type == "AVERAGE" or
                            #                     aggregation_type == "SUM"):
                            #         # Summing or averaging an empty set results in an empty set.
                            #         # NB: SQL returns null for sum over an empty set.
                            #         values_parsed = tuple()
                            #     else:
                            #         if aggregation_type == "COUNT":
                            #             denotation = len(values)
                            #             values_parsed = tuple([float(denotation)])
                            #         else:
                            #             # In this case all values must be numbers (to be summed or averaged).
                            #             try:
                            #                 values_num = [text_utils.convert_to_float(value) for value in values]
                            #             except ValueError:
                            #                 pass
                            #             if aggregation_type == "SUM":
                            #                 denotation = sum(values_num)
                            #                 values_parsed = tuple([float(denotation)])
                            #             elif aggregation_type == "AVERAGE":
                            #                 denotation = sum(values_num) / len(values_num)
                            #                 values_parsed = tuple([float(denotation)])
                            #             else:
                            #                 raise ValueError('Unknwon aggregation type: %s' % aggregation_type)
                                        
                            
                            
                            # if aggregation_type == "SUM" or aggregation_type=="AVERAGE":
                            #     print(table)
                            #     print(cell_coos)
                            #     print(values)
                            #     print("tapex answer:", example['answers'])
                            #     print("tapex agg:", example['aggregation_function'])

                            #     print("tapas answer:", values_parsed)
                            #     print("tapas agg:", id2aggregation[aggregation_function_id])
                            #     input('here')

                            # example['answer_coordinates'] = cell_coos
                            # example['aggregation_function'] = id2aggregation[aggregation_function_id]
                            # example['answers'] = values_parsed
                            example['valid'] = True
                        else:
                            example['valid'] = False
                        return example
                    

                    dataset = dataset.map(attach_tf_input_to_dataset, batched=False, desc="attach_tf_input_to_dataset")
                    
                    conversion_success = len(dataset.filter(lambda x: x["valid"] == True))
                    conversion_error = len(dataset.filter(lambda x: x["valid"] == False))
                    
                    # if is_training:
                    #     dataset = dataset.filter(lambda x: x["valid"] == True)

                    logger.info(f'conversion_success {conversion_success} / conversion_error {conversion_error} --> {conversion_success/(conversion_success+conversion_error)*100}%')

                if 'move_answers_to_table_end' in module_config.config.preprocess:
                    def move_answers_to_table_end(example):
                        example_table = example["table"]
                        example_sql = example["sql"]
                        table = pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
                        answer_coordinates = example['answer_coordinates']

                        # Read gold rows/columns
                        gold_col = [coord[1] for coord in answer_coordinates] + [example['sql']['sel']]
                        gold_col = list(set(gold_col))
                        conditioned_col = list(set(example['conditioned_columns']))
                        gold_row = list(set([coord[0] for coord in answer_coordinates]))
                        
                        original_row_ids = list(range(len(example_table["rows"])))
                        original_column_ids = list(range(len(example_table["header"])))
                        
                        shifted_row_ids = deepcopy(original_row_ids)
                        for index in gold_row:
                            shifted_row_ids.remove(index)
                            shifted_row_ids.append(index)

                        shifted_column_ids = deepcopy(original_column_ids)
                        for index in conditioned_col:
                            shifted_column_ids.remove(index)
                            shifted_column_ids.append(index)
                        for index in gold_col:
                            shifted_column_ids.remove(index)
                            shifted_column_ids.append(index)
                        
                        column_index_to_content = {
                            header_index: header_name for header_index, header_name in enumerate(example_table['header'])
                        }

                        # Change the table
                        new_table = table.iloc[shifted_row_ids, shifted_column_ids]
                        new_example_table = deepcopy(example_table)
                        new_example_table['header'] = [column_index_to_content[index] for index in shifted_column_ids]
                        new_example_table['rows'] = [row.tolist()[1:] for row in new_table.to_records()]
                        new_example_table['types'] = [example_table['types'][index] for index in shifted_column_ids]

                        # Create mapping from original column index to current column index
                        column_mapping = {f: t for f, t in zip(shifted_column_ids, original_column_ids)}
                        row_mapping = {f: t for f, t in zip(shifted_row_ids, original_row_ids)}

                        # change the sql
                        new_example_sql = deepcopy(example_sql)
                        new_example_sql['conds']['column_index'] = [
                            column_mapping[f] for f in new_example_sql['conds']['column_index']
                        ]
                        new_example_sql['sel'] = column_mapping[new_example_sql['sel']]
                        
                        # change other content
                        example['conditioned_columns_origin'] = example['conditioned_columns']
                        example['conditioned_columns'] = [
                            column_mapping[i] for i in example['conditioned_columns']
                        ]
                        example['answer_coordinates_origin'] = example['answer_coordinates']
                        example['answer_coordinates'] = [
                            [row_mapping[r], column_mapping[c]] for r, c in example['answer_coordinates']
                        ]

                        example["table_origin"] = example["table"]
                        example["table"] = new_example_table
                        example["sql_origin"] = example["sql"]
                        example["sql"] = new_example_sql
                        
                        return example
                    dataset = dataset.map(move_answers_to_table_end, batched=False, num_proc=4, desc="move_answers_to_table_end")
                
                if 'move_answers_to_table_end_column_wise' in module_config.config.preprocess:
                    def move_answers_to_table_end_column_wise(example):
                        example_table = example["table"]
                        example_sql = example["sql"]
                        table = pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
                        answer_coordinates = example['answer_coordinates']

                        # Read gold rows/columns
                        gold_col = [coord[1] for coord in answer_coordinates] + [example['sql']['sel']]
                        gold_col = list(set(gold_col))
                        conditioned_col = list(set(example['conditioned_columns']))
                        gold_row = list(set([coord[0] for coord in answer_coordinates]))
                        
                        original_row_ids = list(range(len(example_table["rows"])))
                        original_column_ids = list(range(len(example_table["header"])))
                        
                        shifted_row_ids = deepcopy(original_row_ids)
                        # for index in gold_row:
                        #     shifted_row_ids.remove(index)
                        #     shifted_row_ids.append(index)

                        shifted_column_ids = deepcopy(original_column_ids)
                        for index in conditioned_col:
                            shifted_column_ids.remove(index)
                            shifted_column_ids.append(index)
                        for index in gold_col:
                            shifted_column_ids.remove(index)
                            shifted_column_ids.append(index)
                        
                        column_index_to_content = {
                            header_index: header_name for header_index, header_name in enumerate(example_table['header'])
                        }

                        # Change the table
                        new_table = table.iloc[shifted_row_ids, shifted_column_ids]
                        new_example_table = deepcopy(example_table)
                        new_example_table['header'] = [column_index_to_content[index] for index in shifted_column_ids]
                        new_example_table['rows'] = [row.tolist()[1:] for row in new_table.to_records()]
                        new_example_table['types'] = [example_table['types'][index] for index in shifted_column_ids]

                        # Create mapping from original column index to current column index
                        column_mapping = {f: t for f, t in zip(shifted_column_ids, original_column_ids)}
                        row_mapping = {f: t for f, t in zip(shifted_row_ids, original_row_ids)}

                        # change the sql
                        new_example_sql = deepcopy(example_sql)
                        new_example_sql['conds']['column_index'] = [
                            column_mapping[f] for f in new_example_sql['conds']['column_index']
                        ]
                        new_example_sql['sel'] = column_mapping[new_example_sql['sel']]
                        
                        # change other content
                        example['conditioned_columns_origin'] = example['conditioned_columns']
                        example['conditioned_columns'] = [
                            column_mapping[i] for i in example['conditioned_columns']
                        ]
                        example['answer_coordinates_origin'] = example['answer_coordinates']
                        example['answer_coordinates'] = [
                            [row_mapping[r], column_mapping[c]] for r, c in example['answer_coordinates']
                        ]

                        example["table_origin"] = example["table"]
                        example["table"] = new_example_table
                        example["sql_origin"] = example["sql"]
                        example["sql"] = new_example_sql
                        
                        return example
                    dataset = dataset.map(move_answers_to_table_end_column_wise, batched=False, num_proc=4, desc="move_answers_to_table_end_column_wise")
                    
                if 'split_table_by_column' in module_config.config.preprocess:
                    def split_table_by_column(example):
                        # this function splits column "table" into "sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])
                        answer_coordinates = example['answer_coordinates']
                        gold_col = [coord[1] for coord in answer_coordinates] + [example['sql']['sel']]
                        example['has_gold_sub_table'] = (len(gold_col) > 0)
                        for i in range(num_columns):
                            for j in range(i+1, num_columns):
                                selected_columns = [table['header'][i], table['header'][j]]
                                selected_rows = [[row[i], row[j]] for row in table['rows']]
                                selected_types = [table['types'][i], table['types'][j]]
                                sub_table = deepcopy(table)
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['is_gold'] = (i in gold_col) or (j in gold_col)
                                sub_table['id'] = f"{table['id']}_COLUMN({i}_{j})"
                                if sub_table['is_gold']:
                                    positive_subtables.append(sub_table)
                                else:
                                    negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example


                    dataset = dataset.map(split_table_by_column, batched=False, num_proc=4, desc="split_table_by_column")
                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['has_gold_sub_table']))
                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))


                if 'split_table_by_column_combination' in module_config.config.preprocess:
                    def split_table_by_column_combination(example):
                        # this function splits column "table" into "sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])
                        answer_coordinates = example['answer_coordinates']
                        gold_col = [coord[1] for coord in answer_coordinates] + [example['sql']['sel']]
                        gold_col += example['conditioned_columns']
                        gold_col = list(set(gold_col))
                        example['gold_columns'] = sorted(list(gold_col))

                        sub_tables = []

                        MAX_COLUMNS = 1
                        for max_column in range(1, MAX_COLUMNS+1):
                            for comb in combinations(list(range(num_columns)), max_column):
                                selected_columns = [table['header'][i] for i in comb]
                                selected_rows = [[row[i] for i in comb] for row in table['rows']]
                                selected_types = [table['types'][i] for i in comb]
                                sub_table = deepcopy(table)
                                sub_table['sub_column_indice'] = comb
                                sub_table['sub_type'] = 'column_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_COLUMN({','.join(str_comb)})"

                                if set(comb).issubset(set(gold_col)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                # if sorted(list(gold_col)) == sorted(list(comb)):
                                #     sub_table['is_gold'] = True
                                #     sub_table['cover_gold'] = True
                                # else:
                                #     sub_table['is_gold'] = False
                                #     if set(gold_col).issubset(set(comb)):
                                #         sub_table['cover_gold'] = True
                                #         # if the gold col is 1, we set every combination that contains this gold col as gold
                                #         if len(gold_col) == 1:
                                #             sub_table['is_gold'] = True
                                #     else:
                                #         sub_table['cover_gold'] = False
                                    
                                    
                                sub_tables.append(sub_table)
                        
                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example


                    dataset = dataset.map(split_table_by_column_combination, batched=False, num_proc=4, desc="split_table_by_column_combination")

                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))
                
                if 'split_table_by_row_combination' in module_config.config.preprocess:
                    def split_table_by_row_combination(example):
                        # this function splits "table" into "row-wise sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])
                        answer_coordinates = example['answer_coordinates']
                        gold_col = [coord[1] for coord in answer_coordinates] + [example['sql']['sel']]
                        gold_col += example['conditioned_columns']
                        gold_col = list(set(gold_col))
                        example['gold_columns'] = sorted(list(gold_col))
                        
                        gold_row = [coord[0] for coord in answer_coordinates]
                        example['gold_rows'] = sorted(list(gold_row))

                        sub_tables = []

                        MAX_ROWS = 1
                        for max_row in range(1, MAX_ROWS+1):
                            for comb in combinations(list(range(num_rows)), max_row):
                                selected_columns = table['header']
                                selected_rows = [table['rows'][i] for i in comb]
                                selected_types = table['types']
                                sub_table = deepcopy(table)
                                sub_table['sub_row_indice'] = comb
                                sub_table['sub_type'] = 'row_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_ROW({','.join(str_comb)})"
                                if set(comb).issubset(set(gold_row)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                
                                sub_tables.append(sub_table)
                        
                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example
                    

                    dataset = dataset.map(split_table_by_row_combination, batched=False, num_proc=4, desc="split_table_by_row_combination")
                    
                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))

                if 'split_table_by_mixed_combination' in module_config.config.preprocess:
                    def split_table_by_mixed_combination(example):
                        # this function splits "table" into "column-wise and row-wise sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])
                        answer_coordinates = example['answer_coordinates']
                        gold_col = [coord[1] for coord in answer_coordinates] + [example['sql']['sel']]
                        gold_col += example['conditioned_columns']
                        gold_col = list(set(gold_col))
                        example['gold_columns'] = sorted(list(gold_col))
                        
                        gold_row = [coord[0] for coord in answer_coordinates]
                        example['gold_rows'] = sorted(list(gold_row))

                        sub_tables = []

                        MAX_ROWS = 1
                        for max_row in range(1, MAX_ROWS+1):
                            for comb in combinations(list(range(num_rows)), max_row):
                                selected_columns = table['header']
                                selected_rows = [table['rows'][i] for i in comb]
                                selected_types = table['types']
                                sub_table = deepcopy(table)
                                sub_table['sub_row_indice'] = comb
                                sub_table['sub_type'] = 'row_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_ROW({','.join(str_comb)})"
                                if set(comb).issubset(set(gold_row)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                
                                sub_tables.append(sub_table)
                        
                        MAX_COLUMNS = 1
                        for max_column in range(1, MAX_COLUMNS+1):
                            for comb in combinations(list(range(num_columns)), max_column):
                                selected_columns = [table['header'][i] for i in comb]
                                selected_rows = [[row[i] for i in comb] for row in table['rows']]
                                selected_types = [table['types'][i] for i in comb]
                                sub_table = deepcopy(table)
                                sub_table['sub_column_indice'] = comb
                                sub_table['sub_type'] = 'column_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_COLUMN({','.join(str_comb)})"

                                if set(comb).issubset(set(gold_col)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                sub_tables.append(sub_table)

                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example
                    

                    dataset = dataset.map(split_table_by_mixed_combination, batched=False, num_proc=4, desc="split_table_by_mixed_combination")
                    
                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))


                if 'remove_no_pos_samples_for_training' in module_config.config.preprocess:
                    dataset = dataset.filter(lambda x: x["num_positive_sub_tables"] > 0)


                ###### END OF PREPROCESSING ########
                logger.info(f'saving dataset files to {split_file_path}')
                dataset.save_to_disk(split_file_path)
            
            self.data.wikisql_data[split] = EasyDict({
                'dataset': dataset,
                'column_names': dataset.column_names,
            })

    def LoadComputronData(self, module_config):
        '''
        Load data from computron input
        Required input from io:
        Csv file: each entry is a question with a table
        Example:
        {'question_id': 'nu-0', 
        'question': 'which country had the most cyclists finish within the top 10?', 
        'table': {
            'header': ['Rank', 'Cyclist', 'Team', 'Time', 'UCI ProTour\\nPoints'], 
            'rows': [['1', 'Alejandro Valverde\xa0(ESP)', "Caisse d'Epargne", '5h 29\' 10"', '40'], ['2', 'Alexandr Kolobnev\xa0(RUS)', 'Team CSC Saxo Bank', 's.t.', '30'], ['3', 'Davide Rebellin\xa0(ITA)', 'Gerolsteiner', 's.t.', '25'], ['4', 'Paolo Bettini\xa0(ITA)', 'Quick Step', 's.t.', '20'], ['5', 'Franco Pellizotti\xa0(ITA)', 'Liquigas', 's.t.', '15'], ['6', 'Denis Menchov\xa0(RUS)', 'Rabobank', 's.t.', '11'], ['7', 'Samuel Sánchez\xa0(ESP)', 'Euskaltel-Euskadi', 's.t.', '7'], ['8', 'Stéphane Goubert\xa0(FRA)', 'Ag2r-La Mondiale', '+ 2"', '5'], ['9', 'Haimar Zubeldia\xa0(ESP)', 'Euskaltel-Euskadi', '+ 2"', '3'], ['10', 'David Moncoutié\xa0(FRA)', 'Cofidis', '+ 2"', '1']]
            }
        }
        The module config is:
        {
          "type": "LoadComputronData", "option": "default", 
          "config": {},
        },
        '''

        self.data.computron_data = EasyDict({})

        for split, split_file_path in module_config.config.path.items():
            split_file_path = os.path.join(self.config.DATA_FOLDER, split_file_path)

            if os.path.exists(split_file_path):
                dataset = load_from_disk(split_file_path)
            else:
                with self.io.input(split) as json_file:
                    dataset = pd.read_json(json_file, orient="records")
                    dataset = Dataset.from_pandas(dataset)
                
                column_names = dataset.column_names

                if 'split_table_by_column_combination' in module_config.config.preprocess:
                    def split_table_by_column_combination(example):
                        # this function splits column "table" into "sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])

                        # add type to avoid bugs
                        example['table']['types'] = ['unknown']*num_columns
                        example['table']['id'] = example['table'].pop('name')

                        gold_col = []
                        example['gold_columns'] = gold_col

                        sub_tables = []

                        MAX_COLUMNS = 1
                        for max_column in range(1, MAX_COLUMNS+1):
                            for comb in combinations(list(range(num_columns)), max_column):
                                selected_columns = [table['header'][i] for i in comb]
                                selected_rows = [[row[i] for i in comb] for row in table['rows']]
                                selected_types = ['unknown']*len(selected_columns)
                                sub_table = deepcopy(table)
                                sub_table['sub_column_indice'] = comb
                                sub_table['sub_type'] = 'column_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_COLUMN({','.join(str_comb)})"

                                if set(comb).issubset(set(gold_col)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                    
                                sub_tables.append(sub_table)
                        
                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example


                    dataset = dataset.map(split_table_by_column_combination, batched=False, num_proc=4, desc="split_table_by_column_combination")

                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))
                
                if 'split_table_by_row_combination' in module_config.config.preprocess:
                    def split_table_by_row_combination(example):
                        # this function splits "table" into "row-wise sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])

                        # add type to avoid bugs
                        example['table']['types'] = ['unknown']*num_columns
                        example['table']['id'] = example['table'].pop('name')
                        
                        gold_col = []
                        gold_row = []
                        example['gold_columns'] = gold_col
                        example['gold_rows'] = gold_row

                        sub_tables = []

                        MAX_ROWS = 1
                        for max_row in range(1, MAX_ROWS+1):
                            for comb in combinations(list(range(num_rows)), max_row):
                                selected_columns = table['header']
                                selected_rows = [table['rows'][i] for i in comb]
                                selected_types = ['unknown']*len(selected_columns)
                                sub_table = deepcopy(table)
                                sub_table['sub_row_indice'] = comb
                                sub_table['sub_type'] = 'row_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_ROW({','.join(str_comb)})"
                                if set(comb).issubset(set(gold_row)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                
                                sub_tables.append(sub_table)
                        
                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example
                    

                    dataset = dataset.map(split_table_by_row_combination, batched=False, num_proc=4, desc="split_table_by_row_combination")
                    
                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))

                if 'split_table_by_mixed_combination' in module_config.config.preprocess:
                    def split_table_by_mixed_combination(example):
                        # this function splits "table" into "column-wise and row-wise sub-tables"
                        table = example["table"]
                        positive_subtables = []
                        negative_subtables = []
                        num_columns = len(table['header'])
                        num_rows = len(table['rows'])

                        # add type to avoid bugs
                        example['table']['types'] = ['unknown']*num_columns
                        example['table']['id'] = example['table'].pop('name')

                        gold_col = []
                        gold_row = []
                        example['gold_columns'] = gold_col
                        example['gold_rows'] = gold_row

                        sub_tables = []

                        MAX_ROWS = 1
                        for max_row in range(1, MAX_ROWS+1):
                            for comb in combinations(list(range(num_rows)), max_row):
                                selected_columns = table['header']
                                selected_rows = [table['rows'][i] for i in comb]
                                selected_types = table['types']
                                sub_table = deepcopy(table)
                                sub_table['sub_row_indice'] = comb
                                sub_table['sub_type'] = 'row_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_ROW({','.join(str_comb)})"
                                if set(comb).issubset(set(gold_row)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                
                                sub_tables.append(sub_table)
                        
                        MAX_COLUMNS = 1
                        for max_column in range(1, MAX_COLUMNS+1):
                            for comb in combinations(list(range(num_columns)), max_column):
                                selected_columns = [table['header'][i] for i in comb]
                                selected_rows = [[row[i] for i in comb] for row in table['rows']]
                                selected_types = [table['types'][i] for i in comb]
                                sub_table = deepcopy(table)
                                sub_table['sub_column_indice'] = comb
                                sub_table['sub_type'] = 'column_wise'
                                sub_table['header'] = selected_columns
                                sub_table['rows'] = selected_rows
                                sub_table['types'] = selected_types
                                sub_table['original_table_id'] = table['id']
                                str_comb = [str(i) for i in comb]
                                sub_table['id'] = f"{table['id']}_COLUMN({','.join(str_comb)})"

                                if set(comb).issubset(set(gold_col)):
                                    # if the column of this subtable is a gold col
                                    sub_table['is_gold'] = True
                                else:
                                    sub_table['is_gold'] = False
                                sub_tables.append(sub_table)

                        for sub_table in sub_tables:
                            if sub_table['is_gold']:
                                positive_subtables.append(sub_table)
                            else:
                                negative_subtables.append(sub_table)
                        
                        example['positive_sub_tables'] = positive_subtables
                        example['negative_sub_tables'] = negative_subtables
                        example['num_positive_sub_tables'] = len(positive_subtables)
                        example['num_negative_sub_tables'] = len(negative_subtables)
                        example['num_sub_tables'] = len(positive_subtables) + len(negative_subtables)
                        return example
                    

                    dataset = dataset.map(split_table_by_mixed_combination, batched=False, num_proc=4, desc="split_table_by_mixed_combination")
                    
                    df = dataset.to_pandas()

                    print(df.value_counts(subset=['num_positive_sub_tables']))
                    print(df.value_counts(subset=['num_negative_sub_tables']))
                    print(df.value_counts(subset=['num_sub_tables']))

                
                dataset.save_to_disk(split_file_path)
            
            ###### END OF PREPROCESSING ########

            self.data.computron_data[split] = EasyDict({
                'dataset': dataset,
                'column_names': dataset.column_names,
            })

    def LoadNQTableData(self, module_config):
        """
        Load NQ Tables Corpus
        {
          "type": "LoadNQTableData", "option": "default", 
          "config": {
                "preprocess": [],
                "table_path": {
                    "tables": "..",
                }
                "data_path":{
                    "train": "..",
                    "test": "..",
                },
                "path":{
                    "train": "..",
                    "test": "..",
                },
            },
        },
        """
        self.data.nq_tables_data = EasyDict()

        all_tables = {}
        for split, split_path in module_config.config.table_path.items():
            split_cache_path = module_config.config.path[split]
            split_cache_path = os.path.join(self.config.DATA_FOLDER, split_cache_path)
            if os.path.exists(split_cache_path):
                table_dataset = load_from_disk(split_cache_path)
            else:
                split_path = os.path.join(self.config.DATA_FOLDER, split_path)
                with open(split_path, "r") as json_file:
                    for line in tqdm(json_file):
                        line_data = json.loads(line)
                        num_columns = len(line_data['columns'])
                        columns = [i['text'] for i in line_data['columns']]
                        rows = [
                            [i['text'] for i in row['cells']] for row in line_data['rows']
                        ]
                        example = EasyDict(
                            header=columns,
                            rows=rows,
                            documentTitle=line_data['documentTitle'],
                            documentUrl=line_data['documentUrl'],
                            table_id=line_data['tableId'],
                        )
                        all_tables[example.table_id] = example

                table_dataset = pd.DataFrame.from_dict(all_tables, orient='index')
                table_dataset = Dataset.from_pandas(table_dataset)
                table_dataset = table_dataset.remove_columns(['__index_level_0__'])

                table_dataset.save_to_disk(split_cache_path)

        self.data.nq_tables_data.tables = all_tables

        
        for split, split_path in module_config.config.data_path.items():
            split_path = os.path.join(self.config.DATA_FOLDER, split_path)
            logger.info(f"Loading {split} from {split_path}...")

            split_cache_path = module_config.config.path[split]
            split_cache_path = os.path.join(self.config.DATA_FOLDER, split_cache_path)
            if os.path.exists(split_cache_path):
                dataset = load_from_disk(split_cache_path)
            else:
                if self.data.nq_tables_data.get('bm25_results', None) is None:
                    bm25_results_path = os.path.join(self.config.DATA_FOLDER, module_config.config.bm25_results)
                    if not os.path.exists(bm25_results_path):
                        raise Exception("Please generate BM25 results first.")
                    with open(bm25_results_path, 'r') as bm25_results_file:
                        bm25_results = json.load(bm25_results_file)

                dataset = dict()
                with open(split_path, "r") as json_file:
                    for line in tqdm(json_file):
                        """
                        {"id": "dev_6561754476777446630_1", "table": {"columns": [], "rows": [], "tableId": "Cristiano Ronaldo_B6B8B56142019E98", "documentTitle": "Cristiano Ronaldo", "documentUrl": "https://en.wikipedia.org//w/index.php?title=Cristiano_Ronaldo&amp;oldid=818163684", "alternativeDocumentUrls": ["https://en.wikipedia.org//w/index.php?title=Cristiano_Ronaldo&amp;oldid=800739022", "https://en.wikipedia.org//w/index.php?title=Cristiano_Ronaldo&amp;oldid=801113912", "https://en.wikipedia.org//w/index.php?title=Cristiano_Ronaldo&amp;oldid=801482569", "https://en.wikipedia.org//w/index.php?title=Cristiano_Ronaldo&amp;oldid=802039909", "https://en.wikipedia.org//w/index.php?title=Cristiano_Ronaldo&amp;oldid=807219735", "https://en.wikipedia.org//w/index.php?title=Cristiano_Ronaldo&amp;oldid=808136697", "https://en.wikipedia.org//w/index.php?title=Cristiano_Ronaldo&amp;oldid=816185025"], "alternativeTableIds": ["Cristiano Ronaldo_111FFD216DC1C530", "Cristiano Ronaldo_11A5D7C0A2E7A306", "Cristiano Ronaldo_257BF1983FE919E2", "Cristiano Ronaldo_7D1CD8C764252F9", "Cristiano Ronaldo_DE58EA7051F67D37"]}, "questions": [{"id": "dev_6561754476777446630_1_0", "originalText": "when did cristiano ronaldo go to manchester united", "answer": {"answerTexts": ["2003"]}, "alternativeAnswers": [{"answerTexts": ["2003"]}]}]}
                        """
                        line_data = json.loads(line)
                        
                        table_id = line_data['table']['tableId']
                        alternative_table_ids = line_data['table'].get('alternativeTableIds', [])
                        for alter in alternative_table_ids:
                            if all_tables.get(alter, None) is not None:
                                logger.error(alter, "found in table corpus")
                        
                        for question_dict in line_data['questions']:
                            question_id = question_dict['id']
                            question = question_dict['originalText']
                            answers = question_dict['answer']['answerTexts']
                            alternative_answers = []
                            for alternative_answer in question_dict.get('alternativeAnswers', []):
                                alternative_answer = alternative_answer['answerTexts']
                                if alternative_answer == answers or alternative_answer in alternative_answer:
                                    continue
                                alternative_answers.append(alternative_answer)
                            
                            bm25_retrieved_items = bm25_results.get(question_id, None)
                            if bm25_retrieved_items:
                                bm25_retrieved_item_ids = [i['id'] for i in bm25_retrieved_items]
                            else:
                                logger.error(f"{question_id} not found retrieved bm25 documents!")
                                bm25_retrieved_item_ids = []
                            
                            example = EasyDict(
                                question_id=question_id,
                                question=question,
                                answers=answers,
                                alternative_answers=alternative_answers,
                                pos_item_id=table_id,
                                bm25_retrieved_item_ids=bm25_retrieved_item_ids,
                            )
                            dataset[question_id] = example
                
                dataset = pd.DataFrame.from_dict(dataset, orient='index')
                dataset = Dataset.from_pandas(dataset)
                dataset = dataset.remove_columns(['__index_level_0__'])

                # dataset.save_to_disk(split_cache_path)

            self.data.nq_tables_data[split] = EasyDict({
                'table_dataset': table_dataset,
                'dataset': dataset,
            })
        
    def LoadE2EWTQData(self, module_config):
        """
        Load E2E WTQ Corpus
        {
          "type": "LoadE2EWTQData", "option": "default", 
          "config": {
                "preprocess": [],
                "data_path":{
                    "train": "..",
                    "test": "..",
                },
                "path":{
                    "train": "..",
                    "test": "..",
                },
            },
        },
        """
        self.data.e2e_wtq_data = EasyDict()

        all_tables = {}
        
        table_cache_path = os.path.join(self.config.DATA_FOLDER, module_config.config.path.tables)
        if os.path.exists(table_cache_path):
            table_dataset = load_from_disk(table_cache_path)
        else:
            # Load tables from wikitablequestions
            datasets = load_dataset("wikitablequestions")

            for dataset in datasets.values():
                for item in dataset:
                    table_id = item['table']['name']
                    table = item['table']
                    
                    example = EasyDict(
                        header=table['header'],
                        rows=table['rows'],
                        documentTitle="",
                        documentUrl="",
                        table_id=table['name'],
                    )
                    all_tables[table_id] = example

        datasets = {}

        if self.data.e2e_wtq_data.get('bm25_results', None) is None:
            bm25_results_path = os.path.join(self.config.DATA_FOLDER, module_config.config.bm25_results)
            if not os.path.exists(bm25_results_path):
                raise Exception("Please generate BM25 results first.")
            with open(bm25_results_path, 'r') as bm25_results_file:
                bm25_results = json.load(bm25_results_file)
        
        for split, split_path in module_config.config.data_path.items():
            split_path = os.path.join(self.config.DATA_FOLDER, split_path)
            logger.info(f"Loading {split} from {split_path}...")

            split_cache_path = module_config.config.path[split]
            split_cache_path = os.path.join(self.config.DATA_FOLDER, split_cache_path)
            if os.path.exists(split_cache_path):
                dataset = load_from_disk(split_cache_path)
            else:
                dataset = dict()
                with gzip.open(split_path, 'rt', encoding='utf-8') as f:
                    data = f.readlines()
                
                for data_entry in data:
                    item = json.loads(data_entry)
                    table_id = item['table_id']
                    table_id = table_id.replace(".csv", ".tsv")
                    table = all_tables.get(table_id, None)
                    if table is None:
                        # add this table to corpus
                        example = EasyDict(
                            header=item['header'],
                            rows=item['rows'],
                            documentTitle="",
                            documentUrl="",
                            table_id=table_id,
                        )
                        all_tables[table_id] = example
                        table = all_tables[table_id]
                        logger.info(f"added {table_id} to corpus.")
                    
                    question_id = item['id']
                    question = item['question']
                    answers = item['answers']
                    bm25_retrieved_items = bm25_results.get(question_id, None)
                    if bm25_retrieved_items:
                        bm25_retrieved_item_ids = [i['id'] for i in bm25_retrieved_items]
                    else:
                        logger.error(f"{question_id} not found retrieved bm25 documents!")
                        bm25_retrieved_item_ids = []
                    
                    example = EasyDict(
                        question_id=question_id,
                        question=question,
                        answers=answers,
                        pos_item_id=table_id,
                        bm25_retrieved_item_ids=bm25_retrieved_item_ids,
                    )
                    dataset[question_id] = example
                
                dataset = pd.DataFrame.from_dict(dataset, orient='index')
                dataset = Dataset.from_pandas(dataset)
                dataset = dataset.remove_columns(['__index_level_0__'])
                logger.info(f"Loaded {len(dataset)} samples from {split}")
                # dataset.save_to_disk(split_cache_path)

                datasets[split] = EasyDict({
                    'dataset': dataset,
                })

        
        logger.info(f"Loaded {len(all_tables)} tables...")
        table_dataset = pd.DataFrame.from_dict(all_tables, orient='index')
        table_dataset = Dataset.from_pandas(table_dataset)
        table_dataset = table_dataset.remove_columns(['__index_level_0__'])

        self.data.e2e_wtq_data.tables = all_tables
        for split, d in datasets.items():
            self.data.e2e_wtq_data[split] = EasyDict({
                'table_dataset': table_dataset,
                'dataset': d['dataset'],
            })



    def LoadDataLoaders(self, module_config):
        """
        Load all data loaders.
        {
          "type": "LoadDataLoaders", "option": "default",
          "config": {
            "train": [
                {
                    "dataset_type": "WikiTQDataset",
                    "split": "train",
                    "use_column": "wtq_data",
                },
            ],
            "valid": [
                {
                    "dataset_type": "WikiTQDataset",
                    "split": "valid",
                    "use_column": "wtq_data",
                },
            ],
            "test": [
                {
                    "dataset_type": "WikiTQDataset",
                    "split": "test",
                    "use_column": "wtq_data",
                },
            ],
          }
        }
        """

        for mode in module_config.config.keys():
            for data_config in module_config.config[mode]:
                use_column = data_config.use_column
                use_split = data_config.split
                dataset_type = data_config.dataset_type
                dataset_dict = {
                    'data': self.data[use_column][use_split],
                    'tokenizer': self.tokenizer,
                    'decoder_tokenizer': self.decoder_tokenizer,
                    'feature_extractor': self.feature_extractor,
                    'mode': mode,
                }
                dataset = globals()[dataset_type](self.config, dataset_dict)
                
                # # Save data to src/tgt file format
                # save_path_src = os.path.join(
                #     self.config.DATA_FOLDER, f"{use_split}.source"
                # )
                # save_path_tgt = os.path.join(
                #     self.config.DATA_FOLDER, f"{use_split}.target"
                # )
                # source = []
                # target = []
                # for i in dataset:
                #     source.append(i['question'])
                #     concat_answers = [", ".join(i['answers'])] + [", ".join(ans) for ans in i.get('alternative_answers', [])]
                #     target.append(
                #         ("||".join(concat_answers) + "|||" + i['pos_item_id']).replace("\n", " ")
                #     )
                
                # with open(save_path_src, 'w') as f:
                #     f.write('\n'.join(source) + '\n')
                # with open(save_path_tgt, 'w') as f:
                #     f.write('\n'.join(target) + '\n')
                # print(f"done saving files {save_path_src} and {save_path_tgt}.")
                # input()

                if mode == 'train':
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
                
                data_loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=self.config[mode].batch_size,
                    collate_fn=dataset.collate_fn,
                    # num_workers=2,
                )
                # if mode == 'train':
                # for i in data_loader:
                #     pprint(i)
                #     input('done!')
                #     break

                # self.datasets[mode][f"{mode}/{dataset_type}.{use_split}"] = dataset
                self.data_loaders[mode][f"{mode}/{dataset_type}.{use_split}"] = data_loader

                logger.info(f'[Data Statistics]: {mode} data loader: {mode}/{dataset_type}.{use_split} {len(data_loader)}')
    
