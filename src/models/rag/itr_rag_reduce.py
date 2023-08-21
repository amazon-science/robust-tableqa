# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import copy
import math
import os
import warnings
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from easydict import EasyDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
from transformers import VisualBertModel, VisualBertConfig, BertTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.rag.retrieval_rag import CustomHFIndex, CanonicalHFIndex
import pytorch_lightning as pl
import glob
import time
import random

from pprint import pprint
import pandas as pd
from copy import deepcopy

import logging
logger = logging.getLogger(__name__)

from .itr_rag import ITRRagModel, shift_tokens_right


class ITRRagReduceModel(ITRRagModel):
    '''
    Class for Inner Table Retrieval RAG model
    Column-wise Reduction
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        ITRRagModel.__init__(self, config, data_loader)

    def prepare_inputs_for_generator(self, 
                input_text_sequences: List, 
                retrieved_sub_tables: List, 
                question_ids: List,
                gold_columns: List,
                gold_rows: List,
                tables: List,
                labels, 
                n_docs,
                is_training: bool = False):
        
        batch_size = len(input_text_sequences)

        extended_input_text_sequences = []

        overflow_only = ('overflow_only' in self.config.model_config.modules)
        original_sub_table_order = ('original_sub_table_order' in self.config.model_config.modules)
        # prepare sub tables here
        # e.g. concatenate subtables, postprocessing retrieval results...

        def reduce_table(original_table, ranked_sub_tables):
            table_pd = pd.DataFrame.from_records(original_table['rows'], columns=original_table['header'])
            column_index_to_content = {
                header_index: header_name for header_index, header_name in enumerate(original_table['header'])
            }
            all_column_indices = list(range(len(original_table['header'])))
            current_column_indices = []
            
            processed_sub_tables = []

            # all_rows = range(len(table_pd))
            for i in ranked_sub_tables:
                current_column_indices.extend(i['sub_column_indice'])
                
                processed_sub_table = deepcopy(i)
                processed_sub_table['sub_column_indice'] = current_column_indices.copy()

                if original_sub_table_order:
                    processed_sub_table['sub_column_indice'] = sorted(processed_sub_table['sub_column_indice'])

                selected_table_pd = table_pd.iloc[:, processed_sub_table['sub_column_indice']]
                processed_sub_table['header'] = [column_index_to_content[index] for index in processed_sub_table['sub_column_indice']]
                processed_sub_table['rows'] = selected_table_pd.values.tolist()
                processed_sub_table['types'] = [original_table['types'][index] for index in processed_sub_table['sub_column_indice']]
                processed_sub_tables.append(processed_sub_table)
            return processed_sub_tables

        mask = torch.ones(batch_size, n_docs).to(self.device)
        input_sub_tables = []
        for index, question_id, input_text_sequence, gold_column_list, ranked_sub_tables, table in zip(range(batch_size), question_ids, input_text_sequences, gold_columns, retrieved_sub_tables, tables):
            # concatenate the retrieved column with previous subtables
            processed_sub_tables = reduce_table(table, ranked_sub_tables)
            
            # we set the minimum #col to be 2 here
            if len(processed_sub_tables) > 1:
                processed_sub_tables = processed_sub_tables[1:]
            
            # batch input_text_sequence with each of the retrieved sub_tables
            input_text_and_sub_tables = [(input_text_sequence, sub_table) for sub_table in processed_sub_tables]

            # pprint(processed_sub_tables)
            encoding = self.generator_tokenizer(
                query=[text_table_tuple[0] for text_table_tuple in input_text_and_sub_tables],
                table=[
                    pd.DataFrame.from_records(text_table_tuple[1]["rows"], columns=text_table_tuple[1]["header"]) for text_table_tuple in input_text_and_sub_tables
                ],
                padding='longest',
                truncate=False,
                return_tensors="pt",
            )
            attention_masks = encoding.attention_mask

            max_ts = (attention_masks == 1).nonzero().numpy().tolist()
            token_lengths = defaultdict(int)
            original_table_overflow = False
            for pos in max_ts:
                token_lengths[pos[0]] = pos[1]
                if pos[1] > self.config.data_loader.additional.max_decoder_source_length:
                    original_table_overflow = True
            
            # set input_length for subtables
            for index, sub_table in enumerate(processed_sub_tables):
                sub_table['input_length'] = token_lengths[index]
                # this is for later evaluation
                sub_table['original_table_overflow'] = original_table_overflow
            
            # print(token_lengths)
            token_lengths = [(i, token_lengths[i]) for i in range(len(processed_sub_tables))]
            # print(token_lengths)
            if not is_training:
                
                

                """
                1. [1 1 (1 1 1 1 1) 0 0 0]
                2. [0 0 (0 0 0 0 0)]
                3. [(1 1 1 0 0) 0 0 0 0 0]
                4. [(0 0)] / [(1 1) 0] --> need to pad
                """
                if overflow_only and not original_table_overflow:
                    # if the original table does not overflow
                    # and we only reduce tables for overflow samples
                    # we put the whole table in the input and get predictions
                    token_lengths_reduced = [token_lengths[-1]]*n_docs
                    
                else:
                    token_lengths_reduced = [(i, length) for i, length in token_lengths if length <= self.config.data_loader.additional.max_decoder_source_length]

                    # print('after reduction:', token_lengths_reduced)
                    if len(token_lengths_reduced) == 0:
                        # Case 2: if it does not even fit one table
                        logger.warning(f"question {question_id} table {table['id']} can not fit!")
                        # keep the original table
                        token_lengths_reduced = token_lengths
                    elif len(token_lengths_reduced) < n_docs and len(token_lengths_reduced) > 0:
                        # Case 3
                        token_lengths_reduced = token_lengths[:min(n_docs, len(token_lengths))]
                    else:
                        # Case 1
                        token_lengths_reduced = token_lengths_reduced[-n_docs:]
                    
            else:
                # in training, just make sure the last element has all gold columns
                # token_lengths = [(i, 0) for i in range(len(processed_sub_tables))]

                earliest_element_that_covers_all_gold = -1

                if self.config.model_config.get('supervised', True) == True:
                    for index, subtable in enumerate(processed_sub_tables):
                        cover_all_gold = set(gold_column_list).issubset(set(subtable['sub_column_indice']))
                        if cover_all_gold:
                            earliest_element_that_covers_all_gold = index
                            break
                
                last_element_that_fits_table = -1
                for index, length in token_lengths:
                    if length <= self.config.data_loader.additional.max_decoder_source_length:
                        last_element_that_fits_table = index
                
                if last_element_that_fits_table == -1:
                    token_lengths_reduced = [token_lengths[earliest_element_that_covers_all_gold]]*n_docs
                else:
                    use_index = max(last_element_that_fits_table, earliest_element_that_covers_all_gold)
                    token_lengths_reduced = [token_lengths[use_index]]*n_docs
                

            if n_docs > len(token_lengths_reduced):
                # not enough to fit in n_docs
                # repeat the first table
                token_lengths_reduced = [token_lengths_reduced[0]]*(n_docs - len(token_lengths_reduced)) + token_lengths_reduced
            else:
                token_lengths_reduced = token_lengths_reduced[-n_docs:]
            
            # print('after truncation:', token_lengths_reduced)
            # print('gold_column_list', gold_column_list)
            extended_input_text_sequences += [input_text_and_sub_tables[i] for i, _ in token_lengths_reduced]
            input_sub_tables.append([input_text_and_sub_tables[i][1] for i, _ in token_lengths_reduced])

        # only train with the most rich item
        mask[:, -1] = 1
        # print(mask)
        
        # for index, input_text_sequence in enumerate(input_text_sequences):
        #     for sub_table in retrieved_sub_tables[index]:
        #         extended_input_text_sequences.append(
        #             (input_text_sequence, sub_table)
        #         )
        # pprint(extended_input_text_sequences)
        # input()

        targets = labels
        # We use tapex tokenizer here, so we need to input table
        
        encoding = self.generator_tokenizer(
            query=[text_table_tuple[0] for text_table_tuple in extended_input_text_sequences],
            table=[
                pd.DataFrame.from_records(text_table_tuple[1]["rows"], columns=text_table_tuple[1]["header"]) for text_table_tuple in extended_input_text_sequences
            ],
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt")
        generator_input_ids, generator_attention_mask, generator_token_type_ids = encoding.input_ids, encoding.attention_mask, encoding.get("token_type_ids", None)
        generator_input_ids = generator_input_ids.to(self.device)
        generator_attention_mask = generator_attention_mask.to(self.device)
        if generator_token_type_ids is not None:
            generator_token_type_ids = generator_token_type_ids.to(self.device)
        
        if targets is not None:
            generator_decoder_input_ids = shift_tokens_right(
                targets, self.generator.config.pad_token_id, self.generator.config.decoder_start_token_id
            )
        else:
            generator_decoder_input_ids = None

        return EasyDict(
            generator_input_text_sequences=extended_input_text_sequences,
            generator_input_sub_tables=input_sub_tables,
            generator_input_ids=generator_input_ids,
            generator_attention_mask=generator_attention_mask,
            generator_decoder_input_ids=generator_decoder_input_ids,
            generator_token_type_ids=generator_token_type_ids,
            generator_labels=targets,
            generator_loss_mask=mask,
        )



class ITRRagReduceRowWiseModel(ITRRagModel):
    '''
    Class for Inner Table Retrieval RAG model (reduction version) Row-Wise
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        ITRRagModel.__init__(self, config, data_loader)

    def _get_top_sub_tables_from_n_gram(self,
                                    question_ids: List,
                                    query_hidden_states: torch.Tensor,
                                    n_docs: int,
                                    input_text_sequences: List,
                                    tables: List,
                                    **kwargs
                                    ):
        retrieved_sub_tables = []
        scores = []
        item_embeddings = []

        def return_n_gram(string, n_gram):
            words = string.lower().split()
            return [
                " ".join(words[index:index+n_gram]) for index in range(len(words)-n_gram+1)
            ]
        n_gram_list = self.config.model_config.get('n_gram_list', [1, 2, 3])

        for question_id, question, table, query_embedding in zip(question_ids, input_text_sequences, tables, query_hidden_states):
            sub_table_ids = self.question_id_to_sub_table_ids[question_id]
            sub_tables = [self.sub_table_id_to_content[question_id][sub_table_id] for sub_table_id in sub_table_ids]
            sub_table_embeddings = [self.sub_table_id_to_embeddings[question_id][sub_table_id] for sub_table_id in sub_table_ids] # list of emb_size

            # num_subtables x emb_size
            sub_table_embeddings = torch.stack(sub_table_embeddings)

            # move the embeddings to gpu
            sub_table_embeddings = sub_table_embeddings.to(query_embedding.device)

            sub_table_ratings = []
            for sub_table in sub_tables:
                row_content = sub_table['header'] + sub_table['rows'][0]
                sub_table_score = 0
                for n_gram in n_gram_list:
                    question_n_grams = return_n_gram(question, n_gram)
                    row_content_n_grams = return_n_gram(" ".join(row_content), n_gram)
                    overlap = set(row_content_n_grams).intersection(set(question_n_grams))
                    sub_table_score += len(overlap)
                sub_table_ratings.append(sub_table_score)
            
            sub_table_ratings = torch.FloatTensor(sub_table_ratings).to(query_embedding.device)

            # # We return all ranked sub tables
            # Sort scores
            sorted_ratings, indices = torch.sort(sub_table_ratings, descending=True)
            ranked_scores = sorted_ratings # [:k]
            ranked_indices = indices # [:k]
            
            ranked_sub_tables = [self.sub_table_id_to_content[question_id][sub_table_ids[index]] for index in ranked_indices]

            # num_sub_tables x hidden_size
            ranked_item_embeddings = sub_table_embeddings[ranked_indices]
            # list of [num_sub_tables x hidden_size]
            item_embeddings.append(ranked_item_embeddings)
            scores.append(ranked_scores)
            retrieved_sub_tables.append(ranked_sub_tables)

        
        return EasyDict(dict(
            retrieved_sub_tables=retrieved_sub_tables,
            retrieved_sub_table_scores=scores,
            retrieved_sub_table_embeddings=item_embeddings,
        ))

    def prepare_inputs_for_generator(self, 
                input_text_sequences: List, 
                retrieved_sub_tables: List, 
                question_ids: List,
                gold_columns: List,
                gold_rows: List,
                tables: List,
                labels, 
                n_docs,
                is_training: bool = False):
        
        batch_size = len(input_text_sequences)

        extended_input_text_sequences = []

        overflow_only = ('overflow_only' in self.config.model_config.modules)
        original_sub_table_order = ('original_sub_table_order' in self.config.model_config.modules)
        # prepare sub tables here
        # e.g. concatenate subtables, postprocessing retrieval results...

        def reduce_table(original_table, ranked_sub_tables):
            table_pd = pd.DataFrame.from_records(original_table['rows'], columns=original_table['header'])
            current_row_indices = []
            
            processed_sub_tables = []

            # all_rows = range(len(table_pd))
            for i in ranked_sub_tables:
                current_row_indices.extend(i['sub_row_indice'])
                
                processed_sub_table = deepcopy(i)
                processed_sub_table['sub_row_indice'] = current_row_indices.copy()

                if original_sub_table_order:
                    processed_sub_table['sub_row_indice'] = sorted(processed_sub_table['sub_row_indice'])

                selected_table_pd = table_pd.iloc[processed_sub_table['sub_row_indice'], :]
                processed_sub_table['header'] = original_table['header']
                processed_sub_table['rows'] = selected_table_pd.values.tolist()
                processed_sub_table['types'] = original_table['types']
                processed_sub_tables.append(processed_sub_table)
            return processed_sub_tables

        mask = torch.ones(batch_size, n_docs).to(self.device)
        input_sub_tables = []
        for index, question_id, input_text_sequence, gold_column_list, gold_row_list, ranked_sub_tables, table in zip(range(batch_size), question_ids, input_text_sequences, gold_columns, gold_rows, retrieved_sub_tables, tables):

            max_decoder_source_length = self.config.data_loader.additional.max_decoder_source_length
            # concatenate the retrieved column with previous subtables
            # max_num_subtables = 100
            # processed_sub_tables = reduce_table(table, ranked_sub_tables[:min(max_num_subtables, len(ranked_sub_tables))])
            processed_sub_tables = reduce_table(table, ranked_sub_tables)
            # we do a binary search to find the index of which the concatenated sub table goes overflow
            def return_encoding(input_table):
                encoding = self.generator_tokenizer(
                    query=[input_text_sequence],
                    table=[pd.DataFrame.from_records(input_table["rows"], columns=input_table["header"])],
                    padding='longest',
                    truncate=True,
                    max_length=max_decoder_source_length+1,
                    return_tensors="pt",
                )
                return encoding
            
            def binary_search(start, end):
                if start >= end:
                    # print('found index: ', start)
                    # this index is very close to the boundary
                    encoding = return_encoding(processed_sub_tables[start])
                    if encoding.input_ids.shape[1] > max_decoder_source_length:
                        return binary_search(start-1, start-1)
                    else:
                        return start
                    
                mid = (start + end) // 2
                mid_sub_table = processed_sub_tables[mid]
                encoding = return_encoding(mid_sub_table)
                # print(f'strt {start} end {end} mid {mid} length {encoding.input_ids.shape[1]}')
                if encoding.input_ids.shape[1] > max_decoder_source_length:
                    # mid term overflow
                    return binary_search(start, mid-1)
                else:
                    return binary_search(mid+1, end)
            
            
            # check if search is needed
            if return_encoding(processed_sub_tables[-1]).input_ids.shape[1] > max_decoder_source_length:
                # print('total length:', len(processed_sub_tables))
                sep_index = binary_search(0, len(processed_sub_tables))
                # print('sep_index', sep_index)
                # input()
            else:
                sep_index = len(processed_sub_tables) - 1
            
            token_lengths = defaultdict(int)
            original_table_overflow = False
            for i in range(len(processed_sub_tables)):
                if i <= sep_index:
                    token_lengths[i] = 0  
                else:
                    token_lengths[i] = self.config.data_loader.additional.max_decoder_source_length+1
                    original_table_overflow = True
            
            

            # batch input_text_sequence with each of the retrieved sub_tables
            input_text_and_sub_tables = [(input_text_sequence, sub_table) for sub_table in processed_sub_tables]
            
            # set input_length for subtables
            for index, sub_table in enumerate(processed_sub_tables):
                sub_table['input_length'] = token_lengths[index]
                # this is for later evaluation
                sub_table['original_table_overflow'] = original_table_overflow
            
            # print(token_lengths)
            token_lengths = [(i, token_lengths[i]) for i in range(len(processed_sub_tables))]
            # print(token_lengths)
            if not is_training:
                
                

                """
                1. [1 1 (1 1 1 1 1) 0 0 0]
                2. [0 0 (0 0 0 0 0)]
                3. [(1 1 1 0 0) 0 0 0 0 0]
                4. [(0 0)] / [(1 1) 0] --> need to pad
                """
                if overflow_only and not original_table_overflow:
                    # if the original table does not overflow
                    # and we only reduce tables for overflow samples
                    # we put the whole table in the input and get predictions
                    token_lengths_reduced = [token_lengths[-1]]*n_docs
                    
                else:
                    token_lengths_reduced = [(i, length) for i, length in token_lengths if length <= self.config.data_loader.additional.max_decoder_source_length]

                    # print('after reduction:', token_lengths_reduced)
                    if len(token_lengths_reduced) == 0:
                        # Case 2: if it does not even fit one table
                        logger.warning(f"question {question_id} table {table['id']} can not fit!")
                        # keep the original table
                        token_lengths_reduced = token_lengths
                    elif len(token_lengths_reduced) < n_docs and len(token_lengths_reduced) > 0:
                        # Case 3
                        token_lengths_reduced = token_lengths[:min(n_docs, len(token_lengths))]
                    else:
                        # Case 1
                        token_lengths_reduced = token_lengths_reduced[-n_docs:]
                    
            else:
                # in training, just make sure the last element has all gold columns
                # token_lengths = [(i, 0) for i in range(len(processed_sub_tables))]

                earliest_element_that_covers_all_gold = -1

                if self.config.model_config.get('supervised', True) == True:
                    # pprint([subtable['sub_column_indice'] for subtable in processed_sub_tables])
                    for index, subtable in enumerate(processed_sub_tables):
                        cover_all_gold = set(gold_row_list).issubset(set(subtable['sub_row_indice']))
                        if cover_all_gold:
                            earliest_element_that_covers_all_gold = index
                            break
                    
                    # if earliest_element_that_covers_all_gold == -1:
                    #     # it is possible that (but very rare) the gold row is trimmed
                    #     logger.warning(f"question {question_id}: missing all gold reference.")
                    #     earliest_element_that_covers_all_gold = len(processed_sub_tables) - 1
                

                last_element_that_fits_table = -1
                for index, length in token_lengths:
                    if length <= self.config.data_loader.additional.max_decoder_source_length:
                        last_element_that_fits_table = index
                
                if last_element_that_fits_table == -1:
                    token_lengths_reduced = [token_lengths[earliest_element_that_covers_all_gold]]*n_docs
                else:
                    use_index = max(last_element_that_fits_table, earliest_element_that_covers_all_gold)
                    token_lengths_reduced = [token_lengths[use_index]]*n_docs
                

            if n_docs > len(token_lengths_reduced):
                # not enough to fit in n_docs
                # repeat the first table
                token_lengths_reduced = [token_lengths_reduced[0]]*(n_docs - len(token_lengths_reduced)) + token_lengths_reduced
            else:
                token_lengths_reduced = token_lengths_reduced[-n_docs:]
            
            # print('after truncation:', token_lengths_reduced)
            # print('gold_column_list', gold_column_list)
            extended_input_text_sequences += [input_text_and_sub_tables[i] for i, _ in token_lengths_reduced]
            input_sub_tables.append([input_text_and_sub_tables[i][1] for i, _ in token_lengths_reduced])

        # only train with the most rich item
        mask[:, -1] = 1
        # print(mask)
        
        # for index, input_text_sequence in enumerate(input_text_sequences):
        #     for sub_table in retrieved_sub_tables[index]:
        #         extended_input_text_sequences.append(
        #             (input_text_sequence, sub_table)
        #         )
        # pprint(extended_input_text_sequences)
        # input()

        targets = labels
        # We use tapex tokenizer here, so we need to input table
        
        encoding = self.generator_tokenizer(
            query=[text_table_tuple[0] for text_table_tuple in extended_input_text_sequences],
            table=[
                pd.DataFrame.from_records(text_table_tuple[1]["rows"], columns=text_table_tuple[1]["header"]) for text_table_tuple in extended_input_text_sequences
            ],
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt")
        generator_input_ids, generator_attention_mask, generator_token_type_ids = encoding.input_ids, encoding.attention_mask, encoding.get("token_type_ids", None)
        generator_input_ids = generator_input_ids.to(self.device)
        generator_attention_mask = generator_attention_mask.to(self.device)
        if generator_token_type_ids is not None:
            generator_token_type_ids = generator_token_type_ids.to(self.device)
        
        if targets is not None:
            generator_decoder_input_ids = shift_tokens_right(
                targets, self.generator.config.pad_token_id, self.generator.config.decoder_start_token_id
            )
        else:
            generator_decoder_input_ids = None

        return EasyDict(
            generator_input_text_sequences=extended_input_text_sequences,
            generator_input_sub_tables=input_sub_tables,
            generator_input_ids=generator_input_ids,
            generator_attention_mask=generator_attention_mask,
            generator_decoder_input_ids=generator_decoder_input_ids,
            generator_token_type_ids=generator_token_type_ids,
            generator_labels=targets,
            generator_loss_mask=mask,
        )



class ITRRagReduceMixModel(ITRRagModel):
    '''
    Class for Inner Table Retrieval RAG model (reduction version) Row-Wise
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        ITRRagModel.__init__(self, config, data_loader)

    def _get_top_sub_tables_from_n_gram(self,
                                    question_ids: List,
                                    query_hidden_states: torch.Tensor,
                                    n_docs: int,
                                    input_text_sequences: List,
                                    tables: List,
                                    **kwargs
                                    ):
        retrieved_sub_tables = []
        scores = []
        item_embeddings = []

        def return_n_gram(string, n_gram):
            words = string.lower().split()
            return [
                " ".join(words[index:index+n_gram]) for index in range(len(words)-n_gram+1)
            ]
        n_gram_list = self.config.model_config.get('n_gram_list', [1, 2, 3])

        for question_id, question, table, query_embedding in zip(question_ids, input_text_sequences, tables, query_hidden_states):
            sub_table_ids = self.question_id_to_sub_table_ids[question_id]
            sub_tables = [self.sub_table_id_to_content[question_id][sub_table_id] for sub_table_id in sub_table_ids]
            sub_table_embeddings = [self.sub_table_id_to_embeddings[question_id][sub_table_id] for sub_table_id in sub_table_ids] # list of emb_size

            # num_subtables x emb_size
            sub_table_embeddings = torch.stack(sub_table_embeddings)

            # move the embeddings to gpu
            sub_table_embeddings = sub_table_embeddings.to(query_embedding.device)

            sub_table_ratings = []
            for sub_table in sub_tables:
                if sub_table['sub_type'] == 'row_wise':
                    row_content = sub_table['header'] + sub_table['rows'][0]
                    sub_table_score = 0
                    for n_gram in n_gram_list:
                        question_n_grams = return_n_gram(question, n_gram)
                        row_content_n_grams = return_n_gram(" ".join(row_content), n_gram)
                        overlap = set(row_content_n_grams).intersection(set(question_n_grams))
                        sub_table_score += len(overlap)
                else:
                    column_content = sub_table['header'] + [
                        row[0] for row in sub_table['rows']
                    ]
                    sub_table_score = 0
                    for n_gram in n_gram_list:
                        question_n_grams = return_n_gram(question, n_gram)
                        column_content_n_grams = return_n_gram(" ".join(column_content), n_gram)
                        overlap = set(column_content_n_grams).intersection(set(question_n_grams))
                        sub_table_score += len(overlap)
                
                sub_table_ratings.append(sub_table_score)
            
            sub_table_ratings = torch.FloatTensor(sub_table_ratings).to(query_embedding.device)

            # # We return all ranked sub tables
            # Sort scores
            sorted_ratings, indices = torch.sort(sub_table_ratings, descending=True)
            ranked_scores = sorted_ratings # [:k]
            ranked_indices = indices # [:k]
            
            ranked_sub_tables = [self.sub_table_id_to_content[question_id][sub_table_ids[index]] for index in ranked_indices]

            # num_sub_tables x hidden_size
            ranked_item_embeddings = sub_table_embeddings[ranked_indices]
            # list of [num_sub_tables x hidden_size]
            item_embeddings.append(ranked_item_embeddings)
            scores.append(ranked_scores)
            retrieved_sub_tables.append(ranked_sub_tables)

        
        return EasyDict(dict(
            retrieved_sub_tables=retrieved_sub_tables,
            retrieved_sub_table_scores=scores,
            retrieved_sub_table_embeddings=item_embeddings,
        ))

    def prepare_inputs_for_generator(self, 
                input_text_sequences: List, 
                retrieved_sub_tables: List, 
                question_ids: List,
                gold_columns: List,
                gold_rows: List,
                tables: List,
                labels, 
                n_docs,
                is_training: bool = False):
        
        batch_size = len(input_text_sequences)

        extended_input_text_sequences = []

        overflow_only = ('overflow_only' in self.config.model_config.modules)
        original_sub_table_order = ('original_sub_table_order' in self.config.model_config.modules)
        suppress_ITR = ('suppress_ITR' in self.config.model_config.modules)
        shuffle_sub_table_order_in_training = ('shuffle_sub_table_order_in_training' in self.config.model_config.modules)
        random_reduce = ('random_reduce' in self.config.model_config.modules)
        random_score = ('random_score' in self.config.model_config.modules)
        
        # prepare sub tables here
        # e.g. concatenate subtables, postprocessing retrieval results...

        def reduce_table(original_table, ranked_sub_tables):
            table_pd = pd.DataFrame.from_records(original_table['rows'], columns=original_table['header'])
            current_row_indices = []
            current_column_indices = []
            
            processed_sub_tables = []

            if random_score:
                random.shuffle(ranked_sub_tables)

            # all_rows = range(len(table_pd))
            for i in ranked_sub_tables:
                if i['sub_type'] == 'column_wise':
                    current_column_indices.extend(i['sub_column_indice'])
                else:
                    current_row_indices.extend(i['sub_row_indice'])
                
                if len(current_column_indices) < self.config.model_config.min_columns or len(current_row_indices) == 0:
                    # only rows or columns don't compose a sub table
                    # force at least two columns
                    continue

                processed_sub_table = deepcopy(i)
                processed_sub_table['sub_row_indice'] = current_row_indices.copy()
                processed_sub_table['sub_column_indice'] = current_column_indices.copy()
                
                if original_sub_table_order:
                    processed_sub_table['sub_row_indice'] = sorted(processed_sub_table['sub_row_indice'])
                    processed_sub_table['sub_column_indice'] = sorted(processed_sub_table['sub_column_indice'])

                if shuffle_sub_table_order_in_training and is_training:
                    random.shuffle(processed_sub_table['sub_row_indice'])
                    random.shuffle(processed_sub_table['sub_column_indice'])

                selected_table_pd = table_pd.iloc[processed_sub_table['sub_row_indice'], processed_sub_table['sub_column_indice']]
                processed_sub_table['header'] = selected_table_pd.columns.values.tolist()
                processed_sub_table['rows'] = selected_table_pd.values.tolist()
                processed_sub_table['types'] = [
                    t for col_index, t in enumerate(original_table['types']) if col_index in current_column_indices
                ]
                processed_sub_tables.append(processed_sub_table)
            return processed_sub_tables

        def random_reduce_table(original_table, ranked_sub_tables):
            table_pd = pd.DataFrame.from_records(original_table['rows'], columns=original_table['header'])
            num_rows = len(original_table['rows'])
            num_cols = len(original_table['header'])
            current_row_indices = list(range(num_rows))
            current_column_indices = list(range(num_cols))
            
            processed_sub_tables = []

            # all_rows = range(len(table_pd))
            for i in ranked_sub_tables:
                random.shuffle(current_row_indices)
                random.shuffle(current_column_indices)

                processed_sub_table = deepcopy(i)
                processed_sub_table['sub_row_indice'] = current_row_indices.copy()
                processed_sub_table['sub_column_indice'] = current_column_indices.copy()
                
                if shuffle_sub_table_order_in_training and is_training:
                    random.shuffle(processed_sub_table['sub_row_indice'])
                    random.shuffle(processed_sub_table['sub_column_indice'])

                selected_table_pd = table_pd.iloc[processed_sub_table['sub_row_indice'], processed_sub_table['sub_column_indice']]
                processed_sub_table['header'] = selected_table_pd.columns.values.tolist()
                processed_sub_table['rows'] = selected_table_pd.values.tolist()
                processed_sub_table['types'] = [
                    t for col_index, t in enumerate(original_table['types']) if col_index in current_column_indices
                ]
                processed_sub_tables.append(processed_sub_table)
            return processed_sub_tables

        mask = torch.ones(batch_size, n_docs).to(self.device)
        input_sub_tables = []
        for index, question_id, input_text_sequence, gold_column_list, gold_row_list, ranked_sub_tables, table in zip(range(batch_size), question_ids, input_text_sequences, gold_columns, gold_rows, retrieved_sub_tables, tables):

            max_decoder_source_length = self.config.data_loader.additional.max_decoder_source_length
            # concatenate the retrieved column with previous subtables
            # max_num_subtables = 100
            # processed_sub_tables = reduce_table(table, ranked_sub_tables[:min(max_num_subtables, len(ranked_sub_tables))])
            if random_reduce:
                processed_sub_tables = random_reduce_table(table, ranked_sub_tables)
            else:
                processed_sub_tables = reduce_table(table, ranked_sub_tables)
            # we do a binary search to find the index of which the concatenated sub table goes overflow
            def return_encoding(input_table):
                encoding = self.generator_tokenizer(
                    query=[input_text_sequence],
                    table=[pd.DataFrame.from_records(input_table["rows"], columns=input_table["header"])],
                    padding='longest',
                    truncate=True,
                    max_length=max_decoder_source_length+1,
                    return_tensors="pt",
                )
                return encoding
            
            def binary_search(start, end, verbose=False):
                if start >= end:
                    if verbose:
                        print('found index: ', start)
                    # this index is very close to the boundary
                    encoding = return_encoding(processed_sub_tables[start])
                    if encoding.input_ids.shape[1] > max_decoder_source_length:
                        if start == 0:
                            logger.error(f"question {question_id} exceeds limit.")
                            return -1
                        else:
                            return binary_search(start-1, start-1, verbose)
                    else:
                        return start
                    
                mid = (start + end) // 2
                mid_sub_table = processed_sub_tables[mid]
                encoding = return_encoding(mid_sub_table)
                if verbose:
                    print(f'strt {start} end {end} mid {mid} length {encoding.input_ids.shape[1]}')
                if encoding.input_ids.shape[1] > max_decoder_source_length:
                    # mid term overflow
                    return binary_search(start, mid-1, verbose)
                else:
                    return binary_search(mid+1, end, verbose)
            
            
            
            # check if search is needed
            last_item_encoding = return_encoding(processed_sub_tables[-1])
            if last_item_encoding.input_ids.shape[1] > max_decoder_source_length:
                # print('total length:', len(processed_sub_tables))
                if not random_reduce:
                    sep_index = binary_search(0, len(processed_sub_tables))
                else:
                    sep_index = -1
            else:
                sep_index = len(processed_sub_tables) - 1
            
            token_lengths = defaultdict(int)
            original_table_overflow = False
            for i in range(len(processed_sub_tables)):
                if i <= sep_index:
                    token_lengths[i] = 0  
                else:
                    token_lengths[i] = self.config.data_loader.additional.max_decoder_source_length+1
                    original_table_overflow = True
            

            # batch input_text_sequence with each of the retrieved sub_tables
            input_text_and_sub_tables = [(input_text_sequence, sub_table) for sub_table in processed_sub_tables]
            
            # set input_length for subtables
            for index, sub_table in enumerate(processed_sub_tables):
                sub_table['input_length'] = token_lengths[index]
                # this is for later evaluation
                sub_table['original_table_overflow'] = original_table_overflow
            
            # print(token_lengths)
            token_lengths = [(i, token_lengths[i]) for i in range(len(processed_sub_tables))]

            # print(token_lengths)
            if not is_training:

                """
                1. [1 1 (1 1 1 1 1) 0 0 0]
                2. [0 0 (0 0 0 0 0)]
                3. [(1 1 1 0 0) 0 0 0 0 0]
                4. [(0 0)] / [(1 1) 0] --> need to pad
                """
                if (overflow_only and not original_table_overflow) or suppress_ITR:
                    # if the original table does not overflow
                    # and we only reduce tables for overflow samples
                    # we put the whole table in the input and get predictions
                    token_lengths_reduced = [token_lengths[-1]]*n_docs
                    if random_reduce:
                        # in random reduce, all sub-tables are available
                        token_lengths_reduced = [token_lengths[i] for i in range(n_docs)]
                
                else:
                    token_lengths_reduced = [(i, length) for i, length in token_lengths if length <= self.config.data_loader.additional.max_decoder_source_length]

                    if len(token_lengths_reduced) == 0:
                        # Case 2: if it does not even fit one table
                        logger.warning(f"question {question_id} table {table['id']} can not fit!")
                        # keep the original table
                        token_lengths_reduced = token_lengths
                    elif len(token_lengths_reduced) < n_docs and len(token_lengths_reduced) > 0:
                        # Case 3
                        token_lengths_reduced = token_lengths[:min(n_docs, len(token_lengths))]
                    else:
                        # Case 1
                        token_lengths_reduced = token_lengths_reduced[-n_docs:]
                
                
            else:
                # in training, just make sure the last element has all gold columns
                # token_lengths = [(i, 0) for i in range(len(processed_sub_tables))]

                earliest_element_that_covers_all_gold = -1

                if self.config.model_config.get('supervised', True) == True:
                    # pprint([subtable['sub_column_indice'] for subtable in processed_sub_tables])
                    for index, subtable in enumerate(processed_sub_tables):
                        cover_all_gold = set(gold_row_list).issubset(set(subtable['sub_row_indice'])) and set(gold_column_list).issubset(set(subtable['sub_column_indice']))
                        if cover_all_gold:
                            earliest_element_that_covers_all_gold = index
                            break
                    
                    # if earliest_element_that_covers_all_gold == -1:
                    #     # it is possible that (but very rare) the gold row is trimmed
                    #     logger.warning(f"question {question_id}: missing all gold reference.")
                    #     earliest_element_that_covers_all_gold = len(processed_sub_tables) - 1
                

                last_element_that_fits_table = -1
                for index, length in token_lengths:
                    if length <= self.config.data_loader.additional.max_decoder_source_length:
                        last_element_that_fits_table = index
                
                if last_element_that_fits_table == -1:
                    token_lengths_reduced = [token_lengths[earliest_element_that_covers_all_gold]]*n_docs
                else:
                    use_index = max(last_element_that_fits_table, earliest_element_that_covers_all_gold)
                    token_lengths_reduced = [token_lengths[use_index]]*n_docs
                
            if n_docs > len(token_lengths_reduced):
                # not enough to fit in n_docs
                # repeat the first table
                token_lengths_reduced = [token_lengths_reduced[0]]*(n_docs - len(token_lengths_reduced)) + token_lengths_reduced
            else:
                token_lengths_reduced = token_lengths_reduced[-n_docs:]
            
            # print('gold_column_list', gold_column_list)
            extended_input_text_sequences += [input_text_and_sub_tables[i] for i, _ in token_lengths_reduced]
            input_sub_tables.append([input_text_and_sub_tables[i][1] for i, _ in token_lengths_reduced])
        # only train with the most rich item
        mask[:, -1] = 1
        # print(mask)
        
        # for index, input_text_sequence in enumerate(input_text_sequences):
        #     for sub_table in retrieved_sub_tables[index]:
        #         extended_input_text_sequences.append(
        #             (input_text_sequence, sub_table)
        #         )
        # pprint(extended_input_text_sequences)

        targets = labels
        # We use tapex tokenizer here, so we need to input table
        
        encoding = self.generator_tokenizer(
            query=[text_table_tuple[0] for text_table_tuple in extended_input_text_sequences],
            table=[
                pd.DataFrame.from_records(text_table_tuple[1]["rows"], columns=text_table_tuple[1]["header"]) for text_table_tuple in extended_input_text_sequences
            ],
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt")
        
        generator_input_ids, generator_attention_mask, generator_token_type_ids = encoding.input_ids, encoding.attention_mask, encoding.get("token_type_ids", None)
        generator_input_ids = generator_input_ids.to(self.device)
        generator_attention_mask = generator_attention_mask.to(self.device)
        if generator_token_type_ids is not None:
            generator_token_type_ids = generator_token_type_ids.to(self.device)
        
        if targets is not None:
            generator_decoder_input_ids = shift_tokens_right(
                targets, self.generator.config.pad_token_id, self.generator.config.decoder_start_token_id
            )
        else:
            generator_decoder_input_ids = None

        # pprint(self.generator_tokenizer.batch_decode(generator_input_ids))
        # print(generator_input_ids.shape)
        # input()

        return EasyDict(
            generator_input_text_sequences=extended_input_text_sequences,
            generator_input_sub_tables=input_sub_tables,
            generator_input_ids=generator_input_ids,
            generator_attention_mask=generator_attention_mask,
            generator_decoder_input_ids=generator_decoder_input_ids,
            generator_token_type_ids=generator_token_type_ids,
            generator_labels=targets,
            generator_loss_mask=mask,
        )
