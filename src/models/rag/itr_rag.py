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
from transformers import TapasTokenizer, TapasConfig, TapasForQuestionAnswering
from models.tapas.custom_tokenizer import CustomTapasTokenizer

from transformers.models.rag.retrieval_rag import CustomHFIndex, CanonicalHFIndex
import pytorch_lightning as pl
import glob
import time
import random

from pprint import pprint
import pandas as pd
from copy import deepcopy
from utils.text_utils import _split_thousands, convert_to_float

import logging
logger = logging.getLogger(__name__)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class ITRRagModel(pl.LightningModule):
    '''
    Class for Inner Table Retrieval RAG model
    Column-wise Addition
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        super().__init__()

        self.config = config
        self.data_loader = data_loader
        self.retriever_tokenizer = data_loader.tokenizer
        self.generator_tokenizer = data_loader.decoder_tokenizer
        self.LOAD_LOCAL = False
        
        # Initialising question encoder
        QueryEncoderModelClass = globals()[self.config.model_config.QueryEncoderModelClass]
        QueryEncoderConfigClass = globals()[self.config.model_config.QueryEncoderConfigClass]

        if "$" in self.config.model_config.QueryEncoderModelVersion:
            # it is a local path, we need to load parameters manually to avoid a bug of Huggingface
            self.config.model_config.QueryEncoderModelVersion = os.path.join(self.config.EXPERIMENT_FOLDER, self.config.model_config.QueryEncoderModelVersion.replace('$', ''))
            self.LOAD_LOCAL = True
        
        question_encoder_model_config = QueryEncoderConfigClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion)
        self.question_encoder = QueryEncoderModelClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion,
                                                    config=question_encoder_model_config)

        self.retiever_hidden_size = question_encoder_model_config.hidden_size

        # Initialising generator
        GeneratorModelClass = globals()[self.config.model_config.GeneratorModelClass]
        GeneratorConfigClass = globals()[self.config.model_config.GeneratorConfigClass]
        generator_model_config = GeneratorConfigClass.from_pretrained(self.config.model_config.GeneratorModelVersion)


        
        if 'tapex' in self.config.model_config.GeneratorModelVersion:
            # IMPORTANT: the initial BART model's decoding is penalized by no_repeat_ngram_size, and thus
            # we should disable it here to avoid problematic generation (used for TAPEX model only)
            generator_model_config.no_repeat_ngram_size = 0
            generator_model_config.max_length = 1024
            generator_model_config.early_stopping = False

        self.generator = GeneratorModelClass.from_pretrained(self.config.model_config.GeneratorModelVersion,
                                                    config=generator_model_config)
        
        self.question_encoder.resize_token_embeddings(len(self.retriever_tokenizer))
        self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        
        # if self.LOAD_LOCAL:
        #     self.config.model_config.QueryEncoderCheckpoint = os.path.join(self.config.EXPERIMENT_FOLDER, self.config.model_config.QueryEncoderCheckpoint.replace('$', ''))
        #     checkpoint = torch.load(self.config.model_config.QueryEncoderCheckpoint)
        #     pretrained_state_dict = checkpoint['state_dict']
        #     model_state_dict = self.state_dict()
        #     # print(pretrained_state_dict.keys()) # model.query_encoder.question_encoder.
        #     # print('------------------------------')
        #     # print(model_state_dict.keys()) # question_encoder.question_encoder.
            
        #     # 1. find parameters that can be loaded
            
        #     pretrained_dict = {k.replace('model.query_encoder.question_encoder.', "question_encoder.question_encoder."): v for k, v in pretrained_state_dict.items()}
            
        #     # print(pretrained_dict.keys())
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict.keys()}
        #     print('Loading pretrained weights', [k for k in pretrained_dict.keys()])
        #     # 2. overwrite entries in the existing state dict
        #     model_state_dict.update(pretrained_dict)
        #     # 3. load the new state dict
        #     self.load_state_dict(model_state_dict)

        if self.config.model_config.get("LoadGeneratorModelPath", None) is not None:
            generator_checkpoint_path = self.config.model_config.LoadGeneratorModelPath
            checkpoint = torch.load(generator_checkpoint_path, map_location=self.device)
            pretrained_state_dict = checkpoint['state_dict']
            model_state_dict = self.state_dict()
            # print(pretrained_state_dict.keys()) # model.lm_head.weight
            # print('------------------------------')
            # print(model_state_dict.keys()) # generator.lm_head.weight
            
            # 1. find parameters that can be loaded
            
            pretrained_dict = {"generator." + k[6:]: v for k, v in pretrained_state_dict.items() if k.startswith("model.")}
            
            # print(pretrained_dict.keys())
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict.keys()}
            print('Loading pretrained weights', [k for k in pretrained_dict.keys()])
            # 2. overwrite entries in the existing state dict
            model_state_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_state_dict)
            del pretrained_dict
            del checkpoint
        

        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        # label smoother imported from huggingface transformers
        label_smoothing_factor = self.config.train.additional.get('label_smoothing_factor', 0)
        if label_smoothing_factor != 0:
            from transformers.trainer_pt_utils import LabelSmoother
            self.label_smoother = LabelSmoother(epsilon=label_smoothing_factor)
        else:
            self.label_smoother = None
        
        self.retrieve = self.main_retrieve
        if "n_gram_retrieval" in self.config.model_config.modules:
            self.get_top_sub_tables_from_index = self._get_top_sub_tables_from_n_gram
        else:
            self.get_top_sub_tables_from_index = self._get_top_sub_tables_from_index

        self.init_retrieval()

    def init_retrieval(self):
        """
        Load sub-table embeddings from files
        """
        self.sub_table_id_to_embeddings = defaultdict(dict) # question_id -> sub_table_id -> embedding
        self.sub_table_id_to_content = defaultdict(dict) # question_id -> sub_table_id -> content
        self.question_id_to_sub_table_ids = {} # question_id -> sub_table_id
        self.question_id_to_query_embedding = {}
        total_num_questions = 0
        index_files = self.config.model_config.index_files
        for split, split_index_file_path in index_files.index_paths.items():
            split_index_file_path = os.path.join(self.config.EXPERIMENT_FOLDER, split_index_file_path)
            all_shard_files = glob.glob(f"{split_index_file_path}/static_index_*.pkl")
            for file_path in all_shard_files:
                logger.info(f'Loading index file {file_path}')
                with open(file_path, 'rb') as f:
                    index_data = pickle.load(f)
                item_embeddings=index_data['item_embeddings']
                question_ids=index_data['question_ids']
                sub_tables=index_data['sub_tables']
                question_id_to_sub_table_index=index_data['question_id_to_sub_table_index']
                
                question_id_to_query_index=index_data['question_id_to_query_index']
                query_embeddings=index_data['query_embeddings']
                
                # Load all sub_table and corresponding embeddings into dict
                for question_id in question_ids:
                    
                    total_num_questions += 1
                    start, end = question_id_to_sub_table_index[question_id]
                    current_sub_tables = sub_tables[start:end]
                    current_sub_table_embeddings = item_embeddings[start:end]

                    self.question_id_to_query_embedding[question_id] = query_embeddings[question_id_to_query_index[question_id]]

                    if self.question_id_to_sub_table_ids.get(question_id, None) is not None:
                        logger.warning(f'{file_path} found duplicated question_id={question_id} sub_tables {len(current_sub_tables)}')
                    
                    for sub_table, sub_table_emb in zip(current_sub_tables, current_sub_table_embeddings):
                        sub_table_id = sub_table['id']
                        self.question_id_to_sub_table_ids.setdefault(question_id, []).append(sub_table['id'])
                        self.sub_table_id_to_content[question_id][sub_table['id']] = sub_table
                        self.sub_table_id_to_embeddings[question_id][sub_table['id']] = sub_table_emb
                # break # debug, load only one shard
                logger.info(f"number of questions {len(self.question_id_to_sub_table_ids)}")
                logger.info(f"number of sub tables {len(self.sub_table_id_to_content)}")
        
        for question_id, list_sub_tables in self.question_id_to_sub_table_ids.items():
            self.question_id_to_sub_table_ids[question_id] = list(set(list_sub_tables))

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

    def _get_top_sub_tables_from_index(self,
                                    question_ids: List,
                                    query_hidden_states: torch.Tensor,
                                    n_docs: int,
                                    **kwargs):
        retrieved_sub_tables = []
        scores = []
        item_embeddings = []

        # num_questions = len(question_ids)
        # random_question_ids = random.choices(list(self.question_id_to_sub_table_ids.keys()), k=num_questions)
        # print(question_ids)

        for question_id, query_embedding in zip(question_ids, query_hidden_states):
            sub_table_ids = self.question_id_to_sub_table_ids[question_id]

            sub_table_embeddings = [self.sub_table_id_to_embeddings[question_id][sub_table_id] for sub_table_id in sub_table_ids] # list of emb_size
            
            # num_subtables x emb_size
            sub_table_embeddings = torch.stack(sub_table_embeddings)

            # move the embeddings to gpu
            sub_table_embeddings = sub_table_embeddings.to(query_embedding.device)
            # print('sub_table_embeddings', sub_table_embeddings)
            ## Get scores
            # num_subtables
            
            sub_table_ratings = torch.matmul(query_embedding, sub_table_embeddings.t()) 
            # sub_table_ratings = torch.matmul(self.question_id_to_query_embedding[question_id].to(query_embedding.device), sub_table_embeddings.t()) 

            # We return all ranked sub tables

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

    def main_retrieve(self, 
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, 
                    labels: torch.Tensor, 
                    question_ids: List, 
                    input_text_sequences: List, 
                    n_docs: int,
                    tables: List,
                    **kwargs):
        """ Main retrieval function, retrieve documents using retriever
        Args:
            input_ids (torch.Tensor): [description]
            attention_mask (torch.Tensor): [description]
            labels (torch.Tensor): [description]
            question_ids (List): [description]
            input_text_sequences (List): [description]
            n_docs ([type], optional): [description]. Defaults to None.
        Returns:
            [type]: [description]
        """

        batch_size = input_ids.shape[0]
        
        # Use question_encoder to encode question inputs
        # query_outputs = self.question_encoder(input_ids=input_ids,
        #                                     attention_mask=attention_mask)
        # question_hidden_states = query_outputs.pooler_output
        # print('question_hidden_states', question_hidden_states)

        # for question_id in question_ids:
        #     print(self.question_id_to_query_embedding[question_id])
        question_hidden_states = torch.stack([self.question_id_to_query_embedding[question_id] for question_id in question_ids]).to(input_ids.device)
        
        start_time = time.time()
        
        
        retrieval_res = self.get_top_sub_tables_from_index(
            question_ids=question_ids,
            query_hidden_states=question_hidden_states,
            n_docs=n_docs,
            input_text_sequences=input_text_sequences,
            tables=tables,
        )
        retrieved_sub_tables=retrieval_res.retrieved_sub_tables
        retrieved_sub_table_scores=retrieval_res.retrieved_sub_table_scores
        retrieved_sub_table_embeddings=retrieval_res.retrieved_sub_table_embeddings
        

        # print(
        #     f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
        # )

        # question_hidden_states: batch_size x hidden_size
        # item_hidden_states: batch_size x n_docs x hidden_size
        
        # batch_size x n_docs
        # retrieved_sub_table_scores = torch.stack(retrieved_sub_table_scores)
        # print('retrieved_sub_table_scores', retrieved_sub_table_scores.shape)

        
        assert len(retrieved_sub_tables) == batch_size

        return EasyDict(
            retrieved_sub_tables=retrieved_sub_tables,
            retrieved_sub_table_scores=retrieved_sub_table_scores,
            question_hidden_states=question_hidden_states,
        )


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

        overflow_only = ('overflow_only' in self.config.model_config.modules)
        original_sub_table_order = ('original_sub_table_order' in self.config.model_config.modules)

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


        extended_input_text_sequences = []
        input_sub_tables = []
        mask = torch.ones(batch_size, n_docs).to(self.device)

        for index, question_id, input_text_sequence, gold_column_list, ranked_sub_tables, table in zip(range(batch_size), question_ids, input_text_sequences, gold_columns, retrieved_sub_tables, tables):
            
            processed_sub_tables = reduce_table(table, ranked_sub_tables)
            # we set the minimum #col to be 2 here
            if len(processed_sub_tables) > 1:
                processed_sub_tables = processed_sub_tables[1:]

            # batch input_text_sequence with each of the retrieved sub_tables
            input_text_and_sub_tables = [(input_text_sequence, sub_table) for sub_table in processed_sub_tables]

            if not is_training:
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

                if overflow_only and not original_table_overflow:
                    # if the original table does not overflow
                    # and we only trim tables for overflow samples
                    # we put the whole table in the input and get predictions
                    token_lengths = [(i, token_lengths[i]) for i in range(len(processed_sub_tables))]
                    token_lengths = [token_lengths[-1]]*n_docs
                else:
                    # if the original table overflows
                    token_lengths = [(i, token_lengths[i]) for i in range(len(processed_sub_tables))]

            else:
                # in training, just make sure the last element has all gold columns
                token_lengths = [(i, 0) for i in range(len(processed_sub_tables))]
                
                # earliest_element_that_covers_all_gold = len(token_lengths)-1
                # use earliest_element_that_covers_all_gold = -1 for unsupervised case
                earliest_element_that_covers_all_gold = -1

                # pprint([subtable['sub_column_indice'] for subtable in processed_sub_tables])
                for index, subtable in enumerate(processed_sub_tables):
                    cover_all_gold = set(gold_column_list).issubset(set(subtable['sub_column_indice']))
                    if cover_all_gold:
                        earliest_element_that_covers_all_gold = index
                        break

                earliest_element_that_covers_all_gold = max(
                    earliest_element_that_covers_all_gold, 
                    self.config.data_loader.additional.num_knowledge_passages-1
                )

                earliest_element_that_covers_all_gold = min(
                    earliest_element_that_covers_all_gold,
                    len(token_lengths)-1
                )


                token_lengths = [token_lengths[earliest_element_that_covers_all_gold]]*n_docs
                

            if n_docs > len(token_lengths):
                # repeat the last item
                token_lengths_reduced = token_lengths + [token_lengths[-1]] * (n_docs - len(processed_sub_tables))
            else:
                # take the first n_docs sub tables
                token_lengths_reduced = token_lengths[:n_docs]

            extended_input_text_sequences += [input_text_and_sub_tables[i] for i, _ in token_lengths_reduced]
            input_sub_tables.append([input_text_and_sub_tables[i][1] for i, _ in token_lengths_reduced])

        # ensure that when not all gold col are retrieved, the model tries to learn sth.
        mask[:, -1] = 1

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

    def forward(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      gold_columns: List,
                      gold_rows: List,
                      question_ids: List,
                      input_text_sequences: List,
                      tables: List,
                    **kwargs):
        
        batch_size = input_ids.shape[0]

        n_docs = 1 #self.config.data_loader.additional.num_knowledge_passages

        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            question_ids=question_ids, 
            input_text_sequences=input_text_sequences, 
            n_docs=n_docs,
            tables=tables,
        )

        retrieved_sub_tables=retrieval_results.retrieved_sub_tables
        retrieved_sub_table_scores=retrieval_results.retrieved_sub_table_scores
        question_hidden_states=retrieval_results.question_hidden_states

        # populate the labels
        labels = labels.repeat_interleave(n_docs, 0)

        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(
            input_text_sequences=input_text_sequences,
            retrieved_sub_tables=retrieved_sub_tables,
            labels=labels, 
            question_ids=question_ids,
            gold_columns=gold_columns,
            gold_rows=gold_rows,
            tables=tables,
            n_docs=n_docs,
            is_training=True,
        )

        generator_outputs = self.generator(
                            input_ids=generator_inputs.generator_input_ids,
                            attention_mask=generator_inputs.generator_attention_mask,
                            decoder_input_ids=generator_inputs.generator_decoder_input_ids,
                            labels=labels,
                            return_dict=True)
        
        logits = generator_outputs.logits

        if self.label_smoother is not None:
            smoothed_loss = self.label_smoother(generator_outputs, labels)
        else:
            smoothed_loss = self.loss_fct(
                logits.view(-1, self.generator.config.vocab_size), 
                labels.view(-1)
            )
        
        loss_dict = EasyDict(
            nll_loss=smoothed_loss,
        )

        # loss_dict = self.get_loss(
        #     seq_logits=logits,
        #     doc_scores=retrieved_sub_table_scores,
        #     target=generator_inputs.generator_labels,
        #     exclude_bos_score=False,
        #     n_docs=n_docs,
        #     loss_mask=generator_inputs.generator_loss_mask,
        # )

        # aggregate loss
        total_loss = 0
        for loss_name, loss_ratio in self.config.model_config.loss_ratio.items():
            if loss_ratio != 0:
                loss = loss_dict[loss_name]
                total_loss += loss * loss_ratio
        
        # function to extract grad
        def set_grad(var):
            def hook(grad):
                var.grad = grad
                print('setting grad:', grad)
            return hook
        
        # answers = kwargs.get('answers', None)
        # assert answers is not None
        # retrieval_labels = self.get_retrieval_labels(
        #     batch_answers=answers,
        #     batch_retrieved_docs=retrieved_docs,
        # )
        # print(F.softmax(doc_scores, dim=-1))
        # print(retrieval_labels)
        # print('-------------')
        # # register_hook for Z
        # doc_scores.register_hook(set_grad(doc_scores))
        # input()


        return EasyDict(loss=total_loss,
                        loss_dict=loss_dict,
                        logits=logits,
                        doc_scores=retrieved_sub_table_scores)


    def generate(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      gold_columns: List,
                      gold_rows: List,
                      question_ids: List,
                      input_text_sequences: List,
                      tables: List,
                      n_docs: int=None,
                      **kwargs):

        batch_size = input_ids.shape[0]
        
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages

        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            question_ids=question_ids, 
            input_text_sequences=input_text_sequences, 
            n_docs=n_docs,
            tables=tables,
        )

        retrieved_sub_tables=retrieval_results.retrieved_sub_tables
        retrieved_sub_table_scores=retrieval_results.retrieved_sub_table_scores
        question_hidden_states=retrieval_results.question_hidden_states

        doc_scores = retrieved_sub_table_scores

        # populate labels
        labels = labels.repeat_interleave(n_docs, 0)

        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(
            input_text_sequences=input_text_sequences,
            retrieved_sub_tables=retrieved_sub_tables,
            labels=labels, 
            question_ids=question_ids,
            gold_columns=gold_columns,
            gold_rows=gold_rows,
            tables=tables,
            n_docs=n_docs,
            is_training=False,
        )
        
        # pprint(input_text_sequences)
        # pprint(retrieved_sub_tables)
        # print('--------------')
        # new_labels = labels
        # new_labels[new_labels==-100] = 1
        # print(self.generator_tokenizer.batch_decode(new_labels))
        # pprint(gold_columns)

        # pprint(self.generator_tokenizer.batch_decode(generator_inputs.generator_input_ids))
        # print(generator_inputs.generator_attention_mask)


        ####### Approach 1: Get encoder outputs first and then pass to generate()
        # Get encoder outputs first
        # test_batch = EasyDict({
        #     'input_ids': generator_inputs.generator_input_ids,
        #     'attention_mask': generator_inputs.generator_attention_mask,
        #     'return_dict': True,
        # })

        # encoder_outputs = self.generator.model.encoder(
        #     **test_batch
        # )

        # # Get decoder outputs from encoder_outputs
        # test_batch = {
        #     'encoder_outputs': encoder_outputs,
        #     "max_length": self.config.data_loader.additional.max_target_length,
        #     # 'output_scores': True,
        #     # 'return_dict_in_generate': True,
        #     'num_beams': 1, #self.config.model_config.num_beams,
        # }
        # generation_results = self.generator.generate(**test_batch)
        # generation_outputs = generation_results
        # # generation_outputs = generation_results['sequences']
        # # generation_seq_scores = generation_results['sequences_scores']
        # generation_outputs_decoded = self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)
        # print(generation_outputs_decoded)

        ####### Approach 2: Use generate() directly
        test_batch = EasyDict({
            'input_ids': generator_inputs.generator_input_ids,
            'attention_mask': generator_inputs.generator_attention_mask,
            "max_length": self.config.data_loader.additional.max_target_length,
            "num_beams": self.config.model_config.num_beams,
            "return_dict_in_generate": True,
            'output_scores': True
        })
        generation_results = self.generator.generate(
            **test_batch
        )

        generation_outputs = generation_results['sequences']
        generation_seq_scores = generation_results['sequences_scores']
        generation_outputs_decoded = self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)
        # print(generation_outputs_decoded)
        # print(generation_seq_scores)
        # print(self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=False))


        # Find answer proposals from n_docs outputs for each question
        outputs = []
        generation_outputs_for_docs = []

        # reshape generation_outputs
        generation_outputs = generation_outputs.reshape(batch_size, n_docs, -1)
        # shifted_generation_outputs = shifted_generation_outputs.reshape(batch_size, n_docs, -1)
        
        loss_with_doc_scores = -generation_seq_scores.reshape(batch_size, n_docs)
        
        new_labels = labels
        new_labels[new_labels==-100] = 1
        

        for b in range(batch_size):
            # use topk to get indices of top candidates
            top_cand_inds = (-loss_with_doc_scores[b]).topk(1)[1]
            if 'force_select_last' in self.config.model_config.modules:
                top_cand_inds = [-1]
            outputs.append(generation_outputs[b, top_cand_inds])
            answer_proposals = generation_outputs_decoded[b*n_docs:(b+1)*n_docs]
            generation_outputs_for_docs.append(answer_proposals)
            # pprint(answer_proposals)

            # answer_str = self.generator_tokenizer.batch_decode(new_labels, skip_special_tokens=True)[b]
            # pred_str = answer_proposals[top_cand_inds[0]]

            # if pred_str == answer_str and answer_proposals[-1] != pred_str:
            #     print("scores in retrieval")
            #     for sub_table, sub_table_score in zip(retrieved_sub_tables[b], retrieved_sub_table_scores[b]):
            #         print(f"{sub_table['sub_type']} \t col_index: {sub_table['sub_column_indice']} \t row_index: {sub_table['sub_row_indice']} \t score: {sub_table_score}")
            #     print('gold_columns', gold_columns[b])
            #     print('gold_rows', gold_rows[b])
            #     print("<<<<<<<<<<<<<<<<<<")
            #     print("input sub tables")
            #     for sub_table in generator_inputs.generator_input_sub_tables[b]:
            #         pprint(sub_table)
            #         print("<<<<<<<<<<<<<<<<<<")
                
            #     print("question", input_text_sequences[b])
            #     print("answers", answer_str)
            #     print('gold_columns', gold_columns[b])
            #     print('gold_rows', gold_rows[b])
            #     print('decision', top_cand_inds)
            #     print("final prediction", pred_str)
                
            #     print('generation_outputs_for_docs', answer_proposals)
            #     input()
        
        outputs = torch.cat(outputs)
        # print(self.generator_tokenizer.batch_decode(outputs))
        # input()
        # # Re-forward the generator, and use generation outputs as labels
        # # obtain the loss of each (question, passage) pair

        # # shift genereation results to left by one token
        # # </s> <s> answer </s> --> <s> answer </s> (0)

        # pad_token_id = self.generator.config.pad_token_id

        # shifted_generation_outputs = torch.ones_like(generation_outputs) * pad_token_id
        # shifted_generation_outputs[:, :-1] = generation_outputs[:, 1:]
        
        # # pprint(self.generator_tokenizer.batch_decode(generation_outputs))
        # # print('input:', generation_outputs)
        # # pprint(self.generator_tokenizer.batch_decode(shifted_generation_outputs))
        # # print('output:', shifted_generation_outputs)
        # # input()
        # forward_results = self.generator(
        #                     input_ids=generator_inputs.generator_input_ids,
        #                     attention_mask=generator_inputs.generator_attention_mask,
        #                     # encoder_outputs=encoder_outputs, # use pre-computed encoder outputs
        #                     decoder_input_ids=generation_outputs,
        #                     return_dict=True)
        
        # # Loss for each pair can be computed now
        # logits = forward_results.logits

        # # loss: batch_size x n_docs x seq_len
        # loss_dict = self.get_loss(
        #     seq_logits=logits,
        #     doc_scores=doc_scores,
        #     target=shifted_generation_outputs, # use generation outputs as labels
        #     reduce_loss=False, # do not reduce loss
        #     exclude_bos_score=False,
        #     ignore_index=pad_token_id,
        #     n_docs=n_docs,
        # )

        # loss = loss_dict.nll_loss

        # decode the generation outputs
        # generation_outputs_decoded = self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)
        # print(self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=False))

        # reshape generation_outputs
        # generation_outputs = generation_outputs.reshape(batch_size, n_docs, -1)
        # shifted_generation_outputs = shifted_generation_outputs.reshape(batch_size, n_docs, -1)
        
        # loss_with_doc_scores = loss.sum(-1)

        # for b in range(batch_size):
        #     # use topk to get indices of top candidates
        #     top_cand_inds = (-loss_with_doc_scores[b]).topk(1)[1]
        #     # top_cand_inds = [-1]
        #     outputs.append(generation_outputs[b, top_cand_inds])
        #     answer_proposals = generation_outputs_decoded[b*n_docs:(b+1)*n_docs]
        #     generation_outputs_for_docs.append(answer_proposals)

        #     # print(gold_columns[b])
        
        # # input()
        # outputs = torch.cat(outputs)
        # # print(self.generator_tokenizer.batch_decode(outputs))
        # # input()

        return EasyDict(outputs=outputs, 
                        retrieved_docs=retrieved_sub_tables, 
                        generator_input_sub_tables=generator_inputs.generator_input_sub_tables,
                        doc_scores=[doc_score.cpu().detach().numpy() for doc_score in doc_scores],
                        loss_with_doc_scores=loss_with_doc_scores.cpu().detach().numpy(),
                        generation_outputs_for_docs=generation_outputs_for_docs)


    def non_autoregressive_predict(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      gold_columns: List,
                      gold_rows: List,
                      question_ids: List,
                      input_text_sequences: List,
                      tables: List,
                      n_docs: int=None,
                      labels=None,
                      **kwargs):
        batch_size = input_ids.shape[0]
        
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages

        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            question_ids=question_ids, 
            input_text_sequences=input_text_sequences, 
            n_docs=n_docs,
            tables=tables,
        )

        retrieved_sub_tables=retrieval_results.retrieved_sub_tables
        retrieved_sub_table_scores=retrieval_results.retrieved_sub_table_scores
        question_hidden_states=retrieval_results.question_hidden_states

        doc_scores = retrieved_sub_table_scores

        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(
            input_text_sequences=input_text_sequences,
            retrieved_sub_tables=retrieved_sub_tables,
            labels=labels, 
            question_ids=question_ids,
            gold_columns=gold_columns,
            gold_rows=gold_rows,
            tables=tables,
            n_docs=n_docs,
            is_training=False,
        )

        test_batch = EasyDict({
            'input_ids': generator_inputs.generator_input_ids,
            'attention_mask': generator_inputs.generator_attention_mask,
            'token_type_ids': generator_inputs.generator_token_type_ids,
        })
        generation_results = self.generator(
            **test_batch
        )
        generation_outputs = generation_results.logits.detach().cpu()

        cpu_test_batch = EasyDict({
            'input_ids': generator_inputs.generator_input_ids.cpu(),
            'attention_mask': generator_inputs.generator_attention_mask.cpu(),
            'token_type_ids': generator_inputs.generator_token_type_ids.cpu(),
        })
        predicted_answer_coordinates, predicted_aggregation_indices = self.generator_tokenizer.convert_logits_to_predictions(
            cpu_test_batch, generation_results.logits.detach().cpu(), generation_results.logits_aggregation.detach().cpu()
        )
        
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

        generation_outputs_decoded = []
        all_input_sub_tables = []
        for list_item in generator_inputs.generator_input_sub_tables:
            all_input_sub_tables += list_item
        raw_cell_values = []

        for index, coordinates in enumerate(predicted_answer_coordinates):
            table = all_input_sub_tables[index]
            table = pd.DataFrame.from_records(table['rows'], columns=table['header'])
            all_cell_values = []
            if len(coordinates) == 1:
                # only a single cell:
                answer = table.iat[coordinates[0]]
                all_cell_values.append(answer)
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                all_cell_values = cell_values
            
            raw_cell_values.append(all_cell_values)

            predicted_agg = aggregation_predictions_string[index]
            if predicted_agg == "NONE":
                final_pred = all_cell_values
            else:
                if len(all_cell_values) == 0 and (predicted_agg == "SUM" or predicted_agg == "AVERAGE"):
                    print("empty sum or average!")
                    final_pred = []
                else:
                    if predicted_agg == 'COUNT':
                        final_pred = len(all_cell_values)
                    else:
                        # sum or average, must be values
                        try:
                            all_cell_values = [convert_to_float(value) for value in all_cell_values]
                            if predicted_agg == "SUM":
                                final_pred = sum(all_cell_values)
                            elif predicted_agg == 'AVERAGE':
                                final_pred = sum(all_cell_values)/len(all_cell_values)
                        
                        except ValueError:
                            # print("can not sum or average!", all_cell_values)
                            final_pred = all_cell_values
                            # Change to NONE if we can't sum or average
                            aggregation_predictions_string[index] = "NONE"
            
            generation_outputs_decoded.append(final_pred)

        # Find answer proposals from n_docs outputs for each question
        outputs = []
        generation_outputs_for_docs = []
        agg_predictions = []

        # reshape generation_outputs
        generation_outputs = generation_outputs.reshape(batch_size, n_docs, -1)
        # shifted_generation_outputs = shifted_generation_outputs.reshape(batch_size, n_docs, -1)
        try:
            for b in range(batch_size):
                # use topk to get indices of top candidates
                top_cand_inds = [-1]
                outputs.append(generation_outputs_decoded[b*n_docs:(b+1)*n_docs][top_cand_inds[0]])
                agg_predictions.append(aggregation_predictions_string[b*n_docs:(b+1)*n_docs][top_cand_inds[0]])
                answer_proposals = generation_outputs_decoded[b*n_docs:(b+1)*n_docs]
                generation_outputs_for_docs.append(answer_proposals)
                # pprint(answer_proposals)
        except Exception as e:
            print(">>>>>>", generator_inputs.generator_input_ids.shape)
            print(self.generator_tokenizer.batch_decode(generator_inputs.generator_input_ids))
            pprint(generation_outputs_decoded)
            pprint(predicted_answer_coordinates)
            pprint(predicted_aggregation_indices)
            logger.info(e)
            input()
        # pprint(raw_cell_values)
        # pprint(labels)
        # pprint(generation_outputs_decoded)
        # input()
        
        return EasyDict(outputs=outputs, 
                        agg_predictions=agg_predictions,
                        raw_cell_values=raw_cell_values,
                        retrieved_docs=retrieved_sub_tables, 
                        generator_input_sub_tables=generator_inputs.generator_input_sub_tables,
                        doc_scores=[doc_score.cpu().detach().numpy() for doc_score in doc_scores],
                        loss_with_doc_scores=None,
                        generation_outputs_for_docs=generation_outputs_for_docs)



    def get_loss(
        self, seq_logits, doc_scores, target, reduce_loss=True, epsilon=0.0, exclude_bos_score=False, ignore_index=-100, n_docs=None, retrieval_labels=None, loss_mask=None,
    ):
        """Compute loss
        Args:
            seq_logits (_type_): _description_
            doc_scores (_type_): _description_
            target (_type_): _description_
            reduce_loss (bool, optional): _description_. Defaults to True.
            epsilon (float, optional): _description_. Defaults to 0.0.
            exclude_bos_score (bool, optional): _description_. Defaults to False.
            ignore_index (int, optional): _description_. Defaults to -100.
            n_docs (_type_, optional): _description_. Defaults to None.
            retrieval_labels (_type_, optional): _description_. Defaults to None.
            loss_mask (torch.Tensor, optional): mask the loss. Default to None.
        Returns:
            EasyDict: every loss requested
        """

        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
        
        loss_dict = EasyDict()
        
        # bos_token_id is None for T5
        bos_token_id = self.generator.config.bos_token_id
        # print('bos_token_id', bos_token_id)
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()
        # print('use_bos', use_bos)
        
        batch_size = seq_logits.shape[0] // n_docs
        seq_len = seq_logits.shape[1]
        # seq_logits dim = (batch*n_docs, seq_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            batch_size, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x vocab_size
        # doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        # # print('doc_logprobs', doc_logprobs.shape)
        
        # # RAG-sequence marginalization
        # first_token_scores = seq_logprobs[:, :, :1, :]
        # if use_bos:
        #     second_token_scores = seq_logprobs[:, :, 1:2, :]
        #     remainder = seq_logprobs[:, :, 2:, :]
        #     rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)
        # else:
        #     remainder = seq_logprobs[:, :, 1:, :]
        #     rag_logprobs = torch.cat([first_token_scores + doc_logprobs, remainder], dim=2)


        # Compute NLL Loss for seq_logprobs
        new_target = target.reshape(batch_size, n_docs, -1).unsqueeze(-1)
        assert new_target.dim() == seq_logprobs.dim()

        pad_mask = new_target.eq(ignore_index)

        if pad_mask.any() and ignore_index < 0:
            # fill -100 to be 0, avoid indexing error using gather
            new_target.masked_fill_(pad_mask, 0)

        ll = seq_logprobs.gather(dim=-1, index=new_target)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
        
        ll = ll.squeeze(-1) # batch_size x n_docs x seq_len

        if loss_mask is not None:
            # loss_mask: batch_size x n_docs
            # we mask the loss accordingly
            ll.masked_fill_(loss_mask.unsqueeze(-1).repeat(1, 1, ll.shape[-1]).bool().eq(0), 0.0)
            

        nll_loss = -ll
        loss_dict.nll_loss = nll_loss

        if reduce_loss:
            mask = (pad_mask == 0)
            nll_loss = nll_loss.sum()
            nll_loss = nll_loss / torch.sum(mask)
            loss_dict.nll_loss = nll_loss

        return loss_dict
        



    @staticmethod
    def DistanceCorrelation(tensor_1, tensor_2):
        # tensor_1, tensor_2: [channel]
        # ref: https://en.wikipedia.org/wiki/Distance_correlation
        channel = tensor_1.shape[0]
        zeros = torch.zeros(channel, channel).to(tensor_1.device)
        zero = torch.zeros(1).to(tensor_1.device)
        tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
        """cul distance matrix"""
        a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
        tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
        a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
        """cul distance correlation"""
        A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
        dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
        dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
        dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
        return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)





class ITRRagAdditionRowWiseModel(ITRRagModel):
    '''
    Class for Inner Table Retrieval RAG model
    Row-wise Addition
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
            processed_sub_tables = reduce_table(table, ranked_sub_tables)

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
            
            
            # we set the minimum #col to be 2 here
            if len(processed_sub_tables) > 1:
                processed_sub_tables = processed_sub_tables[1:]

            # batch input_text_sequence with each of the retrieved sub_tables
            input_text_and_sub_tables = [(input_text_sequence, sub_table) for sub_table in processed_sub_tables]
            
            if not is_training:
                # we do a binary search to find the index of which the concatenated sub table goes overflow
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
                
                # set input_length for subtables
                for index, sub_table in enumerate(processed_sub_tables):
                    sub_table['input_length'] = token_lengths[index]
                    # this is for later evaluation
                    sub_table['original_table_overflow'] = original_table_overflow

                if overflow_only and not original_table_overflow:
                    # if the original table does not overflow
                    # and we only trim tables for overflow samples
                    # we put the whole table in the input and get predictions
                    token_lengths = [(i, token_lengths[i]) for i in range(len(processed_sub_tables))]
                    token_lengths = [token_lengths[-1]]*n_docs
                else:
                    # if the original table overflows
                    token_lengths = [(i, token_lengths[i]) for i in range(len(processed_sub_tables))]

            else:

                # in training, just make sure the last element has all gold columns
                token_lengths = [(i, 0) for i in range(len(processed_sub_tables))]
                
                # earliest_element_that_covers_all_gold = len(token_lengths)-1
                # use earliest_element_that_covers_all_gold = -1 for unsupervised case
                earliest_element_that_covers_all_gold = -1

                # pprint([subtable['sub_column_indice'] for subtable in processed_sub_tables])
                for index, subtable in enumerate(processed_sub_tables):
                    cover_all_gold = set(gold_row_list).issubset(set(subtable['sub_row_indice']))
                    if cover_all_gold:
                        earliest_element_that_covers_all_gold = index
                        break
                
                earliest_element_that_covers_all_gold = max(
                    earliest_element_that_covers_all_gold, 
                    self.config.data_loader.additional.num_knowledge_passages-1
                    )

                earliest_element_that_covers_all_gold = min(
                    earliest_element_that_covers_all_gold,
                    len(token_lengths)-1
                )

                token_lengths = [token_lengths[earliest_element_that_covers_all_gold]]*n_docs
                

            if n_docs > len(token_lengths):
                # repeat the last item
                token_lengths_reduced = token_lengths + [token_lengths[-1]] * (n_docs - len(processed_sub_tables))
            else:
                # take the first n_docs sub tables
                token_lengths_reduced = token_lengths[:n_docs]

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
