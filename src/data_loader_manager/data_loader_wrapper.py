# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import sys
import time
import json
import copy
import numpy as np
import json
import torch

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict

import logging
logger = logging.getLogger(__name__)

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from transformers import ViTFeatureExtractor
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from transformers import TapasTokenizer
# For TAPEX model
from transformers import TapexTokenizer, BartConfig, BartForConditionalGeneration
from models.tapas.custom_tokenizer import CustomTapasTokenizer
# Fro ColBERT model
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.infra import Run, RunConfig, ColBERTConfig

class DataLoaderWrapper():
    '''
    Data loader wrapper, general class definitions
    '''

    def __init__(self, config):
        self.config = config
        if self.config.mode in ['create_data']:
            # do not load tokenizers at creating data
            return

        if self.config.model_config.base_model == 'ColBERT':
            logger.info("initializing ColBERT tokenizers...")
            
            config = ColBERTConfig(
                bsize=None,
            )
            config.checkpoint = 'bert-base-uncased'
            
            self.tokenizer = QueryTokenizer(config)
            self.decoder_tokenizer = DocTokenizer(config)

            self.SPECIAL_TOKENS = self.config.model_config.SPECIAL_TOKENS
            self.SPECIAL_TOKENS['additional_special_tokens'] = self.tokenizer.tok.additional_special_tokens + self.SPECIAL_TOKENS['additional_special_tokens']
            
            self.tokenizer.tok.add_special_tokens(self.SPECIAL_TOKENS)
            self.DECODER_SPECIAL_TOKENS = self.config.model_config.DECODER_SPECIAL_TOKENS
            self.DECODER_SPECIAL_TOKENS['additional_special_tokens'] = self.decoder_tokenizer.tok.additional_special_tokens + self.DECODER_SPECIAL_TOKENS['additional_special_tokens']
            self.decoder_tokenizer.tok.add_special_tokens(self.DECODER_SPECIAL_TOKENS)

        elif self.config.model_config.get("QueryEncoderModelClass", "") == 'ColBERT':
            # In RAG with ColBERT, only init query tokenizer with ColBERT tokenizer
            logger.info("initializing ColBERT tokenizers...")
            
            config = ColBERTConfig(
                bsize=None,
            )
            config.checkpoint = 'bert-base-uncased'
            
            self.tokenizer = QueryTokenizer(config)
            
            self.SPECIAL_TOKENS = self.config.model_config.SPECIAL_TOKENS
            self.SPECIAL_TOKENS['additional_special_tokens'] = self.tokenizer.tok.additional_special_tokens + self.SPECIAL_TOKENS['additional_special_tokens']
            
            self.tokenizer.tok.add_special_tokens(self.SPECIAL_TOKENS)
            
            # Then init decoder tokenizer
            DecoderTokenizerClass = globals()[self.config.model_config.DecoderTokenizerClass]
            self.decoder_tokenizer = DecoderTokenizerClass.from_pretrained(self.config.model_config.DecoderTokenizerModelVersion)
            self.DECODER_SPECIAL_TOKENS = self.config.model_config.DECODER_SPECIAL_TOKENS
            self.DECODER_SPECIAL_TOKENS['additional_special_tokens'] = self.decoder_tokenizer.additional_special_tokens + self.DECODER_SPECIAL_TOKENS['additional_special_tokens']
            self.decoder_tokenizer.add_special_tokens(self.DECODER_SPECIAL_TOKENS)

        else:
            # Prepare for tokenizers
            TokenizerClass = globals()[self.config.model_config.TokenizerClass]
            self.tokenizer = TokenizerClass.from_pretrained(self.config.model_config.TokenizerModelVersion)
            self.SPECIAL_TOKENS = self.config.model_config.SPECIAL_TOKENS
            self.SPECIAL_TOKENS['additional_special_tokens'] = self.tokenizer.additional_special_tokens + self.SPECIAL_TOKENS['additional_special_tokens']
            
            self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
            
            # Load second tokenizer if specified
            if self.config.model_config.get('DecoderTokenizerClass', None) is not None:
                DecoderTokenizerClass = globals()[self.config.model_config.DecoderTokenizerClass]
                self.decoder_tokenizer = DecoderTokenizerClass.from_pretrained(self.config.model_config.DecoderTokenizerModelVersion)
                self.DECODER_SPECIAL_TOKENS = self.config.model_config.DECODER_SPECIAL_TOKENS
                self.DECODER_SPECIAL_TOKENS['additional_special_tokens'] = self.decoder_tokenizer.additional_special_tokens + self.DECODER_SPECIAL_TOKENS['additional_special_tokens']
                self.decoder_tokenizer.add_special_tokens(self.DECODER_SPECIAL_TOKENS)
                
            else:
                self.decoder_tokenizer = self.tokenizer

        # Load feature extractor if specified
        if self.config.model_config.get('FeatureExtractorClass', None) is not None:
            # Load image feature extractor
            FeatureExtractorClass = globals()[self.config.model_config.FeatureExtractorClass]
            self.feature_extractor = FeatureExtractorClass.from_pretrained(self.config.model_config.FeatureExtractorModelVersion)
        else:
            self.feature_extractor = None

        self.data_loaders = EasyDict({
            'train': {},
            'valid': {},
            'test': {},
        })
        # Prepare data for training
        # self.build_dataset()

    def set_io(self, io):
        self.io = io

    def build_dataset(self):
        """
        This function loads data and features required for building the dataset
        """

        self.data = EasyDict()

        dataset_modules = self.config.data_loader.dataset_modules.module_list
        for dataset_module in dataset_modules:
            module_config = self.config.data_loader.dataset_modules.module_dict[dataset_module]
            logger.info('Loading dataset module: {}'.format(module_config))
            loading_func = getattr(self, dataset_module)
            loading_func(module_config)
            print('data columns: {}'.format(self.data.keys()))
        
