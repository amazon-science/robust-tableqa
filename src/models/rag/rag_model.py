# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import copy
import math
import os
import warnings

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
# For ColBERT model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.data import Queries
from colbert import Searcher

from transformers.models.rag.retrieval_rag import Index #CustomHFIndex, CanonicalHFIndex
import pytorch_lightning as pl

import time
from pprint import pprint
import pandas as pd

import logging
logger = logging.getLogger(__name__)

import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk
import faiss
import pickle
from typing import Iterable, List, Optional, Tuple

class HFIndexBase(Index):
    def __init__(self, vector_size, dataset, index_initialized=False):
        self.vector_size = vector_size
        self.dataset = dataset
        self._index_initialized = index_initialized
        self._check_dataset_format(with_index=index_initialized)
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")

    def _check_dataset_format(self, with_index: bool):
        if not isinstance(self.dataset, Dataset):
            raise ValueError(f"Dataset should be a datasets.Dataset object, but got {type(self.dataset)}")
        # if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:
        #     raise ValueError(
        #         "Dataset should be a dataset with the following columns: "
        #         "title (str), text (str) and embeddings (arrays of dimension vector_size), "
        #         f"but got columns {self.dataset.column_names}"
        #     )
        if with_index and "embeddings" not in self.dataset.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )

    def init_index(self):
        raise NotImplementedError()

    def is_initialized(self):
        return self._index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors)  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)



class CustomHFIndex(HFIndexBase):
    """
    A wrapper around an instance of [`~datasets.Datasets`]. The dataset and the index are both loaded from the
    indicated paths on disk.
    Args:
        vector_size (`int`): the dimension of the passages embeddings used by the index
        dataset_path (`str`):
            The path to the serialized dataset on disk. The dataset should have 3 columns: title (str), text (str) and
            embeddings (arrays of dimension vector_size)
        index_path (`str`)
            The path to the serialized faiss index on disk.
    """

    def __init__(self, vector_size: int, dataset, index_path=None):
        super().__init__(vector_size, dataset, index_initialized=index_path is None)
        self.index_path = index_path

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        logger.info(f"Loading passages from {dataset_path}")
        if dataset_path is None or index_path is None:
            raise ValueError(
                "Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` "
                "and `dataset.get_index('embeddings').save(index_path)`."
            )
        dataset = load_from_disk(dataset_path)
        return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)

    def init_index(self):
        if not self.is_initialized():
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self._index_initialized = True



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


class RagModel(pl.LightningModule):
    '''
    Class for RAG, taken from RAVQA
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        super().__init__()

        self.config = config
        self.data_loader = data_loader
        self.retriever_tokenizer = data_loader.tokenizer
        self.generator_tokenizer = data_loader.decoder_tokenizer

        
        # Initialising question encoder
        QueryEncoderModelClass = globals()[self.config.model_config.QueryEncoderModelClass]
        
        self.use_colbert = True if self.config.model_config.QueryEncoderModelClass == "ColBERT" else False

        if self.use_colbert:

            if "$" in self.config.model_config.QueryEncoderModelVersion:
                self.config.model_config.QueryEncoderModelVersion = os.path.join(self.config.EXPERIMENT_FOLDER, self.config.model_config.QueryEncoderModelVersion.replace('$', ''))
            
            colbert_config = ColBERTConfig(
                bsize=None,
                use_ib_negatives=True,
                checkpoint=self.config.model_config.QueryEncoderModelVersion,
                rank=self.global_rank,
            )
            colbert_config.checkpoint = colbert_config.checkpoint
            
            self.question_encoder = ColBERT(name=colbert_config.checkpoint, colbert_config=colbert_config)
            # self.question_encoder.raw_tokenizer = self.retriever_tokenizer

            # Resize the bert embedding space to accommodate special tokens
            # logger.info(f'tokenizer lengths = {len(self.tokenizer.tok)} and {len(self.decoder_tokenizer.tok)}')
            # self.model.bert.resize_token_embeddings(
            #     max(len(self.tokenizer.tok), len(self.decoder_tokenizer.tok))
            # )

        else:
            QueryEncoderConfigClass = globals()[self.config.model_config.QueryEncoderConfigClass]
            
            if "$" in self.config.model_config.QueryEncoderModelVersion:
                self.config.model_config.QueryEncoderModelVersion = os.path.join(self.config.EXPERIMENT_FOLDER, self.config.model_config.QueryEncoderModelVersion.replace('$', ''))
            
            
            question_encoder_model_config = QueryEncoderConfigClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion)
            self.question_encoder = QueryEncoderModelClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion,
                                                        config=question_encoder_model_config)
            self.retiever_hidden_size = question_encoder_model_config.hidden_size
            self.question_encoder.resize_token_embeddings(len(self.retriever_tokenizer))

        
        # Initialising generator
        GeneratorModelClass = globals()[self.config.model_config.GeneratorModelClass]
        GeneratorConfigClass = globals()[self.config.model_config.GeneratorConfigClass]
        generator_model_config = GeneratorConfigClass.from_pretrained(self.config.model_config.GeneratorModelVersion)
        self.generator = GeneratorModelClass.from_pretrained(self.config.model_config.GeneratorModelVersion,
                                                    config=generator_model_config)
        
        
        self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        # label smoother imported from huggingface transformers
        label_smoothing_factor = self.config.train.additional.get('label_smoothing_factor', 0)
        if label_smoothing_factor != 0:
            from transformers.trainer_pt_utils import LabelSmoother
            self.label_smoother = LabelSmoother(epsilon=label_smoothing_factor)
        else:
            self.label_smoother = None
        
        self.init_retrieval()

        self.retrieve = self.main_retrieve


    
    def init_retrieval(self):

        # Prepend EXPERIMENT_FOLDER to all paths
        for k, v in self.config.model_config.index_files.items():
            self.config.model_config.index_files[k] = os.path.join(self.config.EXPERIMENT_FOLDER, v)

        if self.use_colbert:
            # Use ColBERT index

            index_path = self.config.model_config.index_files.index_path
            index_root = os.path.dirname(index_path)
            index_name = os.path.basename(index_path)

            with Run().context(RunConfig(nranks=1, rank=self.global_rank, root=index_root, experiment=index_name)):
                
                config = ColBERTConfig(
                    total_visible_gpus=0,
                )
                self.index = Searcher(index="temp_index.nbits=2", config=config)

            # Load embedding and table dataset
            logger.info(f"Loading table data from {self.config.model_config.index_files.index_passages_path}")
            self.table_dataset = load_from_disk(
                self.config.model_config.index_files.index_passages_path
            )
            self.table_dict = self.table_dataset.add_column("table_index", list(range(len(self.table_dataset)))).to_pandas().set_index("table_id", drop=False).to_dict(orient="index")
            
            self.table_dict_by_index = self.table_dataset.add_column("table_index", list(range(len(self.table_dataset)))).to_pandas().to_dict(orient="index")

            logger.info(f"Loading embedding data from {self.config.model_config.index_files.embedding_path}")
            with open(self.config.model_config.index_files.embedding_path, 'rb') as f:
                embedding_data = pickle.load(f)
                self.item_embeddings = {}
                # self.item_embedding_mask = {}
                for index, item_embeddings, item_embedding_mask in tqdm(zip(list(range(len(embedding_data['item_embeddings']))), embedding_data['item_embeddings'], embedding_data['item_embedding_mask'])):
                    self.item_embeddings[index] = (item_embeddings, item_embedding_mask)
                
            return
        
        if self.config.model_config.index_files.index_passages_path == '':
            # use wikidata
            self.index = CanonicalHFIndex(
                vector_size=self.retiever_hidden_size,
                dataset_name=self.config.model_config.index_files.index_dataset,
                dataset_split=self.config.model_config.index_files.index_dataset_split,
                index_name=self.config.model_config.index_files.index_name,
                index_path=None,
                use_dummy_dataset=True if self.config.model_config.index_files.index_dummy else False,
            )
            self.data_source = 'wiki'
        else:
            # use custom corpus
            self.index = CustomHFIndex.load_from_disk(
                vector_size=self.retiever_hidden_size,
                dataset_path=self.config.model_config.index_files.index_passages_path,
                index_path=self.config.model_config.index_files.index_path,
            )
            self.data_source = 'custom'
        print("initializing retrieval")
        self.index.init_index()
        self.table_dict = self.index.dataset.to_pandas().set_index("table_id", drop=False).to_dict(orient="index")
        print("init done.")



    def main_retrieve(self, 
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, 
                    labels: torch.Tensor, 
                    question_ids: List, 
                    input_text_sequences: List, 
                    n_docs=None,
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
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages

        batch_size = input_ids.shape[0]

        # Generate query embeddings and obtain item embeddings from index
        if self.use_colbert:
            Q = (input_ids, attention_mask)
            question_hidden_states = self.question_encoder.query(*Q)
            
            custom_quries = {i: query for i, query in enumerate(input_text_sequences)}
            queries = Queries(data=custom_quries)
            
            ranking = self.index._search_all_Q(queries, question_hidden_states.cpu().detach(), k=n_docs, progress=False)
            
            # pprint(ranking.todict())
            retrieval_results = ranking.todict()

            doc_scores = []
            all_retrieved_doc_indices = []


            for query_index, retrieved_docs in retrieval_results.items():
                retrieved_doc_indices = []
                retrieved_doc_scores = []
                for doc_index, _, doc_score in retrieved_docs:
                    retrieved_doc_indices.append(doc_index)
                    retrieved_doc_scores.append(doc_score)
                
                # Get embeddings
                retrieved_item_embeddings = []
                retrieved_item_embeding_mask = []
                for i in retrieved_doc_indices:
                    emb_tuple = self.item_embeddings[i]
                    retrieved_item_embeddings.append(torch.Tensor(emb_tuple[0]))
                    retrieved_item_embeding_mask.append(torch.Tensor(emb_tuple[1]))
                
                retrieved_item_embeddings = torch.stack(retrieved_item_embeddings).to(self.device)
                retrieved_item_embeding_mask = torch.stack(retrieved_item_embeding_mask).to(self.device)
                
                retrieved_query_embedding = question_hidden_states[[query_index]]

                self.question_encoder.colbert_config.nway = len(retrieved_doc_indices)
                Q_duplicated = retrieved_query_embedding.repeat_interleave(self.question_encoder.colbert_config.nway, dim=0).contiguous()
                scores = self.question_encoder.score(Q_duplicated, retrieved_item_embeddings, retrieved_item_embeding_mask)

                doc_scores.append(scores)
                all_retrieved_doc_indices.append(retrieved_doc_indices)
            
            # batch_size x n_docs
            doc_scores = torch.stack(doc_scores)
            ids = np.array(all_retrieved_doc_indices)
        else:
            # Use question_encoder to encode question inputs
            query_outputs = self.question_encoder(input_ids=input_ids,
                                                attention_mask=attention_mask)
            question_hidden_states = query_outputs.pooler_output
            # print('question_hidden_states', question_hidden_states.shape)

            # start_time = time.time()
            ids, vectors = self.index.get_top_docs(question_hidden_states.cpu().detach().numpy(), n_docs)
            # print(
            #     f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
            # )
            # print(ids)

            # question_hidden_states: batch_size x hidden_size
            # item_hidden_states: batch_size x n_docs x hidden_size
            item_hidden_states = torch.Tensor(vectors).type_as(question_hidden_states)

            # print('item_hidden_states', item_hidden_states.shape)

            # batch_size x n_docs
            doc_scores = (question_hidden_states.unsqueeze(dim=1) * item_hidden_states).sum(dim=-1)
        
        
        pos_item_ids = kwargs.get('pos_item_ids', None)

        modified_doc_scores = doc_scores.clone()

        dont_add_reference_table = 'dont_add_reference_table' in self.config.model_config.modules

        retrieved_docs = []
        for b in range(batch_size):
            doc_data = []
            if self.use_colbert:
                retrieved_doc_indices = all_retrieved_doc_indices[b]
                # print('retrieved_doc_indices', retrieved_doc_indices)
                # contents = self.table_dataset.select(retrieved_doc_indices).to_pandas().to_dict(orient='records') # change to list of dicts
                contents = [self.table_dict_by_index[i] for i in retrieved_doc_indices]
            else:
                contents = self.index.get_doc_dicts(ids[b])
            # print('len(contents)', len(contents), len(ids[b]))

            retrieved_table_ids = [contents[i]['table_id'] for i in range(n_docs)]
            
            if pos_item_ids is not None and not dont_add_reference_table:
                # if the reference table is not here, add it!
                pos_item_id = pos_item_ids[b]

                has_reference_table = False
                for retrieved_table_id in retrieved_table_ids:
                    if retrieved_table_id in pos_item_ids[b]:
                        has_reference_table = True
                
                if not has_reference_table:
                    pos_item = self.table_dict[pos_item_id]
                    if self.use_colbert:
                        # Need to retrieve embeddings first!
                        table_index = pos_item['table_index']
                        emb_tuple = self.item_embeddings[table_index]
                        pos_item_embeddings = torch.Tensor(emb_tuple[0]).unsqueeze(0)
                        pos_item_embedding_mask = torch.Tensor(emb_tuple[1]).unsqueeze(0)
                        
                    new_contents = [pos_item]
                    new_contents += contents[:n_docs-1]
                    contents = new_contents
                    # print('before', modified_doc_scores)
                    # print(question_hidden_states[b])
                    # print(pos_item['embeddings'])
                    modified_doc_scores[b, 1:] = doc_scores[b, :-1]
                    if self.use_colbert:
                        self.question_encoder.colbert_config.nway = 1
                        # print("question_hidden_states[[b]]", question_hidden_states[[b]].shape)
                        # print('pos_item_embeddings', pos_item_embeddings.shape)
                        # print('pos_item_embedding_mask', pos_item_embedding_mask.shape)
                        score = self.question_encoder.score(question_hidden_states[[b]], pos_item_embeddings, pos_item_embedding_mask)
                        # print('score', score)
                        modified_doc_scores[b, 0] = score[0]
                    else:
                        modified_doc_scores[b, 0] = torch.matmul(question_hidden_states[b], torch.Tensor(pos_item['embeddings']).to(self.device))
                    # print('after', modified_doc_scores)
                    # input()
            # print(contents)
            # print(len(contents), n_docs)
            # input()
            for i in range(n_docs):
                passage_data = {
                    'passage_id': str(ids[b, i]),
                    'table_id': contents[i]['table_id'],
                    'header': contents[i]['header'],
                    'rows': contents[i]['rows'],
                    'table_contents': contents[i]['table_contents'],
                    'table_title': contents[i]['documentTitle'],
                    'score': modified_doc_scores[b, i].cpu().detach().numpy(),
                }
                # print(passage_data)
                # print(self.data_loader.data.passages.id2doc[str(ids[b, i])])
                # input()
                doc_data.append(passage_data)
            retrieved_docs.append(doc_data)
        
        assert len(retrieved_docs) == batch_size
        
        return EasyDict(
            retrieved_docs=retrieved_docs,
            doc_scores=modified_doc_scores,
            question_hidden_states=question_hidden_states,
        )


    def prepare_inputs_for_generator(self, 
                input_text_sequences, retrieved_docs, labels, n_docs=None):
        
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
        
        batch_size = len(input_text_sequences)

        extended_input_text_sequences = []

        for index, input_text_sequence in enumerate(input_text_sequences):
            scores = []
            for doc in retrieved_docs[index]:
                extended_input_text_sequences.append(
                    (input_text_sequence + " title: " + doc['table_title'] + " ", doc)
                )
                scores.append(doc['score'])

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
        generator_input_ids, generator_attention_mask = encoding.input_ids, encoding.attention_mask
        generator_input_ids = generator_input_ids.to(labels.device)
        generator_attention_mask = generator_attention_mask.to(labels.device)
        if targets is not None:
            generator_decoder_input_ids = shift_tokens_right(
                targets, self.generator.config.pad_token_id, self.generator.config.decoder_start_token_id
            )
        else:
            generator_decoder_input_ids = None

        return EasyDict(
            generator_input_text_sequences=extended_input_text_sequences,
            generator_input_ids=generator_input_ids,
            generator_attention_mask=generator_attention_mask,
            generator_decoder_input_ids=generator_decoder_input_ids,
            generator_labels=targets,
        )

    def forward(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      question_ids: List,
                      input_text_sequences: List,
                    **kwargs):
        
        batch_size = input_ids.shape[0]
        n_docs = self.config.data_loader.additional.num_knowledge_passages

        pos_item_ids = kwargs.get('pos_item_ids', None)

        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(input_ids, attention_mask, labels, question_ids, input_text_sequences, n_docs=n_docs, pos_item_ids=pos_item_ids)
        retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        
        answers = kwargs.get('answers', None)
        
        assert answers is not None
        assert pos_item_ids is not None
        get_retrieval_labels_results = self.get_retrieval_labels(
            question_ids=question_ids,
            batch_answers=answers,
            batch_pos_item_ids=pos_item_ids,
            batch_retrieved_docs=retrieved_docs,
        )
        retrieval_labels = get_retrieval_labels_results.retrieval_labels
        
        labels = labels.repeat_interleave(n_docs, 0)

        #########################
        if "add_binary_labels_as_prompt" in self.config.model_config.modules:
            new_labels = []
            for batch_index in range(batch_size):
                for label_index in range(n_docs):
                    if retrieval_labels[batch_index, label_index] == 1:
                        answer = "yes. " + ", ".join(answers[batch_index])
                    else:
                        answer = "no. " + ", ".join(answers[batch_index])
                    new_labels.append(answer)
            
            with self.generator_tokenizer.as_target_tokenizer():
                target_encoding = self.generator_tokenizer(
                                        answer=new_labels,
                                        padding='longest',
                                        max_length=self.config.data_loader.additional.max_target_length,
                                        truncation=True, 
                                        return_tensors='pt')
            labels = target_encoding.input_ids
            # replace padding token id's of the labels by -100
            labels = [
                [(label if label != self.generator_tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels
            ]

            labels = torch.LongTensor(labels).to(self.device)
        #########################

        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(   
                                            input_text_sequences=input_text_sequences,
                                            retrieved_docs=retrieved_docs,
                                            labels=labels, 
                                            n_docs=n_docs)
        
        if "rag_training" in self.config.model_config.modules:
            # labels = [0, a, b, c, 2]
            # decoder_input_ids = [2, 0, a, b, c]

            # seq_logits is derived from the decoder input ids [2, 0, a, b, c]
            # so it is expected to be [0, a, b, c, 2]
            # target = [0, a, b, c, 2]
            # thus we don't do shift left here
            # print('decoder_input_ids', generator_inputs.generator_decoder_input_ids)
            generator_outputs = self.generator(
                            input_ids=generator_inputs.generator_input_ids,
                            attention_mask=generator_inputs.generator_attention_mask,
                            decoder_input_ids=generator_inputs.generator_decoder_input_ids,
                            return_dict=True)
        else:
            generator_outputs = self.generator(
                            input_ids=generator_inputs.generator_input_ids,
                            attention_mask=generator_inputs.generator_attention_mask,
                            decoder_input_ids=generator_inputs.generator_decoder_input_ids,
                            labels=labels,
                            return_dict=True)
        
        logits = generator_outputs.logits

        # if self.label_smoother is not None:
        #     smoothed_loss = self.label_smoother(generator_outputs, labels)
        # else:
        #     smoothed_loss = self.loss_fct(
        #         logits.view(-1, self.generator.config.vocab_size), 
        #         labels.view(-1)
        #     )
        
        # loss_dict = EasyDict(
        #     nll_loss=smoothed_loss,
        # )

        label_smoothing_factor = self.config.train.additional.get('label_smoothing_factor', 0)

        if "use_pseudo_in_ravqa_loss" in self.config.model_config.modules:
            get_retrieval_labels_results = self.get_retrieval_labels(
                question_ids=question_ids,
                batch_answers=answers,
                batch_pos_item_ids=pos_item_ids,
                batch_retrieved_docs=retrieved_docs,
                pseudo=True,
            )
            retrieval_labels = get_retrieval_labels_results.retrieval_labels

        loss_dict = self.get_loss(
            seq_logits=logits,
            doc_scores=doc_scores,
            target=generator_inputs.generator_labels,
            exclude_bos_score=True,
            epsilon=label_smoothing_factor,
            n_docs=n_docs,
            retrieval_labels=retrieval_labels,
        )
        # print(loss_dict)
        # input()

        # aggregate loss
        total_loss = 0
        for loss_name, loss_ratio in self.config.model_config.loss_ratio.items():
            if loss_ratio != 0:
                total_loss += loss_dict[loss_name] * loss_ratio
        
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
        
        return EasyDict(loss=total_loss,
                        loss_dict=loss_dict,
                        logits=logits,
                        doc_scores=doc_scores.cpu().detach().numpy())


    def generate(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      question_ids: List,
                      input_text_sequences: List,
                      n_docs: int=None,
                      **kwargs):

        batch_size = input_ids.shape[0]
        pos_item_ids = kwargs.get('pos_item_ids', None)
        answers = kwargs.get('answers', None)

        # whether to include gold document in the retrieval if the gold document is not retrieved
        # only used in oracle testing!
        oracle_in_test = "oracle_in_test" in self.config.model_config.modules
        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(input_ids, attention_mask, labels, question_ids, input_text_sequences, pos_item_ids=pos_item_ids if oracle_in_test else None)
        retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages


        get_retrieval_labels_results = self.get_retrieval_labels(
            question_ids=question_ids,
            batch_answers=answers,
            batch_pos_item_ids=pos_item_ids,
            batch_retrieved_docs=retrieved_docs,
        )
        retrieval_labels = get_retrieval_labels_results.retrieval_labels
        

        # populate labels
        labels = labels.repeat_interleave(n_docs, 0)

        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(
                                            input_text_sequences=input_text_sequences,
                                            retrieved_docs=retrieved_docs,
                                            labels=labels,
                                            n_docs=n_docs)
        
        if "rag_decoding" in self.config.model_config.modules:
            n_beams = self.config.model_config.num_beams
            outputs = []
            generation_outputs_for_docs = []
            loss_with_doc_scores = []
            # RAG decoding. For each input in the batch, run generation with beams, and re-forward the sequence through to obtain loss
            for index in range(batch_size):
                # first, generate beams from documents:
                generator_input_ids = generator_inputs.generator_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)

                # print('generator_input_ids', self.generator_tokenizer.batch_decode(generator_input_ids))
                
                test_batch = EasyDict({
                    'input_ids': generator_input_ids,
                    'attention_mask': generator_inputs.generator_attention_mask[index * n_docs : (index + 1) * n_docs],
                    "max_length": self.config.data_loader.additional.max_target_length,
                    "num_beams": n_beams,
                    "num_return_sequences": n_beams,
                })
                output_sequences = self.generator.generate(
                    **test_batch
                )  # n_docs * n_beam, tgt_len

                # print('output_sequences', self.generator_tokenizer.batch_decode(output_sequences))
                
                num_candidates = output_sequences.shape[
                    0
                ]

                individual_input_ids = generator_inputs.generator_input_ids[index * n_docs : (index + 1) * n_docs].repeat(
                    num_candidates, 1
                )  # (num_candidates*n_docs, max_len)
                
                individual_attention_mask = generator_inputs.generator_attention_mask[index * n_docs : (index + 1) * n_docs]
                individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)
                

                # output_sequences = [2, 0, a, b, c, 2]
                # decoder input ids = [2, 0, a, b, c, 2]
                # labels=output_sequences = [2, 0, a, b, c, 2] --> shift left --> [0, a, b, c, 2, 1]
                loss_target = output_sequences.repeat_interleave(n_docs, dim=0)
                loss_target = torch.cat(
                    [loss_target[:, 1:], loss_target.new(loss_target.shape[0], 1).fill_(self.generator.config.pad_token_id)], 1
                )
                decoder_input_ids = output_sequences.repeat_interleave(n_docs, dim=0)
                # print('individual_input_ids', self.generator_tokenizer.batch_decode(individual_input_ids))
                # print('decoder_input_ids', decoder_input_ids)

                # Re-forward the generator to obtain loss
                reforward_outputs = self.generator(
                                    input_ids=individual_input_ids,
                                    attention_mask=individual_attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    return_dict=True)
                
                logits = reforward_outputs.logits

                individual_doc_scores = doc_scores[index : (index + 1), :]  # doc_scores.shape = [batch, n_docs]
                individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)  # [num_candidates, n_docs]
                
                label_smoothing_factor = self.config.train.additional.get('label_smoothing_factor', 0)
                loss_dict = self.get_loss(
                    seq_logits=logits,
                    doc_scores=individual_doc_scores,
                    target=loss_target,
                    exclude_bos_score=True,
                    epsilon=label_smoothing_factor,
                    n_docs=n_docs,
                    retrieval_labels=None,
                    reduce_loss=False,
                    ignore_index=self.generator.config.pad_token_id,
                )
                rag_loss = loss_dict.rag_loss

                top_cand_inds = (-rag_loss).topk(1)[1]

                # add hypothesis
                outputs.append(output_sequences[top_cand_inds])

                generation_outputs_decoded = self.generator_tokenizer.batch_decode(output_sequences, skip_special_tokens=True) # n_docs * n_beams
                # print('selected:', generation_outputs_decoded[top_cand_inds[0]])
                # print('generation_outputs_decoded', generation_outputs_decoded)
                answer_proposals = []
                doc_loss = []
                for doc_index in range(n_docs):
                    # for each retrieved doc, find the beam with highest confidence
                    loss_of_beams = -rag_loss[doc_index*n_beams: (doc_index+1)*n_beams]
                    # print('loss_of_beams', loss_of_beams)
                    top_index = (loss_of_beams).topk(1)[1][0]
                    answer_proposals.append(
                        generation_outputs_decoded[doc_index*n_beams: (doc_index+1)*n_beams][top_index]
                    )
                    doc_loss.append(
                        loss_of_beams[top_index]
                    )
                doc_loss = torch.FloatTensor(doc_loss)
                # print('doc_loss', doc_loss)

                loss_with_doc_scores.append(
                    doc_loss
                )
                # print('answer_proposals', answer_proposals)
                # input()
                generation_outputs_for_docs.append(answer_proposals)
            
            loss_with_doc_scores = torch.stack(loss_with_doc_scores)

            outputs = self._cat_and_pad(outputs, pad_token_id=self.generator.config.pad_token_id)
            
        else:
            # Not RAG decoding. Simply run generation and return the highest confident answer
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
            # decode the generation outputs
            generation_outputs_decoded = self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)


            outputs = []
            generation_outputs_for_docs = []

            # reshape generation_outputs
            generation_outputs = generation_outputs.reshape(batch_size, n_docs, -1)
            # shifted_generation_outputs = shifted_generation_outputs.reshape(batch_size, n_docs, -1)
            
            # loss_with_doc_scores = -sentence_confidence 
            
            loss_with_doc_scores = generation_seq_scores.reshape(batch_size, n_docs)

            # # Add document scores to the confidence
            # doc_scores_softmax = F.log_softmax(doc_scores, dim=-1)
            # loss_with_doc_scores += doc_scores_softmax
            
            
            if "add_binary_labels_as_prompt" in self.config.model_config.modules:
                for b in range(batch_size):
                    loss_mask = ['no. ' in generated_str for generated_str in generation_outputs_decoded[b*n_docs:(b+1)*n_docs]]
                    for i in range(n_docs):
                        if loss_mask[i]:
                            loss_with_doc_scores[b, i] = loss_with_doc_scores[b, i] - 100.0
            

            final_decisions = []
            for b in range(batch_size):
                # use topk to get indices of top candidates
                top_cand_inds = (loss_with_doc_scores[b]).topk(1)[1]
                final_decisions.append(top_cand_inds)
                outputs.append(generation_outputs[b, top_cand_inds])
                answer_proposals = generation_outputs_decoded[b*n_docs:(b+1)*n_docs]
                generation_outputs_for_docs.append(answer_proposals)

            outputs = torch.cat(outputs)

        
            # for index, batch_docs in enumerate(retrieved_docs):
            #     print(input_text_sequences[index])
            #     for doc in batch_docs:
            #         print("------- doc -------")
            #         print(doc['header'])
            #         pprint(doc['rows'])
            #         print("------- doc end -------")
            #     print('retrieval_labels', retrieval_labels[index])
            #     print("answers", answers[index])
            #     print('doc predictions:', generation_outputs_for_docs[index])
            #     print('confidence', loss_with_doc_scores[index])
            #     # print('sentence_confidence', sentence_confidence[index])
            #     print('final decision:', final_decisions[index])
            #     input()

        return EasyDict(outputs=outputs, 
                        retrieved_docs=retrieved_docs, 
                        doc_scores=doc_scores.cpu().detach().numpy(),
                        loss_with_doc_scores=loss_with_doc_scores.cpu().detach().numpy(),
                        generation_outputs_for_docs=generation_outputs_for_docs)

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        output = (
            tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
        )
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]] = t
            ind += t.shape[0]
        return output
    
    def get_loss(
        self, seq_logits, doc_scores, target, reduce_loss=True, epsilon=0.0, exclude_bos_score=False, ignore_index=-100, n_docs=None, retrieval_labels=None,
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
        Returns:
            EasyDict: every loss requested
        """

        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
        
        loss_dict = EasyDict()
        
        # bos_token_id is None for T5
        bos_token_id = self.generator.config.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        
        batch_size = seq_logits.shape[0] // n_docs
        seq_len = seq_logits.shape[1]
        # seq_logits dim = (batch*n_docs, seq_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            batch_size, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('doc_logprobs', doc_logprobs.shape)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        if use_bos:
            second_token_scores = seq_logprobs[:, :, 1:2, :]
            remainder = seq_logprobs[:, :, 2:, :]
            rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)
        else:
            remainder = seq_logprobs[:, :, 1:, :]
            rag_logprobs = torch.cat([first_token_scores + doc_logprobs, remainder], dim=2)


        # Compute NLL Loss for seq_logprobs
        new_target = target.reshape(batch_size, n_docs, -1).unsqueeze(-1)
        assert new_target.dim() == seq_logprobs.dim()

        pad_mask = new_target.eq(ignore_index)
        if pad_mask.any():
            # fill -100 to be 0, avoid indexing error using gather
            new_target.masked_fill_(pad_mask, 0)
        

        ll = seq_logprobs.gather(dim=-1, index=new_target)
        smooth_obj = seq_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
            smooth_obj.masked_fill_(pad_mask, 0.0)
        
        
        ll = ll.squeeze(-1) # batch_size x n_docs x seq_len
        smooth_obj = smooth_obj.squeeze(-1)
        
        nll_loss = -ll
        smooth_loss = -smooth_obj

        eps_i = epsilon / seq_logprobs.size(-1)
        smoothed_nll_loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

        if 'mask_non_reference_tables' in self.config.model_config.modules:
            smoothed_nll_loss = smoothed_nll_loss * retrieval_labels.unsqueeze(-1).to(self.device)

        loss_dict.nll_loss = smoothed_nll_loss


        if self.config.model_config.loss_ratio.rag_loss != 0:
            rag_loss = self.get_rag_nll(
                seq_logits, 
                doc_scores, 
                target,
                reduce_loss=reduce_loss,
                epsilon=epsilon,
                exclude_bos_score=exclude_bos_score, 
                ignore_index=ignore_index, 
                n_docs=n_docs,
            )
            loss_dict.rag_loss = rag_loss

        if self.config.model_config.loss_ratio.additional_loss != 0:
            if retrieval_labels is not None:
                target_index = 1
                if "add_binary_labels_as_prompt" in self.config.model_config.modules:
                    target_index = 3

                # This is a practical approximation to matching the current model predictions
                # A generative model will normally generate the correct answer if the first token is generated correctly. This approximation worked well so that we can save some computation here.
                
                first_token_scores = seq_logprobs[:, :, target_index:target_index+1, :]
                first_token_scores = first_token_scores.detach()

                # batch_size x n_docs x voc_size
                first_token_scores = first_token_scores.squeeze(2)
                # batch_size x n_docs
                first_token_prediction = torch.argmax(first_token_scores, dim=-1)
                # print('first_token_prediction', first_token_prediction)

                # batch_size x n_docs
                # print('target.reshape(batch_size, n_docs, -1)', target.reshape(batch_size, n_docs, -1))
                first_token_target = target.reshape(batch_size, n_docs, -1)[:, :, target_index]
                # print('first_token_target', first_token_target)
                
                prediction_labels = (first_token_prediction == first_token_target)
                # print(prediction_labels)
                retrieval_labels = retrieval_labels.to(seq_logits.device).float()
                # print(retrieval_labels)

                RAVQA_loss_type = self.config.model_config.RAVQA_loss_type
                if RAVQA_loss_type == 'Approach5':
                    ##############   approach 5:  ##################
                    # correct prediction + positive pseudo label = 1
                    # wrong prediction + positive pseudo label = -100
                    # correct prediction + negative pseudo label = -100
                    # wrong prediction + negative pseudo label = -100
                    merged_labels = torch.logical_and(prediction_labels, retrieval_labels).float()
                    ignore_mask = (merged_labels==0)
                
                elif RAVQA_loss_type == 'Approach6':
                    ##############   approach 6:  ##################
                    # correct prediction + positive pseudo label = 1
                    # wrong prediction + positive pseudo label = -100
                    # correct prediction + negative pseudo label = -100
                    # wrong prediction + negative pseudo label = 0
                    merged_labels = torch.logical_and(prediction_labels, retrieval_labels).float()
                    ignore_mask = torch.logical_or(
                        torch.logical_and((prediction_labels==0), (retrieval_labels==1)),
                        torch.logical_and((prediction_labels==1), (retrieval_labels==0)),
                        )
                elif RAVQA_loss_type == 'NoPR':
                    ##############   approach NoPR:  ##################
                    # correct prediction = 1
                    # wrong prediction = 0
                    merged_labels = prediction_labels.float()
                    ignore_mask = torch.zeros_like(merged_labels).bool().to(merged_labels.device)


                doc_scores_softmaxed = F.softmax(doc_scores, dim=-1)

                dist_loss = F.binary_cross_entropy(doc_scores_softmaxed, merged_labels, reduction='none')
                dist_loss.masked_fill_(ignore_mask, 0.0)

                count_nonzero = torch.count_nonzero(dist_loss)
                if count_nonzero == 0:
                    dist_loss = 0
                else:
                    dist_loss = dist_loss.sum() / torch.count_nonzero(dist_loss)

                loss_dict.additional_loss = dist_loss
            else:
                loss_dict.additional_loss = 0
        
        if reduce_loss:
            mask = (smoothed_nll_loss != 0)
            smoothed_nll_loss = smoothed_nll_loss.sum()
            smoothed_nll_loss = smoothed_nll_loss / torch.sum(mask)
            loss_dict.nll_loss = smoothed_nll_loss

        return loss_dict
        




    def get_rag_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, ignore_index=-100, n_docs=None
    ):
        # print('seq_logits', seq_logits)
        # print('doc_scores', doc_scores)
        # print('target', target)

        # batch_size*n_docs x seq_len
        # shift tokens left
        # target = torch.cat(
        #     [target[:, 1:], target.new(target.shape[0], 1).fill_(self.generator.config.pad_token_id)], 1
        # )
        target = target.clone()
        ignore_index_pad_mask = target.eq(ignore_index)
        if ignore_index_pad_mask.any():
            # fill -100 to be 0, avoid indexing error using gather
            target.masked_fill_(ignore_index_pad_mask, self.generator.config.pad_token_id)
        
        n_docs = n_docs if n_docs is not None else self.config.data_loader.additional.num_knowledge_passages

        # bos_token_id is None for T5
        # bos_token_id = 0 for BART
        bos_token_id = self.generator.config.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()
        # print('use_bos', use_bos, 'exclude_bos_score', exclude_bos_score)
        # input()
        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.generator.config.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # calculate loss
        # batch_size*n_docs x seq_len --> batch_size x n_docs x seq_len x 1
        # target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        batch_size = seq_logits.shape[0] // n_docs
        target = target.reshape(batch_size, n_docs, -1).unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()
        # print('rag_logprobs', rag_logprobs.shape, rag_logprobs)
        # print('target', target.shape, target)
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        # print('after masking', ll)

        # sum over tokens, exclude bos while scoring
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    def get_retrieval_labels(self, 
                            question_ids: List,
                            batch_answers: List, 
                            batch_pos_item_ids: List,
                            batch_retrieved_docs: List,
                            pseudo: bool = False):
        
        retrieved_docs = batch_retrieved_docs
        log_result = {
            'recall': [],
            'precision': [],
        }
        labels = []
        for question_id, answer_list, pos_item_ids, docs in zip(question_ids, batch_answers, batch_pos_item_ids, retrieved_docs):
            
            doc_table_ids = [doc['table_id'] for doc in docs]

            this_batch_labels = []

            for doc in docs:
                valid = False
                retrieved_table_id = doc['table_id']
                if retrieved_table_id in pos_item_ids:
                    valid = True
                
                if pseudo:
                    contain_all = False
                    for answer in answer_list:
                        if answer.lower() in doc['table_contents'].lower():
                            contain_all = True
                    
                    if contain_all:
                        valid = True
                
                if valid:
                    this_batch_labels.append(1)
                else:
                    this_batch_labels.append(0)
                
            labels.append(this_batch_labels)
                    
        labels = torch.FloatTensor(labels)
        return EasyDict(
            retrieval_labels=labels,
        )
