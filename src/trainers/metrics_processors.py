# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from tqdm import tqdm
from collections import defaultdict
import collections
from pprint import pprint
import string
import re

import wandb
import logging
logger = logging.getLogger(__name__)

def zero_div(x, y):
    return y and x / y or 0

def _split_thousands(delimiter, value):
    split = value.split(delimiter)
    return len(split) > 1 and any(map(lambda x: len(x) == 3, split))

def convert_to_float(value):
    """Converts value to a float using a series of increasingly complex heuristics.

    Args:
        value: object that needs to be converted. Allowed types include
        float/int/strings.

    Returns:
        A float interpretation of value.

    Raises:
        ValueError if the float conversion of value fails.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, str):
        raise ValueError("Argument value is not a string. Can't parse it as float")
    sanitized = value

    try:
        # Example: 1,000.7
        if "." in sanitized and "," in sanitized:
            return float(sanitized.replace(",", ""))
        # 1,000
        if "," in sanitized and _split_thousands(",", sanitized):
            return float(sanitized.replace(",", ""))
        # 5,5556
        if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
            ",", sanitized):
            return float(sanitized.replace(",", "."))
        # 0.0.0.1
        if sanitized.count(".") > 1:
            return float(sanitized.replace(".", ""))
        # 0,0,0,1
        if sanitized.count(",") > 1:
            return float(sanitized.replace(",", ""))
        return float(sanitized)
    except ValueError:
        # Avoid adding the sanitized value in the error message.
        raise ValueError("Unable to convert value to float")


class MetricsProcessor():
    '''
    Metrics processor, general class definitions
    This is to save all metrics that we want to compute
    Data required for metrics computation should be passed in
    And each metrics module will compute metrics and append an entry in metrics_dict
    '''
    def __init__(self) -> None:
        pass

    def compute_metrics(self, data_dict):
        '''
        Compute metrics
        '''
        log_dict = EasyDict({
            "metrics": {},
            "artifacts": {},
        })
        for metrics in self.config.metrics:
            compute_func = getattr(self, metrics.name)
            logger.info(f"Running metrics {str(metrics)}...")
            log_dict = compute_func(metrics, data_dict, log_dict)
            # print(f"Metrics columns {log_dict.metrics.keys()} ")

        return log_dict
    
    def compute_tapas_denotation_accuracy(self, module, data_dict, log_dict):
        """
        Compute denotation accuracy for TAPAS models
        option: default   denotation accuracy for all test samples
        option: valid_samples_only  denotation accuracy for only valid (pass the parsing and tokenization step) test samples
        """
        option = module.get('option', 'default')

        delimiter = ", "

        # define example evaluation
        def evaluate_example(predict_str: str, ground_str: str):
            predict_spans = predict_str.split(delimiter)
            predict_spans = [v for v in predict_spans if v != ""]
            ground_spans = ground_str.split(delimiter)
            ground_spans = [v for v in ground_spans if v != "none"]

            predict_values = defaultdict(lambda: 0)
            ground_values = defaultdict(lambda: 0)
            for span in predict_spans:
                try:
                    float_value = convert_to_float(span)
                    predict_values[float_value] += 1
                except ValueError:
                    predict_values[span.strip()] += 1
            for span in ground_spans:
                try:
                    float_value = convert_to_float(span)
                    ground_values[float_value] += 1
                except ValueError:
                    ground_values[span.strip()] += 1
            _is_correct = predict_values == ground_values
            return _is_correct
        
        def evaluate_example_with_operator(predict_float: str, ground_str: str):
            # in prediction step, the prediction has been set to float type
            
            ref_values = ground_str.split(delimiter)
            ref_values = [v for v in ref_values if v != "none"]
            converted_ref_values = []
            for ref_value in ref_values:
                try:
                    converted_ref_values.append(convert_to_float(ref_value))
                except ValueError:
                    converted_ref_values.append(ref_value)
            ref_values = converted_ref_values


            if predict_float == "":
                # reference is empty, selection=0, sum or average
                if len(ref_values) == 0:
                    return True
                else:
                    return False
            


            # transform predict_str to int
            try:
                predict_float = float(predict_float)
            except Exception as e:
                return False
            
            if len(ref_values) != 1:
                # currently, in the case of using an operator, the reference must be of length 1
                return False
            
            if not isinstance(ref_values[0], float):
                return False

            if abs(predict_float - ref_values[0]) < 1e-9:
                # set a tolerance
                return True
            else:
                return False


        def get_denotation_accuracy(predictions, references, pred_aggregation):
            assert len(predictions) == len(references)
            correct_num = 0
            for predict_str, ground_str, agg_str in zip(predictions, references, pred_aggregation):
                if agg_str == 'NONE':
                    # perform denotation as usual
                    is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
                    # print(predict_str, ground_str, is_correct)
                else:
                    # when the pred is done by operators
                    # transform the answer to floats
                    # transform the pred to floats
                    # check if they match!
                    is_correct = evaluate_example_with_operator(predict_str, ground_str.lower())
                    # print(predict_str, ground_str, is_correct)
                if is_correct:
                    correct_num += 1
            return correct_num / len(predictions)
        
        def get_denotation_accuracy_with_original_table_overflow(predictions, references, pred_aggregation, overflows, raw_cells):
            assert len(predictions) == len(references)
            assert len(predictions) == len(overflows)
            overflow_correct_num = 0
            non_overflow_correct_num = 0
            overflow_num = 0
            non_overflow_num = 0
            for predict_str, ground_str, agg_str, original_table_overflow, raw_cell_values in zip(predictions, references, pred_aggregation, overflows, raw_cells):
                if agg_str == 'NONE':
                    # perform denotation as usual
                    is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
                    # print("label:", ground_str, "\t prediction:", predict_str, agg_str, f"\t raw: {raw_cell_values}",  "\t", is_correct)
                else:
                    # when the pred is done by operators
                    # transform the answer to floats
                    # transform the pred to floats
                    # check if they match!
                    is_correct = evaluate_example_with_operator(predict_str, ground_str.lower())
                    # print("label:", ground_str, "\t prediction:", predict_str, agg_str, f"\t raw: {raw_cell_values}",  "\t", is_correct)
                
                if original_table_overflow:
                    overflow_num += 1
                else:
                    non_overflow_num += 1
                if is_correct:
                    if original_table_overflow:
                        overflow_correct_num += 1
                    else:
                        non_overflow_correct_num += 1
            
            return overflow_correct_num / len(predictions), non_overflow_correct_num / len(predictions), zero_div(overflow_correct_num, overflow_num), zero_div(non_overflow_correct_num, non_overflow_num)
        
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]
            return preds, labels
        
        

        decoded_preds = []
        decoded_labels = []
        pred_aggregation = []
        final_pred = []
        raw_cells = []
        check_original_table_overflow = ('original_table_overflow' in data_dict['batch_predictions'][0].keys())
        if check_original_table_overflow:
            original_table_overflows = []
        
        for pred_dict in data_dict['batch_predictions']:
            if option == 'valid_samples_only':
                # only calculate valid samples
                if pred_dict['valid'] == False:
                    continue
            decoded_preds.append(pred_dict['decoded_output'])
            decoded_labels.append(pred_dict['decoded_label'])
            pred_aggregation.append(pred_dict['predicted_agg'])
            final_pred.append(pred_dict['final_pred'])
            raw_cells.append(pred_dict['raw_cells'])
            if check_original_table_overflow:
                original_table_overflows.append(pred_dict.get('original_table_overflow', True))
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        accuracy = get_denotation_accuracy(final_pred, decoded_labels, pred_aggregation)
        
        if option == 'valid_samples_only':
            prefix = "valid_samples_"
        else:
            prefix = ""
        
        log_dict.metrics[f'{prefix}denotation_accuracy'] = accuracy
        
        if check_original_table_overflow:
            overflow_accuracy_global, non_overflow_accuracy_global, overflow_accuracy, non_overflow_accuracy = get_denotation_accuracy_with_original_table_overflow(decoded_preds, decoded_labels, pred_aggregation, original_table_overflows, raw_cells)
            log_dict.metrics[f'{prefix}global_overflow_denotation_accuracy'] = overflow_accuracy_global
            log_dict.metrics[f'{prefix}global_non_overflow_denotation_accuracy'] = non_overflow_accuracy_global
            log_dict.metrics[f'{prefix}overflow_denotation_accuracy'] = overflow_accuracy
            log_dict.metrics[f'{prefix}non_overflow_denotation_accuracy'] = non_overflow_accuracy
        # input('done')
        return log_dict

    def compute_denotation_accuracy(self, module, data_dict, log_dict):
	         
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            return preds, labels
        
        decoded_preds = [pred_dict['decoded_output'] for pred_dict in data_dict['batch_predictions']]
        decoded_labels = [pred_dict['decoded_label'] for pred_dict in data_dict['batch_predictions']]
        
        check_original_table_overflow = ('original_table_overflow' in data_dict['batch_predictions'][0].keys())
        if check_original_table_overflow:
            original_table_overflows = [pred_dict.get('original_table_overflow', True) for pred_dict in data_dict['batch_predictions']]

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        delimiter = ", "

        vague_match = module.get("vague", 0)
        squad_normalization = module.get('squad_normalization', 0)
        
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            def remove_articles(text):
                regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
                return re.sub(regex, ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))

        # define example evaluation
        def evaluate_example(predict_str: str, ground_str: str):
            predict_spans = predict_str.split(delimiter)
            ground_spans = ground_str.split(delimiter)

            if squad_normalization:
                predict_spans = [normalize_answer(a) for a in predict_spans]
                ground_spans = [normalize_answer(a) for a in ground_spans]
                
            predict_values = defaultdict(lambda: 0)
            ground_values = defaultdict(lambda: 0)
            for span in predict_spans:
                try:
                    predict_values[float(span)] += 1
                except ValueError:
                    predict_values[span.strip()] += 1
            for span in ground_spans:
                try:
                    ground_values[float(span)] += 1
                except ValueError:
                    ground_values[span.strip()] += 1
            _is_correct = predict_values == ground_values
            if not _is_correct and vague_match:
                if len(ground_spans) == 1 and len(predict_spans) == 1:
                    pred_value = predict_spans[0]
                    gr_value = ground_spans[0]
                    try:
                        pred_value = convert_to_float(pred_value)
                        gr_value = convert_to_float(gr_value)
                    except Exception as e:
                        pred_value = predict_spans[0]
                        gr_value = ground_spans[0]
                    
                    if isinstance(pred_value, str) and isinstance(gr_value, str):
                        if pred_value in gr_value:
                            _is_correct = True
                        if gr_value in pred_value:
                            _is_correct = True
                    # print(ground_spans, predict_spans, _is_correct)

            return _is_correct

        def get_denotation_accuracy(predictions, references):
            assert len(predictions) == len(references)
            correct_num = 0
            for predict_str, ground_str in zip(predictions, references):
                is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
                # print(predict_str, '\t', ground_str, '\t', is_correct)
                # input()
                if is_correct:
                    correct_num += 1
            return correct_num / len(predictions)
        
        def get_denotation_accuracy_with_original_table_overflow(predictions, references, overflows):
            assert len(predictions) == len(references)
            assert len(predictions) == len(overflows)
            overflow_correct_num = 0
            non_overflow_correct_num = 0
            overflow_num = 0
            non_overflow_num = 0
            for predict_str, ground_str, original_table_overflow in zip(predictions, references, overflows):
                is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
                if original_table_overflow:
                    overflow_num += 1
                else:
                    non_overflow_num += 1
                if is_correct:
                    if original_table_overflow:
                        overflow_correct_num += 1
                    else:
                        non_overflow_correct_num += 1
            return overflow_correct_num / len(predictions), non_overflow_correct_num / len(predictions), zero_div(overflow_correct_num, overflow_num), zero_div(non_overflow_correct_num, non_overflow_num)

        accuracy = get_denotation_accuracy(decoded_preds, decoded_labels)
        
        log_dict.metrics['denotation_accuracy'] = accuracy

        if check_original_table_overflow:
            overflow_accuracy_global, non_overflow_accuracy_global, overflow_accuracy, non_overflow_accuracy = get_denotation_accuracy_with_original_table_overflow(decoded_preds, decoded_labels, original_table_overflows)
            log_dict.metrics['global_overflow_denotation_accuracy'] = overflow_accuracy_global
            log_dict.metrics['global_non_overflow_denotation_accuracy'] = non_overflow_accuracy_global
            log_dict.metrics['overflow_denotation_accuracy'] = overflow_accuracy
            log_dict.metrics['non_overflow_denotation_accuracy'] = non_overflow_accuracy
    
        return log_dict
    
    def compute_accuracy_with_alternative_answers(self, module, data_dict, log_dict):
        
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            return preds, labels
        
        decoded_preds = [pred_dict['decoded_output'] for pred_dict in data_dict['batch_predictions']]
        decoded_labels = [pred_dict['decoded_label'] for pred_dict in data_dict['batch_predictions']]
        alternative_labels = [pred_dict['alternative_answers'] for pred_dict in data_dict['batch_predictions']]
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        alternative_labels = [
            [l.strip() for l in labels] for labels in alternative_labels
        ]

        delimiter = ", "

        squad_normalization = module.get('squad_normalization', 0)
        
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            def remove_articles(text):
                regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
                return re.sub(regex, ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))

        # define example evaluation
        def evaluate_example(predict_str: str, ground_str: str):
            predict_spans = predict_str.split(delimiter)
            ground_spans = ground_str.split(delimiter)

            if squad_normalization:
                predict_spans = [normalize_answer(a) for a in predict_spans]
                ground_spans = [normalize_answer(a) for a in ground_spans]
                
            predict_values = defaultdict(lambda: 0)
            ground_values = defaultdict(lambda: 0)
            for span in predict_spans:
                try:
                    predict_values[float(span)] += 1
                except ValueError:
                    predict_values[span.strip()] += 1
            for span in ground_spans:
                try:
                    ground_values[float(span)] += 1
                except ValueError:
                    ground_values[span.strip()] += 1
            _is_correct = predict_values == ground_values
            # if not _is_correct:
            #     if len(ground_spans) == 1 and len(predict_spans) == 1:
            #         pred_value = predict_spans[0]
            #         gr_value = ground_spans[0]
            #         try:
            #             pred_value = convert_to_float(pred_value)
            #             gr_value = convert_to_float(gr_value)
            #         except Exception as e:
            #             pred_value = predict_spans[0]
            #             gr_value = ground_spans[0]
                    
            #         if isinstance(pred_value, str) and isinstance(gr_value, str):
            #             if pred_value != '':
            #                 if pred_value in gr_value:
            #                     _is_correct = True
            #                 if gr_value in pred_value:
            #                     _is_correct = True
            #         # print(ground_spans, predict_spans, _is_correct)

            return _is_correct

        def get_denotation_accuracy(predictions, references, alternative_answers):
            assert len(predictions) == len(references)
            correct_num = 0
            for predict_str, ground_str, alternative_strs in zip(predictions, references, alternative_answers):
                is_correct = False
                if evaluate_example(predict_str.lower(), ground_str.lower()):
                    is_correct = True
                for alternative_str in alternative_strs:
                    if evaluate_example(predict_str.lower(), alternative_str.lower()):
                        is_correct = True
                print(predict_str, '\t', ground_str, '\t', alternative_strs,  '\t', is_correct)
                # input()
                if is_correct:
                    correct_num += 1
            return correct_num / len(predictions)
        
        accuracy = get_denotation_accuracy(decoded_preds, decoded_labels, alternative_labels)
        
        log_dict.metrics['denotation_accuracy'] = accuracy

        return log_dict

    def compute_ITR_RAG_retrieval_results(self, module, data_dict, log_dict) -> dict:
        """
        Evaluate the retrieval performance of Inner Table Retrieval - RAG Mode
        Args:
            batch_retrieval_results: list of retrieval results
                            each is {
                                'question_id': question_id,
                                'retrieved_tables_sorted': [(sub_table, rating), ...],
                                'gold_columns': [list of gold columns],
                                'gold_rows': [list of gold rows],
                            }
        """
        batch_results = data_dict['batch_retrieval_results']
        Ks = self.config.model_config.Ks
        # only K < num_knowledge_passages makes sense
        Ks = [K for K in Ks if K <= self.config.data_loader.additional.num_knowledge_passages]
        
        ref_mode = module.get('option', 'column_wise')
        if ref_mode == 'column_wise':
            gold_ref = 'gold_columns'
            indice_ref = 'sub_column_indice'
        else:
            gold_ref = 'gold_rows'
            indice_ref = 'sub_row_indice'
        
        count = len(batch_results)
        result = {
            'full_recall': np.zeros(len(Ks)),
            'overflow_rate': 0,
            'non_overflow_full_recall_rate': 0,
            'overflow_resolve_success_rate': 0,
            'overflow_resolve_fail_rate': 0,
        }
        max_input_length = self.config.data_loader.additional.max_decoder_source_length
        for retrieval_result in batch_results:
            question_id = retrieval_result['question_id']
            retrieved_tables = retrieval_result['retrieved_tables_sorted']
            gold_reference = set(retrieval_result[gold_ref])

            original_table_overflow = retrieved_tables[0][0]['original_table_overflow']
            num_hit_list = []
            overflow = []
            
            for retrieved_sub_table, rating in retrieved_tables:
                if gold_reference.issubset(set(retrieved_sub_table[indice_ref])):
                    num_hit_list.append(True)
                else:
                    num_hit_list.append(False)
                if retrieved_sub_table['input_length'] > max_input_length:
                    overflow.append(1)
                else:
                    overflow.append(0)
            
            # if there is a table that does not overflow, the rate is considered 0
            # if all the subtables overflow, the rate is 1
            is_overflow = (sum(overflow) >= len(overflow))

            result['overflow_rate'] += is_overflow

            # whether there is a non-overflow full recall in the retrieved sub tables
            non_overflow_full_recall = False
            
            for h, o in zip(num_hit_list, overflow):
                if h==1 and o==0:
                    non_overflow_full_recall = True
            
            # whether an originally overflow table gets non-overflow full recall
            overflow_resolve_success = non_overflow_full_recall and original_table_overflow
            overflow_resolve_fail = (not non_overflow_full_recall) and original_table_overflow

            result['non_overflow_full_recall_rate']+= non_overflow_full_recall
            result['overflow_resolve_success_rate']+= overflow_resolve_success
            result['overflow_resolve_fail_rate']+= overflow_resolve_fail
            

            for index, K in enumerate(Ks):
                # obtain the list wrt K
                truncated_num_hit_list = num_hit_list[:min(K, len(num_hit_list))]
                
                # reduce full recall measures whether all gold columns are collected
                full_recall_at_K = (sum(truncated_num_hit_list) > 0)

                result['full_recall'][index] += full_recall_at_K

        result['full_recall'] = result['full_recall']/count
        result['overflow_rate'] = result['overflow_rate']/count
        result['non_overflow_full_recall_rate'] = result['non_overflow_full_recall_rate']/count
        result['overflow_resolve_success_rate'] = result['overflow_resolve_success_rate']/count
        result['overflow_resolve_fail_rate'] = result['overflow_resolve_fail_rate']/count

        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            if metrics_name not in ['full_recall']:
                log_result[metrics_name] = np_array
            else:
                for index, K in enumerate(Ks):
                    log_result[f'{metrics_name}_at_{K}'] = float(np_array[index])
        log_dict.metrics.update(log_result)
        return log_dict

    def compute_ITR_mix_RAG_retrieval_results(self, module, data_dict, log_dict) -> dict:
        """
        Evaluate the retrieval performance of Inner Table Retrieval - RAG Mode
        Args:
            batch_retrieval_results: list of retrieval results
                            each is {
                                'question_id': question_id,
                                'retrieved_tables_sorted': [(sub_table, rating), ...],
                                'gold_columns': [list of gold columns],
                                'gold_rows': [list of gold rows],
                            }
        """
        batch_results = data_dict['batch_retrieval_results']
        Ks = self.config.model_config.Ks
        # only K < num_knowledge_passages makes sense
        Ks = [K for K in Ks if K <= self.config.data_loader.additional.num_knowledge_passages]
        
        def get_cells_from_col_row_index(row_indice, col_indice):
            cells = []
            for row_index in row_indice:
                for column_index in col_indice:
                    cells.append((row_index, column_index))
            return set(cells)
        
        count = len(batch_results)
        result = {
            'full_recall': np.zeros(len(Ks)),
            'overflow_rate': 0,
            'non_overflow_full_recall_rate': 0,
            'overflow_resolve_success_rate': 0,
            'overflow_resolve_fail_rate': 0,
        }
        max_input_length = self.config.data_loader.additional.max_decoder_source_length
        for retrieval_result in batch_results:
            question_id = retrieval_result['question_id']
            retrieved_tables = retrieval_result['retrieved_tables_sorted']
            gold_reference = get_cells_from_col_row_index(
                retrieval_result['gold_rows'], retrieval_result['gold_columns']
            )

            original_table_overflow = retrieved_tables[0][0]['original_table_overflow']
            num_hit_list = []
            overflow = []
            
            for retrieved_sub_table, rating in retrieved_tables:
                retrieved_cells = get_cells_from_col_row_index(
                    retrieved_sub_table['sub_row_indice'], retrieved_sub_table['sub_column_indice']
                )
                if gold_reference.issubset(retrieved_cells):
                    num_hit_list.append(True)
                else:
                    num_hit_list.append(False)
                if retrieved_sub_table['input_length'] > max_input_length:
                    overflow.append(1)
                else:
                    overflow.append(0)
            
            # if there is a table that does not overflow, the rate is considered 0
            # if all the subtables overflow, the rate is 1
            is_overflow = (sum(overflow) >= len(overflow))

            result['overflow_rate'] += is_overflow

            # whether there is a non-overflow full recall in the retrieved sub tables
            non_overflow_full_recall = False
            
            for h, o in zip(num_hit_list, overflow):
                if h==1 and o==0:
                    non_overflow_full_recall = True
            
            # whether an originally overflow table gets non-overflow full recall
            overflow_resolve_success = non_overflow_full_recall and original_table_overflow
            overflow_resolve_fail = (not non_overflow_full_recall) and original_table_overflow

            result['non_overflow_full_recall_rate']+= non_overflow_full_recall
            result['overflow_resolve_success_rate']+= overflow_resolve_success
            result['overflow_resolve_fail_rate']+= overflow_resolve_fail
            

            for index, K in enumerate(Ks):
                # obtain the list wrt K
                truncated_num_hit_list = num_hit_list[:min(K, len(num_hit_list))]
                
                # reduce full recall measures whether all gold columns are collected
                full_recall_at_K = (sum(truncated_num_hit_list) > 0)

                result['full_recall'][index] += full_recall_at_K

        result['full_recall'] = result['full_recall']/count
        result['overflow_rate'] = result['overflow_rate']/count
        result['non_overflow_full_recall_rate'] = result['non_overflow_full_recall_rate']/count
        result['overflow_resolve_success_rate'] = result['overflow_resolve_success_rate']/count
        result['overflow_resolve_fail_rate'] = result['overflow_resolve_fail_rate']/count

        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            if metrics_name not in ['full_recall']:
                log_result[metrics_name] = np_array
            else:
                for index, K in enumerate(Ks):
                    log_result[f'{metrics_name}_at_{K}'] = float(np_array[index])
        log_dict.metrics.update(log_result)
        return log_dict
    
    def compute_ITR_retrieval_results(self, module, data_dict, log_dict) -> dict:
        """
        Evaluate the retrieval performance of Inner Table Retrieval
        Args:
            batch_retrieval_results: list of retrieval results
                            each is {
                                'question_id': question_id,
                                'retrieved_tables_sorted': [(sub_table, rating), ...],
                                'gold_columns': [list of gold columns],
                                'gold_rows': [list of gold rows],
                            }
        """
        batch_results = data_dict['batch_retrieval_results']
        Ks = self.config.model_config.Ks
        truncate = module.get('truncate', False)
        ref_mode = module.get('option', 'column_wise')
        if ref_mode == 'column_wise':
            gold_ref = 'gold_columns'
        else:
            gold_ref = 'gold_rows'

        if truncate:
            # only K < num_knowledge_passages makes sense
            Ks = [K for K in Ks if K <= self.config.data_loader.additional.num_knowledge_passages]
        
        count = len(batch_results)
        result = {
            'precision': np.zeros(len(Ks)),
            'recall': np.zeros(len(Ks)),
            'full_recall': np.zeros(len(Ks)),
        }

        for retrieval_result in batch_results:
            question_id = retrieval_result['question_id']
            retrieved_tables = retrieval_result['retrieved_tables_sorted']
            num_hit_list = []
            for retrieved_sub_table, rating in retrieved_tables:
                num_hit_list.append((retrieved_sub_table['is_gold']==True))
            for index, K in enumerate(Ks):
                # obtain the list wrt K
                truncated_num_hit_list = num_hit_list[:min(K, len(num_hit_list))]
                # compute the recall and precision
                recall_at_K = int(sum(truncated_num_hit_list)>0)
                precision_at_K = sum(truncated_num_hit_list) / len(truncated_num_hit_list)
                # full recall measures whether all gold columns are collected
                full_recall_at_K = (len(retrieval_result[gold_ref]) <= sum(truncated_num_hit_list))

                result['recall'][index] += recall_at_K
                result['precision'][index] += precision_at_K
                result['full_recall'][index] += full_recall_at_K

        result['precision'] = result['precision']/count
        result['recall'] = result['recall']/count
        result['full_recall'] = result['full_recall']/count
        
        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            for index, K in enumerate(Ks):
                log_result[f'{metrics_name}_at_{K}'] = float(np_array[index])

        log_dict.metrics.update(log_result)
        return log_dict
    
    def compute_ITR_mix_retrieval_results(self, module, data_dict, log_dict) -> dict:
        """
        Evaluate the retrieval performance of Inner Table Retrieval (mixing row and column)
        Args:
            batch_retrieval_results: list of retrieval results
                            each is {
                                'question_id': question_id,
                                'retrieved_tables_sorted': [(sub_table, rating), ...],
                                'gold_columns': [list of gold columns],
                                'gold_rows': [list of gold rows],
                            }
        """

        def get_cells_from_col_row_index(row_indice, col_indice):
            cells = []
            for row_index in row_indice:
                for column_index in col_indice:
                    cells.append((row_index, column_index))
            return set(cells)

        batch_results = data_dict['batch_retrieval_results']
        Ks = self.config.model_config.Ks
        truncate = module.get('truncate', False)
        
        # we are mixing the row subtables and column subtables

        if truncate:
            # only K < num_knowledge_passages makes sense
            Ks = [K for K in Ks if K <= self.config.data_loader.additional.num_knowledge_passages]
        
        count = len(batch_results)
        result = {
            'precision': np.zeros(len(Ks)),
            'recall': np.zeros(len(Ks)),
            'full_recall': np.zeros(len(Ks)),
        }

        for retrieval_result in batch_results:
            question_id = retrieval_result['question_id']
            retrieved_tables = retrieval_result['retrieved_tables_sorted']

            gold_rows = retrieval_result['gold_rows']
            gold_columns = retrieval_result['gold_columns']
            gold_cells = get_cells_from_col_row_index(gold_rows, gold_columns)
            

            num_hit_list = []
            retrieved_cells_all = []
            current_cols = []
            current_rows = []
            for retrieved_sub_table, rating in retrieved_tables:
                num_hit_list.append((retrieved_sub_table['is_gold']==True))
                if retrieved_sub_table['sub_type'] == 'row_wise':
                    current_rows += retrieved_sub_table['sub_row_indice']
                else:
                    current_cols += retrieved_sub_table['sub_column_indice']
                current_cells = get_cells_from_col_row_index(current_rows, current_cols)
                retrieved_cells_all.append(current_cells)

            for index, K in enumerate(Ks):
                # obtain the list wrt K
                truncated_num_hit_list = num_hit_list[:min(K, len(num_hit_list))]
                
                # compute the recall and precision
                recall_at_K = int(sum(truncated_num_hit_list)>0)
                precision_at_K = sum(truncated_num_hit_list) / len(truncated_num_hit_list)
                # full recall measures whether all gold columns are collected
                full_recall_at_K = gold_cells.issubset(retrieved_cells_all[min(K, len(retrieved_cells_all))-1])
                result['recall'][index] += recall_at_K
                result['precision'][index] += precision_at_K
                result['full_recall'][index] += full_recall_at_K
            
        
        result['precision'] = result['precision']/count
        result['recall'] = result['recall']/count
        result['full_recall'] = result['full_recall']/count
        
        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            for index, K in enumerate(Ks):
                log_result[f'{metrics_name}_at_{K}'] = float(np_array[index])

        log_dict.metrics.update(log_result)
        return log_dict

    def compute_TQA_DPR_scores(self, module, data_dict, log_dict) -> dict:
        """
        Compute DPR scores for Open-domain TQA
        """
        batch_results = data_dict['batch_retrieval_results']
        Ks = data_dict['Ks']

        # Total number of questions
        count = len(batch_results)
        result = {
            'precision': np.zeros(len(Ks)),
            'recall': np.zeros(len(Ks)),
        }
        for retrieved_results in batch_results:
            retrieved_tables_sorted = retrieved_results['retrieved_tables_sorted']
            pos_item_ids = retrieved_results['pos_item_ids']
            hit_list = [retrieved_table_id in pos_item_ids for retrieved_table_id in retrieved_tables_sorted]
            for index, K in enumerate(Ks):
                truncated_list = hit_list[:K]
                recall_at_K = int(sum(truncated_list) > 0)
                precision_at_K = sum(truncated_list) / K
                result['recall'][index] += recall_at_K
                result['precision'][index] += precision_at_K
        
        
        result['precision'] = result['precision']/count
        result['recall'] = result['recall']/count
        
        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            for index, K in enumerate(Ks):
                log_result[f'{metrics_name}_at_{K}'] = float(np_array[index])

        log_dict.metrics.update(log_result)
        return log_dict

    def compute_RAG_retrieval_results(self, module, data_dict, log_dict) -> dict:
        """
        Compute retrieval scores for Open-domain TQA
        """
        retrieved_docs = data_dict['batch_retrieved_docs']
        batch_pos_item_ids = data_dict['batch_pos_item_ids']
        question_ids = data_dict['batch_question_ids']
        batch_answers = data_dict['batch_answers']

        log_result = {
            'recall': [],
            'precision': [],
        }
        labels = []
        for question_id, answer_list, pos_item_ids, docs in zip(question_ids, batch_answers, batch_pos_item_ids, retrieved_docs):
            K = len(docs)
            doc_table_ids = [doc['table_id'] for doc in docs]

            this_batch_labels = []
            for retrieved_table_id in doc_table_ids:
                if retrieved_table_id in pos_item_ids:
                    this_batch_labels.append(1)
                else:
                    this_batch_labels.append(0)
                
            labels.append(this_batch_labels)
            if sum(this_batch_labels) > 0:
                log_result['recall'].append(1)
            else:
                log_result['recall'].append(0)
            log_result['precision'].append(sum(this_batch_labels)/K)

        for metric in log_result.keys():
            log_result[metric] = np.mean(np.array(log_result[metric]))
        
        log_result['n_retrieved_docs'] = K

        log_dict.metrics.update(log_result)
        return log_dict

    def compute_token_f1(self, module, data_dict, log_dict):
        """
        This function computes token f1
        Some helper functions are taken from SQUAD: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
        """
        
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            return preds, labels
        
        decoded_preds = [pred_dict['decoded_output'] for pred_dict in data_dict['batch_predictions']]
        decoded_labels = [pred_dict['decoded_label'] for pred_dict in data_dict['batch_predictions']]
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        delimiter = ", "

        vague_match = module.get("vague", 0)
        squad_normalization = module.get('squad_normalization', 0)
        
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            def remove_articles(text):
                regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
                return re.sub(regex, ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def get_tokens(s):
            if not s: return []
            return normalize_answer(s).split()
        
        def compute_f1(a_gold, a_pred):
            gold_toks = get_tokens(a_gold)
            pred_toks = get_tokens(a_pred)
            common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
            num_same = sum(common.values())
            if len(gold_toks) == 0 or len(pred_toks) == 0:
                # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                return int(gold_toks == pred_toks)
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        # define example evaluation
        def evaluate_example(predict_str: str, ground_str: str):
            predict_spans = predict_str.split(delimiter)
            ground_spans = ground_str.split(delimiter)
            
            f1_score = compute_f1(' '.join(ground_spans), ' '.join(predict_spans))

            return f1_score

        def get_f1(predictions, references):
            assert len(predictions) == len(references)
            f1 = 0
            for predict_str, ground_str in zip(predictions, references):
                f1_score = evaluate_example(predict_str.lower(), ground_str.lower())
                # print(predict_str, '\t', ground_str, '\t', f1_score)
                # input()
                f1 += f1_score
            return f1 / len(predictions)
        
        f1 = get_f1(decoded_preds, decoded_labels)
        
        log_dict.metrics['token_f1'] = f1

        return log_dict
