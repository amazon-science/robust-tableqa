// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import 'tapex_ITR_column_wise_wtq.jsonnet';

// here we put the index file paths
local index_files = {
  "index_paths": {
    "train": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_single_row/test/wtq_original_sets/step_6803/test.ITRWikiTQDataset.train",
    "validation": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_single_row/test/wtq_original_sets/step_6803/test.ITRWikiTQDataset.validation",
    "test": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_single_row/test/wtq_original_sets/step_6803/test.ITRWikiTQDataset.test",
  },
};

local override = {
  "model_config": {
    "ModelClass": "ITRRagReduceRowWiseModel",
    "QueryEncoderModelVersion": "$DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_single_row/train/saved_model/step_6803/query_encoder",
    "index_files": index_files,
  },
  "data_loader": {
    "dataset_modules": {
      "module_dict":{
        "LoadWikiTQData": {
          "type": "LoadWikiTQData", "option": "default",
          "config": {
            "preprocess": ["split_table_by_row_combination"],
            "path": {
              "train": "TableQA_data/wtq/preprocessed_split_table_by_row_combination_single_train.arrow",
              "validation": "TableQA_data/wtq/preprocessed_split_table_by_row_combination_single_validation.arrow",
              "test": "TableQA_data/wtq/preprocessed_split_table_by_row_combination_single_test.arrow",
            }
          },
        },
      },
    },
  },
  "metrics": [
    {'name': 'compute_denotation_accuracy'},
    // {'name': 'compute_ITR_RAG_retrieval_results', 'option': 'row_wise'},
  ],
};

std.mergePatch(base_env, override)
