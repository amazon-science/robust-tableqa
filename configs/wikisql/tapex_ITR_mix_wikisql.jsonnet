// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import 'tapex_ITR_wikisql.jsonnet';

// here we put the index file paths
local index_files = {
  "index_paths": {
    "train": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/original_sets/step_11604/test.ITRWikiSQLDataset.train",
    "validation": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/original_sets/step_11604/test.ITRWikiSQLDataset.validation",
    "test": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/original_sets/step_11604/test.ITRWikiSQLDataset.test",
  },
};

local override = {
  "model_config": {
    "ModelClass": "ITRRagReduceMixModel",
    "QueryEncoderModelVersion": "$DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/train/saved_model/step_11604/query_encoder",
    "min_columns": 2,
    "index_files": index_files,
  },
  "data_loader": {
    "dataset_modules": {
      "module_dict":{
        "LoadWikiSQLData": {
          "type": "LoadWikiSQLData", "option": "default",
          "config": {
            "preprocess": ["split_table_by_mixed_combination"],
            "path": {
              "train": "TableQA_data/wikisql/preprocessed_split_table_by_mixed_combination_train.arrow",
              "validation": "TableQA_data/wikisql/preprocessed_split_table_by_mixed_combination_validation.arrow",
              "test": "TableQA_data/wikisql/preprocessed_split_table_by_mixed_combination_test.arrow",
            }
          },
        },
      },
    },
  },
  "metrics": [
    {'name': 'compute_denotation_accuracy'},
    {'name': 'compute_ITR_mix_RAG_retrieval_results', 'option': 'default'},
  ],
};

std.mergePatch(base_env, override)
