// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import 'dpr_ITR_wikisql.jsonnet';

local override = {
  "data_loader": {
    "dataset_modules": {
      "module_dict":{
        "LoadWikiSQLData": {
          "type": "LoadWikiSQLData", "option": "default",
          "config": {
            "preprocess": ["move_answers_to_table_end_column_wise", "split_table_by_column_combination"],
            "path": {
              "train": "TableQA_data/wikisql/preprocessed_move_answers_to_table_end_column_wise_split_table_by_column_combination_single_train.arrow",
              "validation": "TableQA_data/wikisql/preprocessed_move_answers_to_table_end_column_wise_split_table_by_column_combination_single_validation.arrow",
              "test": "TableQA_data/wikisql/preprocessed_move_answers_to_table_end_column_wise_split_table_by_column_combination_single_test.arrow",
            }
          },
        },
      },
    },
  },
};

std.mergePatch(base_env, override)
