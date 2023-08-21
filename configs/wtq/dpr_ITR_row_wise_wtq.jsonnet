// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import 'dpr_ITR_column_wise_wtq.jsonnet';

local override = {
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
  "metrics": [],
};

std.mergePatch(base_env, override)
