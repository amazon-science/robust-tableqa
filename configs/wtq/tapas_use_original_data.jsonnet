// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import '../tapas_base.jsonnet';


local override = {
  "model_config": {
    "ModelVersion": "google/tapas-base-finetuned-wtq",
  },
  "data_loader": {
    "dataset_modules": {
      "module_dict":{
        "LoadWikiTQData": {
          "config": {
            "preprocess": ["use_original_tapas_data", "check_tapas_tokenization_compatibility"],
            "tapas_path": {
                "train": "TableQA_data/wtq/tf_examples/random-split-1-train.pkl",
                "validation": "TableQA_data/wtq/tf_examples/random-split-1-dev.pkl",
                "test": "TableQA_data/wtq/tf_examples/test.pkl",
            },
            "path": {
              "train": "TableQA_data/wtq/preprocessed_train_full.arrow",
              "validation": "TableQA_data/wtq/preprocessed_validation_full.arrow",
              "test": "TableQA_data/wtq/preprocessed_test_full.arrow",
            },
          }
        },
      },
    },
  },
};

std.mergePatch(base_env, override)
