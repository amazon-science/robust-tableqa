// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import '../base_env.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 200;
local save_interval = 200;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2022;



local override = {
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "DPR",
    "ModelClass": "RetrieverDPR",
    "QueryEncoderModelClass": "DPRQuestionEncoder",
    "QueryEncoderConfigClass": "DPRConfig",
    "QueryEncoderModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    "ItemEncoderModelClass": "DPRContextEncoder",
    "ItemEncoderConfigClass": "DPRConfig",
    "ItemEncoderModelVersion": "facebook/dpr-ctx_encoder-single-nq-base",
    "TokenizerClass": "DPRQuestionEncoderTokenizer",
    "TokenizerModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    "DecoderTokenizerClass": "DPRContextEncoderTokenizer",
    "DecoderTokenizerModelVersion": "facebook/dpr-ctx_encoder-single-nq-base",
    "pretrained": 1,
    "modules": [
      "separate_query_and_item_encoders",
    ],
    "Ks": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100],
    "num_negative_samples": 4,
    "prepend_tokens": {
      "query_encoder": "",
      "item_encoder": "",
    },
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
    "DECODER_SPECIAL_TOKENS":{
      "additional_special_tokens": ["<HEADER>", "<HEADER_SEP>", "<HEADER_END>", "<ROW>", "<ROW_SEP>", "<ROW_END>"],
    },
    "input_modules": {
      "module_list":[
        {"type": "QuestionInput",  "option": "default", 
                  "separation_tokens": {"start": "", "end": ""}},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessInputTokenization", "option": "default"},
      ],
    },
    "decoder_input_modules": {
      "module_list":[
        {"type": "TextBasedTableInput",  "option": "default",
                  "separation_tokens": {"header_start": "<HEADER>", "header_sep": "<HEADER_SEP>", "header_end": "<HEADER_END>", "row_start": "<ROW>", "row_sep": "<ROW_SEP>", "row_end": "<ROW_END>"}},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessDecoderInputTokenization", "option": "default"},
      ],
    },
    "output_modules": {
      "module_list":[
        {"type": "SimilarityOutput", "option": "default"},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessConcatenateLabels", "option": "default"},
      ],
    },
  },
  "data_loader": {
    "type": "DataLoaderForTableQA",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':512,
      'max_decoder_source_length': 512,
      'max_target_length':128,
    },
    "dataset_modules": {
      "module_list": [
        "LoadWikiSQLData",
        "LoadDataLoaders",
      ],
      "module_dict":{
        "LoadWikiSQLData": {
          "type": "LoadWikiSQLData", "option": "default",
          "config": {
            "preprocess": ["split_table_by_column_combination"],
            "path": {
              "train": "TableQA_data/wikisql/preprocessed_split_table_by_column_combination_single_train.arrow",
              "validation": "TableQA_data/wikisql/preprocessed_split_table_by_column_combination_single_validation.arrow",
              "test": "TableQA_data/wikisql/preprocessed_split_table_by_column_combination_single_test.arrow",
            }
          },
        },
        "LoadDataLoaders": {
          "type": "LoadDataLoaders", "option": "default",
          "config": {
            "train": [
                {
                    "dataset_type": "ITRWikiSQLDataset",
                    "split": "train",
                    "use_column": "wikisql_data",
                },
            ],
            "valid": [
                {
                    "dataset_type": "ITRWikiSQLDataset",
                    "split": "validation",
                    "use_column": "wikisql_data",
                },
                {
                    "dataset_type": "ITRWikiSQLDataset",
                    "split": "test",
                    "use_column": "wikisql_data",
                },
            ],
            "test": [
                {
                    "dataset_type": "ITRWikiSQLDataset",
                    "split": "train",
                    "use_column": "wikisql_data",
                },
                {
                    "dataset_type": "ITRWikiSQLDataset",
                    "split": "validation",
                    "use_column": "wikisql_data",
                },
                {
                    "dataset_type": "ITRWikiSQLDataset",
                    "split": "test",
                    "use_column": "wikisql_data",
                },
            ],
          }
        }
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "ITRDPRExecutor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "save_interval":save_interval,
    "scheduler": "none",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
        "save_top_k_metric": "valid/ITRWikiSQLDataset.validation/full_recall_at_5",
    }
  },
  "valid": {
    "batch_size":valid_batch_size,
    "step_size":valid_step_size,
    "break_interval": break_interval,
    "additional": {
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "additional": {
        "multiprocessing": 4,
    },
  },
  "metrics": [
    {'name': 'compute_ITR_retrieval_results'},
  ],
};

std.mergePatch(base_env, override)
