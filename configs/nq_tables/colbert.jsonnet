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
  "WANDB": {
    "tags": [],
  },
  "model_config": {
    "base_model": "ColBERT",
    "ModelClass": "ColBERT",
    "TokenizerClass": "QueryTokenizer",
    "DecoderTokenizerClass": "DocTokenizer",
    "EncoderModelVersion": "$TableQA_data/checkpoints/colbertv2.0",
    "pretrained": 1,
    "modules": [
      "separate_query_and_item_encoders",
    ],
    "Ks": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100],
    "num_negative_samples": 4,
    "bm25_ratio": 0,
    "bm25_top_k": 3,
    "nbits": 16,
    "prepend_tokens": {
      "query_encoder": "",
      "item_encoder": "",
    },
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
    "DECODER_SPECIAL_TOKENS":{
      "additional_special_tokens": ["<HEADER>", "<HEADER_SEP>", "<HEADER_END>", "<ROW>", "<ROW_SEP>", "<ROW_END>", "<BOT>", "<EOT>"],
    },
    "input_modules": {
      "module_list":[
        {"type": "QuestionInput",  "option": "default", 
                  "separation_tokens": {"start": "", "end": ""}},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessColBERTQuestionInputTokenization", "option": "default"},
      ],
    },
    "decoder_input_modules": {
      "module_list":[
        {"type": "TextBasedTableInput",  "option": "default", "add_title": 1,
                  "separation_tokens": {"header_start": "<HEADER>", "header_sep": "<HEADER_SEP>", "header_end": "<HEADER_END>", "row_start": "<ROW>", "row_sep": "<ROW_SEP>", "row_end": "<ROW_END>", "title_start": "<BOT>", "title_end": "EOT"}
                  //"separation_tokens": {"header_start": "[SEP]", "header_sep": "[SEP]", "header_end": "", "row_start": "[SEP]", "row_sep": "[SEP]", "row_end": "", "title_start": "", "title_end": ""}
                  },
      ],
      "postprocess_module_list": [
        {"type": "PostProcessColBERTItemInputTokenization", "option": "default"},
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
      'max_source_length':32,
      'max_decoder_source_length': 512,
      'max_target_length':128,
    },
    "dataset_modules": {
      "module_list": [
        "LoadNQTableData",
        "LoadDataLoaders",
      ],
      "module_dict":{
        "LoadNQTableData": {
          "type": "LoadNQTableData", "option": "default",
          "config": {
            "preprocess": [],
            "table_path": {
              "tables": "TableQA_data/nq_tables/tables/tables.jsonl",
            },
            "bm25_results": "TableQA_data/nq_tables/nq_table_bm25_results.json",
            "data_path": {
              "train": "TableQA_data/nq_tables/interactions/train.jsonl",
              "validation": "TableQA_data/nq_tables/interactions/dev.jsonl",
              "test": "TableQA_data/nq_tables/interactions/test.jsonl",
            },
            "path": {
              "tables": "TableQA_data/nq_tables/preprocessed_tables.arrow",
              "train": "TableQA_data/nq_tables/preprocessed_train.arrow",
              "validation": "TableQA_data/nq_tables/preprocessed_validation.arrow",
              "test": "TableQA_data/nq_tables/preprocessed_test.arrow",
            },
          },
        },
        "LoadDataLoaders": {
          "type": "LoadDataLoaders", "option": "default",
          "config": {
            "train": [
                {
                    "dataset_type": "DPRNQTablesDataset",
                    "split": "train",
                    "use_column": "nq_tables_data",
                },
            ],
            "valid": [
                {
                    "dataset_type": "DPRNQTablesDataset",
                    "split": "validation",
                    "use_column": "nq_tables_data",
                },
                {
                    "dataset_type": "DPRNQTablesDataset",
                    "split": "test",
                    "use_column": "nq_tables_data",
                },
            ],
            "test": [
                {
                    "dataset_type": "DPRNQTablesDataset",
                    "split": "validation",
                    "use_column": "nq_tables_data",
                },
                {
                    "dataset_type": "DPRNQTablesDataset",
                    "split": "test",
                    "use_column": "nq_tables_data",
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
    "type": "ColBERTExecutor",
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
        "save_top_k_metric": "valid/DPRNQTablesDataset.validation/recall_at_5",
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
    {'name': 'compute_TQA_DPR_scores'},
  ],
};

std.mergePatch(base_env, override)
