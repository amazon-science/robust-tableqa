// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import '../base_env.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 1000;
local save_interval = 1000;
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
    "base_model": "TAPEX",
    "ModelClass": "BartForConditionalGeneration",
    "TokenizerClass": "TapexTokenizer",
    "TokenizerModelVersion": "microsoft/tapex-large",
    "ConfigClass": "BartConfig",
    "ModelVersion": "microsoft/tapex-large",
    "pretrained": 1,
    "modules": [
    ],
    "num_beams": 5,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
    "input_modules": {
      "module_list":[
        {"type": "QuestionInput",  "option": "default", 
                  "separation_tokens": {"start": "", "end": ""}},
        {"type": "TableInput",  "option": "default",
                  "separation_tokens": {}},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessTAPEXInputTokenization", "option": "default"},
      ],
    },
    "decoder_input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "output_modules": {
      "module_list":[
        {"type": "FlattenedAnswerOutput", "option": "default"},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessTAPEXOutputTokenization", "option": "default"},
      ],
    },
  },
  "cache":{
    "regenerate":{},
  },
  "data_loader": {
    "type": "DataLoaderForTableQA",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':1024,
      'max_decoder_source_length': 1024,
      'max_target_length':128,
    },
    "dataset_modules": {
      "module_list": [
        "LoadWikiTQData",
        "LoadDataLoaders",
      ],
      "module_dict":{
        "LoadDataLoaders": {
          "type": "LoadDataLoaders", "option": "default",
          "config": {
          "train": [
              {
                  "dataset_type": "WikiTQDataset",
                  "split": "train",
                  "use_column": "wtq_data",
              },
          ],
          "valid": [
              {
                  "dataset_type": "WikiTQDataset",
                  "split": "validation",
                  "use_column": "wtq_data",
              },
              {
                  "dataset_type": "WikiTQDataset",
                  "split": "test",
                  "use_column": "wtq_data",
              },
            ],
            "test": [
                {
                    "dataset_type": "WikiTQDataset",
                    "split": "validation",
                    "use_column": "wtq_data",
                },
                {
                    "dataset_type": "WikiTQDataset",
                    "split": "test",
                    "use_column": "wtq_data",
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
    "type": "TAPEXExecutor",
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
        "save_top_k_metric": "valid/WikiTQDataset.validation/denotation_accuracy",
        "weight_decay": 0.01,
        "label_smoothing_factor": 0.1,
    },
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
    {'name': 'compute_denotation_accuracy'},
  ],
};

std.mergePatch(base_env, override)
