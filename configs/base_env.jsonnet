// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

// SPDX-License-Identifier: CC-BY-NC-4.0


// This is the base environment file
// It serves as default values for all other jsonnet config files
// Please override these values directly in corresponding config files


// Default values for training control
local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;
local early_stop_patience = 3;

local seed=2021;

// data path configuration
local wandb_cache_dir = '';
local wandb_dir = '';
local default_cache_folder = 'cache';


{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "",
  "TENSORBOARD_FOLDER": "",
  "WANDB": {
    "CACHE_DIR":  wandb_cache_dir,
    "DIR": wandb_dir,
    "entity": "",
    "project": "TableQA_publication",
    "tags": [],
  },
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "TAPAS",
    "pretrained": 1,
    "modules": [],
    "input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "rag_modules": {
      "module_list":[],
    },
    "decoder_input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "output_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
  },
  "cache":{
    "default_folder": default_cache_folder,
    "regenerate":{},
  },
  "data_loader": {
    "type": "DataLoaderForTableQA",
    "datasets": {},
    "dummy_dataloader": 0,
    "additional":{},
    "dataset_modules": {
      "module_list": [],
      "module_dict":{   // all available modules
        "LoadWikiTQData": {
          "type": "LoadWikiTQData", "option": "default",
          "config": {
            "preprocess": [],
            "path": {
              "train": "TableQA_data/wtq/preprocessed_train.arrow",
              "validation": "TableQA_data/wtq/preprocessed_validation.arrow",
              "test": "TableQA_data/wtq/preprocessed_test.arrow",
            }
          },
        },
        "LoadWikiSQLData": {
          "type": "LoadWikiSQLData", "option": "default",
          "config": {
            "preprocess": [],
            "path": {
              "train": "TableQA_data/wikisql/preprocessed_train.arrow",
              "validation": "TableQA_data/wikisql/preprocessed_validation.arrow",
              "test": "TableQA_data/wikisql/preprocessed_test.arrow",
            }
          },
        },
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "TAPASExecutor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch":-1,
    "save_interval":save_interval,
    "load_model_path": "",
    "scheduler": "none",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
        "plugins": [],
        "save_top_k": -1,
        "save_top_k_metric": "valid/denotation_accuracy",
        "save_top_k_mode": "max",
        "early_stop_patience": early_stop_patience,
    }
  },
  "valid": {
    "batch_size":valid_batch_size,
    "step_size":valid_step_size,
    "additional": {
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "load_model_path": "",
    "additional": {
        "multiprocessing": 4,
    },
  }
}
