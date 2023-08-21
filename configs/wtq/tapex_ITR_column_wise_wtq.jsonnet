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

// here we put the index file paths
local index_files = {
  "index_paths": {
    "train": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_single_column/test/wtq_original_sets/step_3801/test.ITRWikiTQDataset.train",
    "validation": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_single_column/test/wtq_original_sets/step_3801/test.ITRWikiTQDataset.validation",
    "test": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_single_column/test/wtq_original_sets/step_3801/test.ITRWikiTQDataset.test",
  },
};

local override = {
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "ITR_TAPEX",
    "ModelClass": "ITRRagModel",
    "TokenizerClass": "DPRQuestionEncoderTokenizer",  // question encoder tokenizer
    "TokenizerModelVersion": "facebook/dpr-question_encoder-single-nq-base", // question encoder tokenizer version
    "DecoderTokenizerClass": "TapexTokenizer",  // generator tokenizer
    "DecoderTokenizerModelVersion": "microsoft/tapex-large", // generator tokenizer version
    
    "QueryEncoderModelClass": "DPRQuestionEncoder", // question encoder
    "QueryEncoderConfigClass": "DPRConfig", // question encoder
    // "QueryEncoderModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    "QueryEncoderModelVersion": "$DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_single_column/train/saved_model/step_3801/query_encoder",
    

    "GeneratorModelClass": "BartForConditionalGeneration", // answer generator
    "GeneratorConfigClass": "BartConfig",
    "GeneratorModelVersion": "microsoft/tapex-large",
    "pretrained": 1,
    "loss_ratio": {
      "nll_loss": 1,
    },
    "modules": [
      "freeze_question_encoder",
    ],
    "supervised": 0,
    "Ks": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100],
    "num_beams": 5,
    "SPECIAL_TOKENS":{
      "additional_special_tokens": [],
    },
    "DECODER_SPECIAL_TOKENS":{
      "additional_special_tokens": [],
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
    "index_files": index_files,
  },
  "data_loader": {
    "type": "DataLoaderForTableQA",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':1024,
      'max_decoder_source_length': 1024,
      'max_target_length':128,
      'num_knowledge_passages': 5,
    },
    "dataset_modules": {
      "module_list": [
        "LoadWikiTQData",
        "LoadDataLoaders",
      ],
      "module_dict":{
        "LoadWikiTQData": {
          "type": "LoadWikiTQData", "option": "default",
          "config": {
            "preprocess": ["split_table_by_column_combination"],
            "path": {
              "train": "TableQA_data/wtq/preprocessed_split_table_by_column_combination_single_train.arrow",
              "validation": "TableQA_data/wtq/preprocessed_split_table_by_column_combination_single_validation.arrow",
              "test": "TableQA_data/wtq/preprocessed_split_table_by_column_combination_single_test.arrow",
            }
          },
        },
        "LoadDataLoaders": {
          "type": "LoadDataLoaders", "option": "default",
          "config": {
            "train": [
                {
                    "dataset_type": "ITRRAGWikiTQDataset",
                    "split": "train",
                    "use_column": "wtq_data",
                },
            ],
            "valid": [
                {
                    "dataset_type": "ITRRAGWikiTQDataset",
                    "split": "validation",
                    "use_column": "wtq_data",
                },
                {
                    "dataset_type": "ITRRAGWikiTQDataset",
                    "split": "test",
                    "use_column": "wtq_data",
                },
            ],
            "test": [
                {
                    "dataset_type": "ITRRAGWikiTQDataset",
                    "split": "validation",
                    "use_column": "wtq_data",
                },
                {
                    "dataset_type": "ITRRAGWikiTQDataset",
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
    "type": "ITRRAGExecutor",
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
        "save_top_k_metric": "valid/ITRRAGWikiTQDataset.validation/denotation_accuracy",
        "weight_decay": 0.01,
        "label_smoothing_factor": 0.1,
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
    {'name': 'compute_denotation_accuracy'},
    // {'name': 'compute_ITR_RAG_retrieval_results'},
  ],
};

std.mergePatch(base_env, override)
