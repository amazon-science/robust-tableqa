// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import 'frozen_rag.jsonnet';


local override = {
  "model_config": {
    "loss_ratio": {
      "nll_loss": 1,
      "rag_loss": 0,
      "additional_loss": 1,
    },
    "RAVQA_loss_type": "Approach5",
    "modules": [
    ],
    "Ks": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100],
    "num_beams": 5,
  },
  "train": {
    "retriever_lr": 0.00001,
  }
};

std.mergePatch(base_env, override)
