// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import 'tapex_ITR_wikisql.jsonnet';


local override = {
  "model_config": {
    "ModelClass": "ITRRagReduceModel",
  },
};

std.mergePatch(base_env, override)
