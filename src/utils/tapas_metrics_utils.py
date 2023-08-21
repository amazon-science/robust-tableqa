# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Denotation accuracy calculation for TAPAS predictions over WikiSQL."""

import math
import os
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Mapping

from absl import logging
import dataclasses
import pandas as pd
import sklearn.metrics
import utils.text_utils as text_utils


def _collect_cells_from_table(cell_coos,
                              table):
  cell_values = []
  for cell in cell_coos:
    value = str(table.iat[cell[1], cell[0]])
    cell_values.append(value)
  return cell_values


def _safe_convert_to_float(value):
  float_value = text_utils.convert_to_float(value)
  if math.isnan(float_value):
    raise ValueError('Value is NaN %s' % value)
  return float_value


def _parse_value(value):
  """Parses a cell value to a number or lowercased string."""
  try:
    return _safe_convert_to_float(value)
  except ValueError:
    try:
      return value.lower()
    except ValueError:
      return value


def _to_float32s(elements):
  return tuple(text_utils.to_float32(v) for v in elements)


