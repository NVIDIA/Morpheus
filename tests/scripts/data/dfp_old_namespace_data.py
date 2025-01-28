#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
'''
This script is used as test input for morpheus_namespace_update.py script.
'''

# Disable all checkers
# flake8: noqa
# isort: skip_file
# yapf: disable
# pylint: skip-file

# old DFP import patterns
from dfp.utils.config_generator import ConfigGenerator
from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage
from morpheus_dfp.stages.dfp_split_users_stage import DFPSplitUsersStage  # no update
import dfp.stages.dfp_training
import dfp.stages.dfp_inference_stage as inference_stage
import dfp


def empty_imports_function_scope():
    '''
    Empty imports from morpheus_dfp, llm and vdb.
    '''
    from dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage
    from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
    from dfp.utils.regex_utils import iso_date_regex
    from morpheus_dfp.utils.schema_utils import SchemaBuilder  # no update
    from dfp.modules import dfp_data_prep
