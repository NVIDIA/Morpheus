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

# old LLM import patterns
from morpheus.llm import LLMContext
from morpheus.llm.services.llm_service import LLMService

# old vdb import patterns
from morpheus.service.vdb import faiss_vdb_service
from morpheus.service import vdb
from morpheus.modules.output import write_to_vector_db
from morpheus.modules.output.write_to_vector_db import preprocess_vdb_resources
import morpheus.service.vdb

# These should be skipped
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus_llm.service.vdb import milvus_client  # no update
from morpheus_llm.llm import LLMEngine  # no update


def empty_imports_function_scope():
    '''
    Empty imports from llm and vdb.
    '''
    from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
    from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
    import morpheus.modules.schemas.write_to_vector_db_schema
