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
This script is used to update imports related to DFP and LLM morpheus modules.
Usage:
    python morpheus_namespace_update.py --directory <directory> --dfp
    python morpheus_namespace_update.py --directory <directory> --llm
'''
import os
import re

import click


def replace_imports_in_file(file_path, old_module, new_module):
    '''
    Simple module replacement function.
    '''
    do_write = False
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Take care of old imports of style "import old_module.stages.dfp_inference_stage as ..."
        if re.findall(rf"(import {old_module})(\W+)", content):
            do_write = True
            content = re.sub(rf"(import {old_module})(\W+)", rf"import {new_module}\2", content)

        # Take care of old imports of style "from old_module.stages.dfp_inference_stage import ..."
        if re.findall(rf"(from {old_module})(\S+)", content):
            do_write = True
            content = re.sub(rf"(from {old_module})(\S+)", rf"from {new_module}\2", content)

    if do_write:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def replace_llm_imports_in_file(file_path):
    '''
    LLM module replacement requires special handling.
    '''
    do_write = False
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # simple replace
    pat = "import morpheus.llm"
    if re.findall(pat, content):
        do_write = True
        content = re.sub(pat, "import morpheus_llm", content)

    # Take care of old imports of style "from morpheus.llm import ..." and
    # "from morpheus.llm.services.llm_service import ..."
    module = "llm"
    if re.findall(rf"(from morpheus\.)({module})", content):
        do_write = True
        content = re.sub(rf"(from morpheus\.)({module})", r"from morpheus_llm.\2", content)

    # Take care of old imports of style "from morpheus.stages.llm.llm_engine_stage import ..."
    module = "llm"
    if re.findall(rf"(from morpheus\.)(\w+)(\.{module})", content):
        do_write = True
        content = re.sub(rf"(from morpheus\.)(\w+)(\.{module})", r"from morpheus_llm.\2\3", content)

    # Take care of old imports of style "from morpheus.service.vdb import faiss_vdb_service"
    module = "vdb"
    if re.findall(rf"(from morpheus\.)(\w+)(\.{module})", content):
        do_write = True
        content = re.sub(rf"(from morpheus\.)(\w+)(\.{module})", r"from morpheus_llm.\2\3", content)

    # Take care of old imports of style "from morpheus.service import vdb" and "import morpheus.service.vdb"
    old_pat = "from morpheus.service import vdb"
    new_pat = "from morpheus_llm.service import vdb"
    if re.findall(old_pat, content):
        do_write = True
        content = re.sub(old_pat, new_pat, content)
    old_pat = "import morpheus.service.vdb"
    new_pat = "import morpheus_llm.service.vdb"
    if re.findall(old_pat, content):
        do_write = True
        content = re.sub(old_pat, new_pat, content)

    # Take care of old imports of style -
    # "from morpheus.modules.output.write_to_vector_db import preprocess_vdb_resources"
    # "from morpheus.stages.write_to_vector_db_stage import WriteToVectorDBStage"
    module = "write_to_vector_db"
    if re.findall(rf"(from morpheus\.)(\S+)(\.{module})", content):
        do_write = True
        content = re.sub(rf"(from morpheus\.)(\S+)(\.{module})", r"from morpheus_llm.\2\3", content)

    # Take care of old imports of style "from morpheus.modules.output import write_to_vector_db"
    if re.findall(rf"(from morpheus\.)(\S+)( import {module})", content):
        do_write = True
        content = re.sub(rf"(from morpheus\.)(\S+)( import {module})", r"from morpheus_llm.\2\3", content)

    # Take care of old imports of style "import morpheus.modules.output.write_to_vector_db_schema"
    if re.findall(rf"(import morpheus\.)(\S+)(\.{module})", content):
        do_write = True
        content = re.sub(rf"(import morpheus\.)(\S+)(\.{module})", r"import morpheus_llm.\2\3", content)

    if do_write:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


@click.command()
@click.option('--directory', default='./', help='directory for updating')
@click.option('--dfp', is_flag=True, help='Replace dfp imports')
@click.option('--llm', is_flag=True, help='Replace llm and vdb imports')
def replace_imports(directory, dfp, llm):
    '''
    Walk files in the given directory and replace imports.
    '''
    if not llm and not dfp:
        print("Please provide either --dfp or --llm")
        return

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip this script
            if os.path.abspath(file_path) == os.path.abspath(__file__):
                continue
            if file.endswith(".py"):
                if dfp:
                    replace_imports_in_file(file_path, 'dfp', 'morpheus_dfp')
                if llm:
                    replace_llm_imports_in_file(file_path)


if __name__ == "__main__":
    replace_imports()
