# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from datetime import datetime

import pytest

from morpheus.config import Config


@pytest.fixture(name="dfp_arg_parser")
def dfp_arg_parser_fixture():
    from morpheus_dfp.utils.dfp_arg_parser import DFPArgParser
    dfp_arg_parser = DFPArgParser(skip_user=["unittest-skip-user"],
                                  only_user=["unittest-only-user"],
                                  start_time=datetime(1993, 4, 5, 6, 7, 8),
                                  log_level=logging.DEBUG,
                                  cache_dir=".cache",
                                  sample_rate_s="20",
                                  duration="2days",
                                  source="unittest",
                                  tracking_uri="http://unittest",
                                  silence_monitors=False,
                                  mlflow_experiment_name_formatter="unittest-experiment",
                                  mlflow_model_name_formatter="unittest-model",
                                  train_users="unittest-train-users")
    dfp_arg_parser.init()
    yield dfp_arg_parser


@pytest.fixture(name="schema")
def schema_fixture(config: Config):
    from morpheus_dfp.utils.schema_utils import SchemaBuilder
    schema_builder = SchemaBuilder(config, "duo")
    yield schema_builder.build_schema()


def test_constructor(config: Config, dfp_arg_parser: "DFPArgParser", schema: "Schema"):  # noqa: F821
    from morpheus_dfp.utils.config_generator import ConfigGenerator

    config_generator = ConfigGenerator(config=config, dfp_arg_parser=dfp_arg_parser, schema=schema, encoding="latin1")

    assert config_generator._config is config
    assert config_generator._dfp_arg_parser is dfp_arg_parser
    assert config_generator._encoding == "latin1"
    assert config_generator._start_time_str == "1993-04-05T06:07:08+00:00"
    assert config_generator._end_time_str == "1993-04-07T06:07:08+00:00"
