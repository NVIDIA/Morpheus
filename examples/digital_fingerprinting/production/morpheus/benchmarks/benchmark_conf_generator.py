# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Benchmark config generator for DFP."""

import glob
import json
import logging
import typing
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from os import path

import mlflow
import pandas as pd

from morpheus_dfp.utils.config_generator import ConfigGenerator
from morpheus_dfp.utils.config_generator import generate_ae_config
from morpheus_dfp.utils.dfp_arg_parser import DFPArgParser
from morpheus_dfp.utils.schema_utils import SchemaBuilder

logger = logging.getLogger(f"morpheus.{__name__}")

THIS_DIR = path.dirname(path.abspath(__file__))


def set_mlflow_tracking_uri(tracking_uri):
    """Set mlflow tracking uri and logger level."""
    mlflow.set_tracking_uri(tracking_uri)
    logging.getLogger('mlflow').setLevel(logger.level)


def load_json(filepath: str):
    """Load a json file."""
    full_filepath = path.join(THIS_DIR, filepath)
    with open(full_filepath, 'r', encoding='UTF-8') as (f):
        json_dict = json.load(f)
    return json_dict


class BenchmarkConfGenerator:
    """
    Benchmark config generator for DFP.

    Parameters
    ----------
    pipe_conf : typing.Dict[str, any]
        Pipeline configuration.
    """

    def __init__(self, pipe_conf: typing.Dict[(str, any)]):
        self._pipe_conf = pipe_conf
        self._config = self._create_config()

    @property
    def pipe_config(self):
        """Get the pipe_config."""
        return self._config

    @property
    def source(self):
        """Get the source from the pipe_config."""
        return self._pipe_conf.get('source')

    def _get_model_name_formatter(self) -> str:
        model_name_formatter = f'DFP-{self.source}-' + '{user_id}'
        return model_name_formatter

    def _get_experiment_name_formatter(self) -> str:
        experiment_name_formatter = f'dfp/{self.source}/training/' + '{reg_model_name}'
        return experiment_name_formatter

    def _get_start_stop_time(self) -> typing.Tuple[(datetime, datetime)]:
        start_time = self._pipe_conf.get('start_time')
        start_time = datetime.strptime(start_time, '%Y-%m-%d')
        duration = self._pipe_conf.get('duration')
        duration = timedelta(seconds=(pd.Timedelta(duration).total_seconds()))
        if start_time is None:
            end_time = datetime.now(tz=(timezone.utc))
            start_time = end_time - duration
        else:
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=(timezone.utc))
            end_time = start_time + duration
        return tuple((start_time, end_time))

    def _create_config(self):
        config = generate_ae_config(source=(self._pipe_conf.get('source')),
                                    userid_column_name=(self._pipe_conf.get('userid_column_name')),
                                    timestamp_column_name=(self._pipe_conf.get('timestamp_column_name')),
                                    pipeline_batch_size=(self._pipe_conf.get('pipeline_batch_size')),
                                    edge_buffer_size=(self._pipe_conf.get('edge_buffer_size')),
                                    num_threads=(self._pipe_conf.get('num_threads')))
        return config

    def get_stages_conf(self) -> typing.Dict[(str, any)]:
        """Get the stages config."""
        stages_conf = {}
        start_stop_time = self._get_start_stop_time()
        stages_conf['start_time'] = start_stop_time[0]
        stages_conf['end_time'] = start_stop_time[1]
        stages_conf['duration'] = self._pipe_conf.get('duration')
        stages_conf['sampling_rate_s'] = 0
        stages_conf['cache_dir'] = './.cache/dfp'
        stages_conf['include_generic'] = True
        stages_conf['include_individual'] = False
        stages_conf['skip_users'] = []
        stages_conf['only_users'] = []
        stages_conf['model_name_formatter'] = self._get_model_name_formatter()
        stages_conf['experiment_name_formatter'] = self._get_experiment_name_formatter()
        return stages_conf

    def get_filenames(self) -> typing.List[str]:
        """Get the filenames from the config."""
        if 'glob_path' in self._pipe_conf:
            input_glob = self._pipe_conf.get('glob_path')
            input_glob = path.join(THIS_DIR, input_glob)
            filenames = glob.glob(input_glob)
        else:
            if 'file_path' in self._pipe_conf:
                file_path = self._pipe_conf.get('file_path')
                full_file_path = path.join(THIS_DIR, file_path)
                filenames = [full_file_path]
            else:
                if 'message_path' in self._pipe_conf:
                    file_path = self._pipe_conf.get('message_path')
                    full_file_path = path.join(THIS_DIR, file_path)
                    filenames = [full_file_path]
                else:
                    raise KeyError('Configuration needs the glob path or file path attribute.')
        assert len(filenames) > 0
        return filenames

    def get_schema(self):
        """Get schema."""
        schema_builder = SchemaBuilder((self.pipe_config), source=(self.source))
        schema = schema_builder.build_schema()
        return schema

    def get_module_conf(self):
        """Get module configuration."""
        dfp_arg_parser = DFPArgParser(skip_user=[],
                                      only_user=[],
                                      start_time=(datetime.strptime(self._pipe_conf.get('start_time'), '%Y-%m-%d')),
                                      log_level=logger.level,
                                      cache_dir='./.cache/dfp',
                                      sample_rate_s=0,
                                      duration=(self._pipe_conf.get('duration')),
                                      source=(self.source),
                                      tracking_uri=mlflow.get_tracking_uri(),
                                      silence_monitors=True,
                                      mlflow_experiment_name_formatter=self._get_experiment_name_formatter(),
                                      mlflow_model_name_formatter=self._get_model_name_formatter(),
                                      train_users='generic')
        dfp_arg_parser.init()
        config_generator = ConfigGenerator(self.pipe_config, dfp_arg_parser, self.get_schema())
        module_conf = config_generator.get_module_conf()
        return module_conf
