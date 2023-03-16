# Copyright (c) 2023, NVIDIA CORPORATION.
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

import os

from dfp.utils.derive_args import DeriveArgs
from dfp.utils.derive_args import pyobj2str
from dfp.utils.module_ids import DFP_DATA_PREP
from dfp.utils.module_ids import DFP_INFERENCE
from dfp.utils.module_ids import DFP_INFERENCE_PIPELINE
from dfp.utils.module_ids import DFP_POST_PROCESSING
from dfp.utils.module_ids import DFP_ROLLING_WINDOW
from dfp.utils.module_ids import DFP_SPLIT_USERS
from dfp.utils.module_ids import DFP_TRAINING
from dfp.utils.module_ids import DFP_TRAINING_PIPELINE
from dfp.utils.regex_utils import iso_date_regex_pattern
from dfp.utils.schema_utils import Schema

from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import load_labels_file
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.messages.multi_message import MultiMessage
from morpheus.utils.module_ids import FILE_BATCHER
from morpheus.utils.module_ids import FILE_TO_DF
from morpheus.utils.module_ids import FILTER_DETECTIONS
from morpheus.utils.module_ids import MLFLOW_MODEL_WRITER
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_ids import SERIALIZE
from morpheus.utils.module_ids import WRITE_TO_FILE


class ConfigGenerator:

    def __init__(self, config: Config, derive_args: DeriveArgs, schema: Schema, encoding: str = "latin1"):

        self._config = config
        self._derive_args = derive_args
        self._encoding = encoding
        self._source_schema_str = pyobj2str(schema.source, encoding=encoding)
        self._preprocess_schema_str = pyobj2str(schema.preprocess, encoding=encoding)
        self._input_message_type = pyobj2str(MultiMessage, encoding)

    def inf_pipe_module_conf(self):

        module_conf = {
            "module_id": DFP_INFERENCE_PIPELINE,
            "module_name": "dfp_inference_pipeline",
            "namespace": MODULE_NAMESPACE,
            FILE_BATCHER: {
                "module_id": FILE_BATCHER,
                "module_name": "file_batcher",
                "namespace": MODULE_NAMESPACE,
                "period": "D",
                "sampling_rate_s": self._derive_args.sample_rate_s,
                "start_time": self._derive_args.start_time,
                "end_time": self._derive_args.end_time,
                "iso_date_regex_pattern": iso_date_regex_pattern
            },
            FILE_TO_DF: {
                "module_id": FILE_TO_DF,
                "module_name": "FILE_TO_DF",
                "namespace": MODULE_NAMESPACE,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "parser_kwargs": {
                    "lines": False, "orient": "records"
                },
                "cache_dir": self._derive_args.cache_dir,
                "filter_null": True,
                "file_type": "JSON",
                "schema": {
                    "schema_str": self._source_schema_str, "encoding": self._encoding
                }
            },
            DFP_SPLIT_USERS: {
                "module_id": DFP_SPLIT_USERS,
                "module_name": "dfp_split_users",
                "namespace": MODULE_NAMESPACE,
                "include_generic": self._derive_args.include_generic,
                "include_individual": self._derive_args.include_individual,
                "skip_users": self._derive_args.skip_users,
                "only_users": self._derive_args.only_users,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "userid_column_name": self._config.ae.userid_column_name,
                "fallback_username": self._config.ae.fallback_username
            },
            DFP_ROLLING_WINDOW: {
                "module_id": DFP_ROLLING_WINDOW,
                "module_name": "dfp_rolling_window",
                "namespace": MODULE_NAMESPACE,
                "min_history": 1,
                "min_increment": 0,
                "max_history": self._derive_args.duration,
                "cache_dir": self._derive_args.cache_dir,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            DFP_DATA_PREP: {
                "module_id": DFP_DATA_PREP,
                "module_name": "dfp_data_prep",
                "namespace": MODULE_NAMESPACE,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "schema": {
                    "schema_str": self._preprocess_schema_str, "encoding": self._encoding
                }
            },
            DFP_INFERENCE: {
                "module_id": DFP_INFERENCE,
                "module_name": "dfp_inference",
                "namespace": MODULE_NAMESPACE,
                "model_name_formatter": self._derive_args.model_name_formatter,
                "fallback_username": self._config.ae.fallback_username,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            FILTER_DETECTIONS: {
                "module_id": FILTER_DETECTIONS,
                "module_name": "filter_detections",
                "namespace": MODULE_NAMESPACE,
                "field_name": "mean_abs_z",
                "threshold": 2.0,
                "filter_source": "DATAFRAME",
                "schema": {
                    "input_message_type": self._input_message_type, "encoding": self._encoding
                }
            },
            DFP_POST_PROCESSING: {
                "module_id": DFP_POST_PROCESSING,
                "module_name": "dfp_post_processing",
                "namespace": MODULE_NAMESPACE,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            SERIALIZE: {
                "module_id": SERIALIZE,
                "module_name": "serialize",
                "namespace": MODULE_NAMESPACE,
                "exclude": ['batch_count', 'origin_hash', '_row_hash', '_batch_id']
            },
            WRITE_TO_FILE: {
                "module_id": WRITE_TO_FILE,
                "module_name": "write_to_file",
                "namespace": MODULE_NAMESPACE,
                "filename": "dfp_detections_{}.csv".format(self._derive_args.source),
                "overwrite": True
            }
        }

        return module_conf

    def tra_pipe_module_conf(self):
        module_conf = {
            "module_id": DFP_TRAINING_PIPELINE,
            "module_name": "dfp_training_pipeline",
            "namespace": MODULE_NAMESPACE,
            FILE_BATCHER: {
                "module_id": FILE_BATCHER,
                "module_name": "file_batcher",
                "namespace": MODULE_NAMESPACE,
                "period": "D",
                "sampling_rate_s": self._derive_args.sample_rate_s,
                "start_time": self._derive_args.start_time,
                "end_time": self._derive_args.end_time,
                "iso_date_regex_pattern": iso_date_regex_pattern
            },
            FILE_TO_DF: {
                "module_id": FILE_TO_DF,
                "module_name": "FILE_TO_DF",
                "namespace": MODULE_NAMESPACE,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "parser_kwargs": {
                    "lines": False, "orient": "records"
                },
                "cache_dir": self._derive_args.cache_dir,
                "filter_null": True,
                "file_type": "JSON",
                "schema": {
                    "schema_str": self._source_schema_str, "encoding": self._encoding
                }
            },
            DFP_SPLIT_USERS: {
                "module_id": DFP_SPLIT_USERS,
                "module_name": "dfp_split_users",
                "namespace": MODULE_NAMESPACE,
                "include_generic": self._derive_args.include_generic,
                "include_individual": self._derive_args.include_individual,
                "skip_users": self._derive_args.skip_users,
                "only_users": self._derive_args.only_users,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "userid_column_name": self._config.ae.userid_column_name,
                "fallback_username": self._config.ae.fallback_username
            },
            DFP_ROLLING_WINDOW: {
                "module_id": DFP_ROLLING_WINDOW,
                "module_name": "dfp_rolling_window",
                "namespace": MODULE_NAMESPACE,
                "min_history": 300,
                "min_increment": 300,
                "max_history": self._derive_args.duration,
                "cache_dir": self._derive_args.cache_dir,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            DFP_DATA_PREP: {
                "module_id": DFP_DATA_PREP,
                "module_name": "dfp_data_prep",
                "namespace": MODULE_NAMESPACE,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "schema": {
                    "schema_str": self._preprocess_schema_str, "encoding": self._encoding
                }
            },
            DFP_TRAINING: {
                "module_id": DFP_TRAINING,
                "module_name": "dfp_training",
                "namespace": MODULE_NAMESPACE,
                "model_kwargs": {
                    "encoder_layers": [512, 500],  # layers of the encoding part
                    "decoder_layers": [512],  # layers of the decoding part
                    "activation": 'relu',  # activation function
                    "swap_p": 0.2,  # noise parameter
                    "lr": 0.001,  # learning rate
                    "lr_decay": 0.99,  # learning decay
                    "batch_size": 512,
                    "verbose": False,
                    "optimizer": 'sgd',  # SGD optimizer is selected(Stochastic gradient descent)
                    "scaler": 'standard',  # feature scaling method
                    "min_cats": 1,  # cut off for minority categories
                    "progress_bar": False,
                    "device": "cuda"
                },
                "feature_columns": self._config.ae.feature_columns,
                "epochs": 30,
                "validation_size": 0.10
            },
            MLFLOW_MODEL_WRITER: {
                "module_id": MLFLOW_MODEL_WRITER,
                "module_name": "mlflow_model_writer",
                "namespace": MODULE_NAMESPACE,
                "model_name_formatter": self._derive_args.model_name_formatter,
                "experiment_name_formatter": self._derive_args.experiment_name_formatter,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "conda_env": {
                    'channels': ['defaults', 'conda-forge'],
                    'dependencies': ['python={}'.format('3.8'), 'pip'],
                    'pip': ['mlflow', 'dfencoder'],
                    'name': 'mlflow-env'
                },
                "databricks_permissions": None
            }
        }

        return module_conf


def generate_ae_config(labels_file: str,
                       userid_column_name: str,
                       timestamp_column_name: str,
                       use_cpp: bool = False,
                       num_threads: int = os.cpu_count()):

    config = Config()

    CppConfig.set_should_use_cpp(use_cpp)

    config.num_threads = num_threads

    config.ae = ConfigAutoEncoder()

    config.ae.feature_columns = load_labels_file(get_package_relative_file(labels_file))
    config.ae.userid_column_name = userid_column_name
    config.ae.timestamp_column_name = timestamp_column_name

    return config
