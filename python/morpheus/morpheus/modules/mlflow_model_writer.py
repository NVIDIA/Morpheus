# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import logging

import mrc
from mrc.core import operators as ops

from morpheus.controllers.mlflow_model_writer_controller import MLFlowModelWriterController
from morpheus.utils.module_ids import MLFLOW_MODEL_WRITER
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(MLFLOW_MODEL_WRITER, MORPHEUS_MODULE_NAMESPACE)
def mlflow_model_writer(builder: mrc.Builder):
    """
    This module uploads trained models to the mlflow server.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - conda_env (str): Conda environment for the model; Example: `path/to/conda_env.yml`; Default: `[Required]`
            - databricks_permissions (dict): Permissions for the model; See Below; Default: None
            - experiment_name_formatter (str): Formatter for the experiment name;
                Example: `experiment_name_{timestamp}`; Default: `[Required]`
            - model_name_formatter (str): Formatter for the model name; Example: `model_name_{timestamp}`;
                Default: `[Required]`
            - timestamp_column_name (str): Name of the timestamp column; Example: `timestamp`; Default: timestamp
            - timeout (float): Timeout for get requests.

        databricks_permissions:
            - read (array): List of users with read permissions; Example: `["read_user1", "read_user2"]`; Default: -
            - write (array): List of users with write permissions; Example: `["write_user1", "write_user2"]`; Default: -
    """

    config = builder.get_current_module_config()

    timeout = config.get("timeout", 1.0)
    timestamp_column_name = config.get("timestamp_column_name", "timestamp")

    if ("model_name_formatter" not in config):
        raise ValueError("Model name formatter is required")

    if ("experiment_name_formatter" not in config):
        raise ValueError("Experiment name formatter is required")

    if ("conda_env" not in config):
        raise ValueError("Conda environment is required")

    model_name_formatter = config["model_name_formatter"]
    experiment_name_formatter = config["experiment_name_formatter"]
    conda_env = config.get("conda_env", None)

    databricks_permissions = config.get("databricks_permissions", None)

    controller = MLFlowModelWriterController(model_name_formatter=model_name_formatter,
                                             experiment_name_formatter=experiment_name_formatter,
                                             databricks_permissions=databricks_permissions,
                                             conda_env=conda_env,
                                             timeout=timeout,
                                             timestamp_column_name=timestamp_column_name)

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(controller.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

    node = builder.make_node(MLFLOW_MODEL_WRITER, mrc.core.operators.build(node_fn))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
