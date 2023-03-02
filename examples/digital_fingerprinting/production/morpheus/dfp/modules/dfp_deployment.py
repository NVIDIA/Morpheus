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

import logging

import dfp.modules.dfp_inference_pipe  # noqa: F401
import dfp.modules.dfp_training_pipe  # noqa: F401
import mrc
from mrc.core.node import Broadcast

import morpheus.loaders.fsspec_loader
from morpheus.utils.loader_ids import FSSPEC_LOADER
from morpheus.utils.module_ids import DATA_LOADER
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_config_with_overrides
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_DEPLOYMENT
from ..utils.module_ids import DFP_INFERENCE_PIPE
from ..utils.module_ids import DFP_TRAINING_PIPE

logger = logging.getLogger("morpheus.{}".format(__name__))


@register_module(DFP_DEPLOYMENT, MODULE_NAMESPACE)
def dfp_deployment(builder: mrc.Builder):
    module_config = get_module_config(DFP_DEPLOYMENT, builder)

    fsspec_dataloader_conf = get_config_with_overrides(module_config, FSSPEC_LOADER, "fsspec_dataloader")
    fsspec_dataloader_conf["module_id"] = DATA_LOADER  # Work around some naming issues.

    dfp_training_pipe_conf = get_config_with_overrides(module_config, DFP_TRAINING_PIPE, "dfp_training_pipe")
    dfp_inference_pipe_conf = get_config_with_overrides(module_config, DFP_INFERENCE_PIPE, "dfp_inference_pipe")

    if "output_port_count" not in module_config:
        raise KeyError("Missing required configuration 'output_port_count'")

    output_port_count = module_config.get("output_port_count")

    fsspec_dataloader_module = load_module(fsspec_dataloader_conf, builder=builder)

    # Load module from registry.
    dfp_training_pipe_module = load_module(dfp_training_pipe_conf, builder=builder)
    dfp_inference_pipe_module = load_module(dfp_inference_pipe_conf, builder=builder)

    # Create broadcast node to fork the pipeline.
    boradcast = Broadcast(builder, "broadcast")

    # Make an edge between modules
    builder.make_edge(fsspec_dataloader_module.output_port("output"), boradcast)
    builder.make_edge(boradcast, dfp_training_pipe_module.input_port("input"))
    builder.make_edge(boradcast, dfp_inference_pipe_module.input_port("input"))

    out_streams = [dfp_training_pipe_module.output_port("output"), dfp_inference_pipe_module.output_port("output")]

    # Register input port for a module.
    builder.register_module_input("input", fsspec_dataloader_module.input_port("input"))

    # Register output ports for a module.
    for i in range(output_port_count):
        # Output ports are registered in increment order.
        builder.register_module_output(f"output-{i}", out_streams[i])
