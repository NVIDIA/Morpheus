# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

from morpheus.config import ConfigOnnxToTRT

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
except ImportError as e:
    raise ImportError(
        "The ONNX to TensorRT conversion utility requires additional packages, which can be installed using the "
        "following command:\n"
        "conda env update --solver=libmamba -n morpheus --file "
        "conda/environments/model-utils_cuda-125_arch-$(arch).yaml") from e


# pylint: disable=no-member
def gen_engine(config: ConfigOnnxToTRT):
    """
    This class converts an Onnx model to a TRT model.

    Parameters
    ----------
    config : `morpheus.config.ConfigOnnxToTRT`
        ONNX to TRT generator configuration.

    """

    # Local imports to avoid requiring TensorRT to generate the docs

    trt_logger = trt.Logger()

    input_model = config.input_model

    logger.info("Loading ONNX file: '%s'", input_model)

    # Otherwise we are creating a new model
    explicit_branch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(trt_logger) as builder, \
        builder.create_network(explicit_branch) as network, \
            trt.OnnxParser(network, trt_logger) as parser:

        with open(input_model, "rb") as model_file:
            if (not parser.parse(model_file.read())):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError("Count not parse Onnx file. See log.")

        # Now we need to build and serialize the model
        with builder.create_builder_config() as builder_config:

            builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                                 config.max_workspace_size * (1024 * 1024))
            builder_config.set_flag(trt.BuilderFlag.FP16)

            # Create the optimization files
            for min_batch, max_batch in config.batches:
                profile = builder.create_optimization_profile()

                min_shape = (min_batch, config.seq_length)
                shape = (max_batch, config.seq_length)

                for i in range(network.num_inputs):
                    in_tensor = network.get_input(i)
                    profile.set_shape(in_tensor.name, min=min_shape, opt=shape, max=shape)

                builder_config.add_optimization_profile(profile)

            # Actually build the engine
            print("Building engine. This may take a while...")
            serialized_engine = builder.build_serialized_network(network, builder_config)

            # Now save a copy to prevent building next time
            print("Writing engine to: {config.output_model}")

            with open(config.output_model, "wb") as f:
                f.write(serialized_engine)

            print("Complete!")
