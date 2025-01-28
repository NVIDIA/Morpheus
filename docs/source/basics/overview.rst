..
   SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Morpheus CLI Overview
=====================

This section focuses on the Morpheus CLI and illustrates how the CLI can be used to configure and run a Morpheus
Pipeline.

Organization
------------

The Morpheus CLI is built on the Click Python package which allows for nested commands and chaining multiple commands
together. At a high level, the CLI is broken up into two main sections:

 * ``run``
    * For running FIL, NLP or OTHER pipelines.
 * ``tools``
    * Tools/Utilities to help set up, configure and run pipelines and external resources.

Users can get help on any command by passing ``--help`` to a command or sub-command. For example, to get help on the
run:

.. code-block:: console

   $ morpheus run --help
   Usage: morpheus run [OPTIONS] COMMAND [ARGS]...

   Run subcommand, used for running a pipeline

   Options:
   --num_threads INTEGER RANGE     Number of internal pipeline threads to use  [default: 64; x>=1]
   --pipeline_batch_size INTEGER RANGE
                                    Internal batch size for the pipeline. Can be much larger than the model batch size. Also used for Kafka consumers  [default: 256; x>=1]
   --model_max_batch_size INTEGER RANGE
                                    Max batch size to use for the model  [default: 8; x>=1]
   --edge_buffer_size INTEGER RANGE
                                    The size of buffered channels to use between nodes in a pipeline. Larger values reduce backpressure at the cost of memory. Smaller
                                    values will push messages through the pipeline quicker. Must be greater than 1 and a power of 2 (i.e. 2, 4, 8, 16, etc.)  [default:
                                    128; x>=2]
   --use_cpp BOOLEAN               [Deprecated] Whether or not to use C++ node and message types or to prefer python. Only use as a last resort if bugs are encountered.
                                    Cannot be used with --use_cpu_only  [default: True]
   --use_cpu_only                  Whether or not to run in CPU only mode, setting this to True will disable C++ mode. Cannot be used with --use_cpp
   --manual_seed INTEGER RANGE     Manually seed the random number generators used by Morpheus, useful for testing.  [x>=1]
   --help                          Show this message and exit.

   Commands:
   pipeline-fil    Run the inference pipeline with a FIL model
   pipeline-nlp    Run the inference pipeline with a NLP model
   pipeline-other  Run a custom inference pipeline without a specific model type


Currently, Morpheus pipeline can be operated in four different modes.

 * ``pipeline-fil``
    * This pipeline mode is used to run inference on FIL (Forest Inference Library) models such as XGBoost, RandomForestClassifier, etc.
 * ``pipeline-nlp``
    * This pipeline mode is used to run inference on NLP models, it offers the ability to tokenize the input data prior to submitting the inference requests.
 * ``pipeline-other``
    * Run a customized inference pipeline without using a specific model type.

Similar to the run command, we can get help on the tools:

.. code-block:: console

   $ morpheus tools --help
   Usage: morpheus tools [OPTIONS] COMMAND [ARGS]...

   Tools subcommand

   Options:
   --help  Show this message and exit.

   Commands:
   autocomplete  Utility for installing/updating/removing shell completion for Morpheus
   onnx-to-trt   Converts an ONNX model to a TRT engine


The help text will show arguments, options and all possible sub-commands. Help for each of these sub-commands can be
queried in the same manner:

.. code-block:: console

   $ morpheus tools onnx-to-trt --help
   Usage: morpheus tools onnx-to-trt [OPTIONS]

   Converts an ONNX model to a TRT engine

   Options:
   --input_model PATH              [required]
   --output_model PATH             [required]
   --batches <INTEGER INTEGER>...  [required]
   --seq_length INTEGER            [required]
   --max_workspace_size INTEGER    [default: 16000]
   --help                          Show this message and exit.

ONNX To TensorRT
----------------
The ONNX to TensorRT (TRT) conversion utility requires additional packages, which can be installed using the following command:
```bash
conda env update --solver=libmamba -n morpheus --file conda/environments/model-utils_cuda-125_arch-$(arch).yaml
```

Example usage of the ONNX to TRT conversion utility can be found in `models/README.md <https://github.com/nv-morpheus/Morpheus/blob/branch-25.02/models/README.md#generating-trt-models-from-onnx>`_.

AutoComplete
------------

The Morpheus CLI supports bash, fish, zsh, and PowerShell autocompletion. To set up autocomplete, it must first be
installed. Morpheus comes with a tool to assist with this:

.. code-block:: console

   $ morpheus tools autocomplete install
   bash completion installed in ~/.bash_completion
   $ source ~/.bash_completion

After autocomplete has been installed, ``[TAB]`` can be used to show all commands, options and arguments when building
pipelines via the CLI:

.. code-block:: console

   $ morpheus run pipeline- # [TAB][TAB]
   pipeline-fil  pipeline-nlp
