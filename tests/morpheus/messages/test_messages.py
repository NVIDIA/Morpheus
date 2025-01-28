# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp
import pytest

import cudf

import morpheus._lib.messages as _messages
import morpheus.config
import morpheus.utils.type_utils
from morpheus import messages
from morpheus.messages.memory import tensor_memory


def check_message(python_type: type, cpp_type: type, should_be_cpp: bool, no_cpp_class: bool, *args: tuple, **kwargs):
    instance = python_type(*args, **kwargs)

    # Check that the C++ type is set in the class
    expected_cpp_class = None if no_cpp_class else cpp_type
    assert python_type._cpp_class is expected_cpp_class

    # Check that the isinstance to Python type works
    assert isinstance(instance, python_type)

    # Check that the instantiated class is the right type
    expected_class = cpp_type if should_be_cpp and cpp_type is not None else python_type
    assert instance.__class__ is expected_class


def check_all_messages(should_be_cpp: bool, no_cpp_class: bool):

    df = cudf.DataFrame(range(1), columns="test")

    check_message(messages.MessageMeta, _messages.MessageMeta, should_be_cpp, no_cpp_class, *(df, ))

    # UserMessageMeta doesn't contain a C++ impl, so we should
    # always received the python impl
    check_message(messages.UserMessageMeta, None, should_be_cpp, no_cpp_class, *(None, None))

    check_message(tensor_memory.TensorMemory, _messages.TensorMemory, should_be_cpp, no_cpp_class, **{"count": 1})
    check_message(messages.InferenceMemory, _messages.InferenceMemory, should_be_cpp, no_cpp_class, **{"count": 1})

    cp_array = cp.zeros((1, 2))

    check_message(messages.InferenceMemoryNLP,
                  _messages.InferenceMemoryNLP,
                  should_be_cpp,
                  no_cpp_class,
                  **{
                      "count": 1, "input_ids": cp_array, "input_mask": cp_array, "seq_ids": cp_array
                  })

    check_message(messages.InferenceMemoryFIL,
                  _messages.InferenceMemoryFIL,
                  should_be_cpp,
                  no_cpp_class,
                  **{
                      "count": 1, "input__0": cp_array, "seq_ids": cp_array
                  })

    # No C++ impl, should always get the Python class
    check_message(messages.InferenceMemoryAE,
                  None,
                  should_be_cpp,
                  no_cpp_class,
                  **{
                      "count": 1, "inputs": cp_array, "seq_ids": cp_array
                  })

    check_message(messages.ResponseMemory, _messages.ResponseMemory, should_be_cpp, no_cpp_class, **{"count": 1})

    check_message(messages.ResponseMemoryProbs,
                  _messages.ResponseMemoryProbs,
                  should_be_cpp,
                  no_cpp_class,
                  **{
                      "count": 1, "probs": cp_array
                  })

    # No C++ impl
    check_message(messages.ResponseMemoryAE, None, should_be_cpp, no_cpp_class, **{"count": 1, "probs": cp_array})


@pytest.mark.gpu_mode
def test_constructor_cpp():
    check_all_messages(morpheus.config.CppConfig.get_should_use_cpp(), False)
