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

import pytest

from morpheus.utils.execution_chain import ExecutionChain


def func1(arg1: int, arg2: int) -> dict:
    return {'result': arg1 + arg2}


def func2(arg1: int, arg2: int) -> dict:
    return {'result': arg1 - arg2}


# pylint: disable=unused-argument
def func_error(arg1: int, arg2: int) -> dict:
    raise ValueError("An error occurred.")


def test_execution_chain():
    chain = ExecutionChain([func1, func2])
    result = chain(arg1=3, arg2=3)
    assert result == {'result': 0}


def test_execution_chain_error():
    chain = ExecutionChain([func1, func_error, func2])

    with pytest.raises(Exception, match="Execution failed processing function func_error. Error: An error occurred."):
        chain(arg1=3, arg2=3)


def test_add_function():
    chain = ExecutionChain([func1])
    chain.add_function(func2)
    assert chain.functions == [func1, func2]


def test_remove_function():
    chain = ExecutionChain([func1, func2])
    chain.remove_function(func1)
    assert chain.functions == [func2]


def test_replace_function():
    chain = ExecutionChain([func1])
    chain.replace_function(func1, func2)
    assert chain.functions == [func2]


def test_nested_execution_chain():

    def inner_func1(arg1, arg2) -> dict:
        return {'result': arg1 + arg2}

    def inner_func2(result) -> dict:
        chain = ExecutionChain([inner_func1])
        return {'nested_result': chain(arg1=2, arg2=3)}

    chain = ExecutionChain([inner_func1, inner_func2])
    result = chain(arg1=3, arg2=3)
    assert result == {'nested_result': {'result': 5}}
