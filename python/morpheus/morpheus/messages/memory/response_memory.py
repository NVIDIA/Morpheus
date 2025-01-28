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

import dataclasses
import logging

import pandas as pd

import morpheus._lib.messages as _messages
from morpheus.messages.data_class_prop import DataClassProp
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.utils import logger as morpheus_logger
from morpheus.utils.type_aliases import NDArrayType

logger = logging.getLogger(__name__)


@dataclasses.dataclass(init=False)
class ResponseMemory(TensorMemory, cpp_class=_messages.ResponseMemory):
    """Output memory block holding the results of inference."""

    def __new__(cls, *args, **kwargs):
        morpheus_logger.deprecated_message_warning(cls, TensorMemory)
        return super().__new__(cls, *args, **kwargs)

    def get_output(self, name: str):
        """
        Get the Tensor stored in the container identified by `name`. Alias for `ResponseMemory.get_tensor`.

        Parameters
        ----------
        name : str
            Key used to do lookup in tensors dict of message container.

        Returns
        -------
        NDArrayType
            Tensors corresponding to name.

        Raises
        ------
        KeyError
            If output name does not exist in message container.

        """
        return self.get_tensor(name)

    def set_output(self, name: str, tensor: NDArrayType):
        """
        Update the output tensor identified by `name`. Alias for `ResponseMemory.set_tensor`

        Parameters
        ----------
        name : str
            Key used to do lookup in tensors dict of the container.
        tensor : NDArrayType
            Tensor as either a CuPy or NumPy array.

        Raises
        ------
        ValueError
            If the number of rows in `tensor` does not match `count`
        """
        self.set_tensor(name, tensor)


@dataclasses.dataclass(init=False)
class ResponseMemoryProbs(ResponseMemory, cpp_class=_messages.ResponseMemoryProbs):
    """
    Subclass of `ResponseMemory` containng an output tensor named 'probs'.

    Parameters
    ----------
    probs : NDArrayType
        Probabilities tensor
    """
    probs: dataclasses.InitVar[NDArrayType] = DataClassProp(ResponseMemory._get_tensor_prop, ResponseMemory.set_output)

    def __init__(self, *, count: int, probs: NDArrayType):
        super().__init__(count=count, tensors={'probs': probs})


@dataclasses.dataclass(init=False)
class ResponseMemoryAE(ResponseMemory, cpp_class=None):
    """
    Subclass of `ResponseMemory` specific to the AutoEncoder pipeline.

    Parameters
    ----------
    probs : NDArrayType
        Probabilities tensor

    user_id : str
        User id the inference was performed against.

    explain_df : pd.Dataframe
        Explainability Dataframe, for each feature a column will exist with a name in the form of: `{feature}_z_loss`
        containing the loss z-score along with `max_abs_z` and `mean_abs_z` columns
    """
    probs: dataclasses.InitVar[NDArrayType] = DataClassProp(ResponseMemory._get_tensor_prop, ResponseMemory.set_output)
    user_id: str = ""
    explain_df: pd.DataFrame = None

    def __init__(self, *, count: int, probs: NDArrayType):
        super().__init__(count=count, tensors={'probs': probs})
