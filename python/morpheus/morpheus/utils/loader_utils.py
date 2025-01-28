# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

from morpheus.messages import DataLoaderRegistry as registry

logger = logging.getLogger(__name__)


def register_loader(loader_id):
    """
    Registers a loader if not exists in the dataloader registry.

    Parameters
    ----------
    loader_id : str
        Unique identifier for a loader in the dataloader registry.

    Returns
    -------
    inner_func
        Encapsulated function.
    """

    def inner_func(func):
        # Register a loader if not exists in the registry.
        if not registry.contains(loader_id):
            registry.register_loader(loader_id, func)
            logger.debug("Loader '%s' was successfully registered.", loader_id)
        else:
            logger.debug("Loader: '%s' already exists.", loader_id)

        return func

    return inner_func
