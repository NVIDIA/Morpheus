#!/usr/bin/env python
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

import cudf

# Morpheus.common is required to register pre-made loaders
import morpheus.common  # noqa: F401 # pylint:disable=unused-import
from morpheus import messages
from morpheus.messages import DataLoaderRegistry


def test_loader_registry_contains():
    assert (not DataLoaderRegistry.contains("not_a_loader"))

    should_have = ["file", "grpc", "payload", "rest"]
    for loader in should_have:
        # Make sure all the loaders we expect to be in the registry are
        assert (DataLoaderRegistry.contains(loader))

    loaders = DataLoaderRegistry.list()
    for loader in should_have:
        # Make sure all the loaders in the registry are in the list
        assert (loader in loaders)  # pylint: disable=unsupported-membership-test

        # Make sure all the loaders in the list are contained in the registry
        assert (DataLoaderRegistry.contains(loader))


def test_loader_registry_register_loader():

    def test_loader(control_message: messages.ControlMessage, task: dict):
        task_properties = task['properties']
        if ('files' not in task_properties):
            raise ValueError("No files specified in config")

        files = task_properties['files']

        df = None
        for file in files:
            filepath = file['path']
            if df is None:
                df = cudf.read_csv(filepath)
            else:
                df = df.append(cudf.read_csv(filepath))

        control_message.payload(messages.MessageMeta(df))

        return control_message

    # Should be able to register a new loader
    DataLoaderRegistry.register_loader("test_loader_registry_register_loader", test_loader)
    assert (DataLoaderRegistry.contains("test_loader_registry_register_loader"))

    # Should be able to overwrite an existing loader if we request it
    DataLoaderRegistry.register_loader("test_loader_registry_register_loader", test_loader, False)

    try:
        # Shouldn't allow us to overwrite an existing loader by default
        DataLoaderRegistry.register_loader("test_loader_registry_register_loader", test_loader)
        assert (False)
    except RuntimeError:
        assert (True)


def test_loader_registry_unregister_loader():

    def test_loader(control_message: messages.ControlMessage, task: dict):
        task_properties = task['properties']
        if ('files' not in task_properties):
            raise ValueError("No files specified in config")

        files = task_properties['files']

        df = None
        for file in files:
            filepath = file['path']
            if df is None:
                df = cudf.read_csv(filepath)
            else:
                df = df.append(cudf.read_csv(filepath))

        control_message.payload(messages.MessageMeta(df))

        return control_message

    # Should be able to register a new loader
    DataLoaderRegistry.register_loader("test_loader_registry_unregister_loader", test_loader)
    assert (DataLoaderRegistry.contains("test_loader_registry_unregister_loader"))

    # Should be able to unregister a loader
    DataLoaderRegistry.unregister_loader("test_loader_registry_unregister_loader")
    assert (not DataLoaderRegistry.contains("test_loader_registry_unregister_loader"))

    # Shouldn't be able to unregister a loader that doesn't exist
    try:
        DataLoaderRegistry.unregister_loader("test_loader_registry_unregister_loader")
        assert (False)
    except RuntimeError:
        assert (True)

    # Should be able to unregister a loader that doesn't exist if we request it
    DataLoaderRegistry.unregister_loader("test_loader_registry_unregister_loader", False)


if __name__ == "__main__":
    test_loader_registry_contains()
    test_loader_registry_register_loader()
    test_loader_registry_unregister_loader()
