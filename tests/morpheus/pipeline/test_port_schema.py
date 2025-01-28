#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from morpheus.pipeline.stage_schema import PortSchema


@pytest.mark.parametrize("port_type", [float, None])
def test_constructor(port_type: type):
    port_schema = PortSchema(port_type=port_type)
    assert port_schema.get_type() is port_type
    assert not port_schema.is_complete()


@pytest.mark.parametrize("port_type", [float, None])
def test_set_type(port_type: type):
    port_schema = PortSchema(port_type=port_type)

    port_schema.set_type(int)
    assert port_schema.get_type() is int


def test_complete():
    port_schema = PortSchema(port_type=float)
    assert not port_schema.is_complete()

    port_schema._complete()
    assert port_schema.is_complete()


def test_complete_error_no_type():
    port_schema = PortSchema()

    with pytest.raises(AssertionError):
        port_schema._complete()

    assert not port_schema.is_complete()


def test_complete_error_called_twice():
    port_schema = PortSchema(port_type=float)

    port_schema._complete()

    with pytest.raises(AssertionError):
        port_schema._complete()

    # Should still be complete
    assert port_schema.is_complete()
