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

import typing

if typing.TYPE_CHECKING:
    from .stage_base import StageBase


class PortSchema:

    def __init__(self, port_type: type = None):
        self._type = port_type
        self._completed = False

    def get_type(self) -> type:
        return self._type

    def set_type(self, value: type):
        assert not self._completed, "Attempted to set type on completed PortSchema"
        self._type = value

    def _complete(self):
        assert not self._completed, "Attempted to PortSchema._complete() twice"
        assert self._type is not None, "Attempted to complete PortSchema without setting type"
        self._completed = True

    def is_complete(self) -> bool:
        return self._completed


class StageSchema:

    def __init__(self, stage: "StageBase"):
        self._input_schemas = []
        for port in stage.input_ports:
            input_schema = port.get_input_schema()
            assert input_schema.is_complete(), \
                f"Attempted to create StageSchema for {stage} with incomplete input port schemas"
            self._input_schemas.append(input_schema)

        self._output_schemas = [PortSchema() for _ in range(len(stage.output_ports))]

    @property
    def input_schemas(self) -> list[PortSchema]:
        """
        Return all input schemas, one for each input port.
        """
        return self._input_schemas

    @property
    def input_types(self) -> list[type]:
        """
        Return the type associated with each input port.

        Convenience function for calling `port_schema.get_type()` for each element in `input_schemas`.
        """
        return [port_schema.get_type() for port_schema in self._input_schemas]

    @property
    def input_schema(self) -> PortSchema:
        """
        Single port variant of input_schemas. Will fail if there are multiple input ports.
        """
        assert len(self._input_schemas) == 1, \
            "Attempted to access input_schema property on StageSchema with multiple inputs"
        return self._input_schemas[0]

    @property
    def input_type(self) -> type:
        """
        Single port variant of input_types. Will fail if there are multiple input ports.
        """
        return self.input_schema.get_type()

    @property
    def output_schemas(self) -> list[PortSchema]:
        """
        Return all output schemas, one for each output port.
        """
        return self._output_schemas

    @property
    def output_schema(self) -> PortSchema:
        """
        Single port variant of output_schemas. Will fail if there are multiple output ports.
        """
        assert len(self._output_schemas) == 1, \
            "Attempted to access output_schema property on StageSchema with multiple outputs"
        return self._output_schemas[0]

    def _complete(self):
        """
        Calls `_complete` on all output port schemas.
        This will trigger an assertion error if any of the output port schemas do not have a type set, or have
        previously been completed. Users should not call this function directly, as this is called internally by the
        `StageBase` and `Receiver` classes.
        """
        for port_schema in self.output_schemas:
            # This locks the port schema
            port_schema._complete()
