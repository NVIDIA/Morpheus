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

import asyncio
import logging
import string
import typing

from morpheus_llm.llm import LLMContext
from morpheus_llm.llm import LLMNodeBase

logger = logging.getLogger(__name__)


class PromptTemplateNode(LLMNodeBase):
    """
    Populates a template string with the values from the upstream node.

    Parameters
    ----------
    template : str
        The template string to populate.
    template_format : str, optional default="f-string"
        The format of the template string. Must be one of: f-string, jinja.
    """

    def __init__(self, template: str, template_format: typing.Literal["f-string", "jinja"] = "f-string") -> None:
        super().__init__()
        self._template_str = template
        self._template_format = template_format

        if (self._template_format == "f-string"):
            formatter = string.Formatter()
            # The parse method is returning an iterable of tuples in the form of:
            # (literal_text, field_name, format_spec, conversion)
            # https://docs.python.org/3.10/library/string.html#string.Formatter.parse
            self._input_names = []
            for (_, field_name, _, _) in formatter.parse(self._template_str):
                if field_name == '':
                    raise ValueError("Unnamed fields in templates are not supported")

                if field_name is not None:
                    self._input_names.append(field_name)

        elif (self._template_format == "jinja"):
            from jinja2 import Template
            from jinja2 import meta

            self._template_jinja = Template(self._template_str, enable_async=True, trim_blocks=True, lstrip_blocks=True)

            self._input_names = list(
                meta.find_undeclared_variables(self._template_jinja.environment.parse(self._template_str)))
        else:
            raise ValueError(f"Invalid template format: {self._template_format}, must be one of: f-string, jinja")

    def get_input_names(self):
        return self._input_names

    async def execute(self, context: LLMContext):  # pylint: disable=invalid-overridden-method

        # Get the keys from the task
        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        if (self._template_format == "f-string"):
            output_list = [self._template_str.format(**x) for x in input_list]
        elif (self._template_format == "jinja"):
            render_coros = [self._template_jinja.render_async(**inputs) for inputs in input_list]

            output_list = await asyncio.gather(*render_coros)

        context.set_output(output_list)

        return context
