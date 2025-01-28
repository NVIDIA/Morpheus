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

from _utils.llm import execute_node
from morpheus_llm.llm.nodes.prompt_template_node import PromptTemplateNode

MULTI_LINE_JINJA_TEMPLATE = """Testing a loop:
{% for lv in list_values -%}Title: {{ lv.title }}, Summary: {{ lv.summary }}
{% endfor %}
{{ query }}"""


@pytest.mark.parametrize(
    "template,template_format,values,expected_output",
    [("Hello {name}!",
      "f-string", {
          'name': ['World', 'Universe', 'Galaxy', 'Moon']
      }, ["Hello World!", "Hello Universe!", "Hello Galaxy!", "Hello Moon!"]),
     ("I would like one {fruit} and one {vegetable}.",
      "f-string", {
          'fruit': ['apple', 'plum'], 'vegetable': ['carrot', 'broccoli']
      }, ["I would like one apple and one carrot.", "I would like one plum and one broccoli."]),
     ("I would like one {{ fruit }} and one {{ vegetable }}.",
      "jinja", {
          'fruit': ['apple', 'plum'], 'vegetable': ['carrot', 'broccoli']
      }, ["I would like one apple and one carrot.", "I would like one plum and one broccoli."]),
     (MULTI_LINE_JINJA_TEMPLATE,
      "jinja",
      {
          'list_values': [[{
              'title': 'title1', 'summary': 'summary1'
          }, {
              'title': 'title2', 'summary': 'summary2'
          }], [{
              'title': 'rockets', 'summary': 'space'
          }]],
          'query': ['query1', 'query2']
      },
      [
          "Testing a loop:\nTitle: title1, Summary: summary1\nTitle: title2, Summary: summary2\nquery1",
          "Testing a loop:\nTitle: rockets, Summary: space\nquery2",
      ])],
    ids=["f-string-hello-world", "f-string-fruit-vegetable", "jinja-fruit-vegetable", "jinja-multi-line"])
def test_prompt_template_node(template: str, template_format: str, values: dict, expected_output: list[str]):
    node = PromptTemplateNode(template=template, template_format=template_format)
    assert sorted(node.get_input_names()) == sorted(values.keys())

    assert execute_node(node, **values) == expected_output


@pytest.mark.parametrize("template_format", ["f-string", "jinja"])
def test_prompt_template_pass_thru(template_format: str):
    template = "template without any variables should rase an exception"
    node = PromptTemplateNode(template=template, template_format=template_format)
    assert len(node.get_input_names()) == 0

    inputs = {'input': ['unused', 'placeholder']}
    assert execute_node(node, **inputs) == [template, template]


def test_unsupported_template_format():
    with pytest.raises(ValueError):
        PromptTemplateNode(template="Hello {name}!", template_format="unsupported")


@pytest.mark.parametrize("template", ["Hello {}!", "fruit: {fruit}, vegetable: {}, juice: {juice}"])
def test_no_unnamed_fields(template: str):
    with pytest.raises(ValueError):
        PromptTemplateNode(template=template, template_format="f-string")
