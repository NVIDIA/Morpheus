# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
"""Main entrypoint for the Morpheus CLI, when installed this module is executed as `morpheus`"""


def run_cli():
    """Main entrypoint for the CLI"""
    from morpheus.cli.commands import cli  # pylint: disable=cyclic-import
    cli(obj={}, auto_envvar_prefix='MORPHEUS', show_default=True, prog_name="morpheus")


if __name__ == '__main__':

    run_cli()
