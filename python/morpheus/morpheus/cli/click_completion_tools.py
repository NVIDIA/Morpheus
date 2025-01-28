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
"""Methods to add morpheus auto completion to shells"""

import os
import subprocess

import click


def get_code(shell=None, path=None) -> str:
    """
    Return shell code to add morpheus auto completion. Return code is intended to be copy/pasted into shell startup
    scripts. Currently only bash is supported.
    """
    if (shell is None):
        shell = "bash"

    if (shell != "bash"):
        raise NotImplementedError("Only 'bash' can be used for the shell at this time")

    path = path or os.path.expanduser('~') + '/.bash_completion'

    tool_name = click.get_current_context().find_root().info_name

    # Support debugging this script
    if (tool_name.endswith(".py")):
        tool_name = "morpheus"

    tool_env_name = f"_{tool_name.upper().replace('-', '_')}_COMPLETE"

    response = subprocess.check_output([f"{tool_env_name}=bash_source morpheus"], shell=True).decode("utf-8")

    code = "# >>> morpheus completion >>>\n{}\n# <<< morpheus completion <<<\n".format(response.rstrip("\n"))

    return shell, path, code


def install_code(append=False, shell=None, path=None):
    """
    Write shell auto completion code to the shell startup script `path`. If `append` is `True` the code will be appended
    to the file, when `False` the file will be overwritten. Currently only bash is supported.
    """
    shell, path, code = get_code(shell, path)

    output_lines = []

    if (os.path.exists(path)):
        with open(path, 'r', encoding='UTF-8') as fh:
            input_lines = fh.readlines()

            found_match = False

            for line in input_lines:

                # If we have a match
                if (not append and "# >>> morpheus completion >>>" in line):
                    found_match = True

                if (not found_match):
                    output_lines.append(line)

                if (found_match and "# <<< morpheus completion <<<" in line):
                    found_match = False

    # Now append the output lines with our code
    output_lines.extend(code.splitlines(keepends=True))

    with open(path, 'w', encoding='UTF-8') as fh:
        fh.writelines(output_lines)

    return shell, path
