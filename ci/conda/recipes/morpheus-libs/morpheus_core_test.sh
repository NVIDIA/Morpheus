# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# For now conda tests are disabled on aarch64, largely due to difficulties with installing a cuda enabled version of
# pytorch on aarch64 from a requirements file.
if [[ $(arch) == "aarch64" ]]; then
    exit 0
fi

python3 <<EOF
import importlib.resources
import subprocess
requirements_file = importlib.resources.path("morpheus", "requirements_morpheus_core_arch-$(arch).txt")
subprocess.call(f"pip install -r {requirements_file}".split())
EOF

pytest tests/morpheus
