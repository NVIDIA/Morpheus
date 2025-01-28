# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

# See the docstring in versioneer.py for instructions. Note that you must
# re-run 'versioneer setup' after changing this section, and commit the
# resulting files.

import versioneer
from setuptools import find_packages  # noqa: E402
from setuptools import setup  # noqa: E402

setup(
    name="morpheus_llm",
    version=versioneer.get_version(),  # pylint: disable=no-member
    description="Morpheus LLM",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    author="NVIDIA Corporation",
    include_package_data=True,
    packages=find_packages(exclude=['tests']),
    install_requires=[],
    license="Apache",
    python_requires='>=3.10, <4',
    cmdclass=versioneer.get_cmdclass(),  # pylint: disable=no-member
)
