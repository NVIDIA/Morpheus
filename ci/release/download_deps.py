#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""
Script to automate downloading of source code for third party dependencies

Intentionally using as few third-party dependencies as possible to allow running this script outside of a Morpheus
Conda environment.
"""

import logging
import os
import shutil
import sys
import tempfile

SCRIPT_DIR = os.path.relpath(os.path.dirname(__file__))
PROJ_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
UTILITIES_RELEASE_DIR = os.path.join(PROJ_ROOT, "external/utilities/ci/release")

sys.path.append(UTILITIES_RELEASE_DIR)
# pylint: disable=wrong-import-position
from download_deps_lib import PACKAGE_TO_URL_FN_T  # noqa: E402
from download_deps_lib import TAG_BARE  # noqa: E402
from download_deps_lib import TAG_NAME_DASH_BARE  # noqa: E402
from download_deps_lib import download_source_deps  # noqa: E402
from download_deps_lib import parse_args  # noqa: E402

# pylint: enable=wrong-import-position

CONDA_JSON_CMD = "./docker/run_container_release.sh conda list --json > .tmp/container_pkgs.json"

# In some cases multiple packages are derived from a single upstream repo, please keep sorted
PACKAGE_ALIASES = {  # <conda package nanme>: <upstream name>
    "elasticsearch": "elasticsearch-py",
    "grpcio": "grpc",
    "grpcio-status": "grpc",
    "milvus": "milvus-lite",
    "nlohmann_json": "json",
    'python': 'cpython',
    "python-confluent-kafka": "confluent-kafka-python",
    "python-graphviz": "graphviz",
    "torch": "pytorch",
    'versioneer': 'python-versioneer',
}

KNOWN_GITHUB_URLS = {  # <package>: <github repo>, please keep sorted
    'appdirs': 'https://github.com/ActiveState/appdirs',
    'c-ares': 'https://github.com/c-ares/c-ares',
    'click': 'https://github.com/pallets/click',
    'confluent-kafka-python': 'https://github.com/confluentinc/confluent-kafka-python',
    'cpython': 'https://github.com/python/cpython',
    'cupy': 'https://github.com/cupy/cupy',
    'databricks-cli': 'https://github.com/databricks/databricks-cli',
    'datacompy': 'https://github.com/capitalone/datacompy',
    'dfencoder': 'https://github.com/AlliedToasters/dfencoder',
    'dill': 'https://github.com/uqfoundation/dill',
    'docker-py': 'https://github.com/docker/docker-py',
    'elasticsearch-py': 'https://github.com/elastic/elasticsearch-py',
    'feedparser': 'https://github.com/kurtmckee/feedparser',
    'gflags': 'https://github.com/gflags/gflags',
    'glog': 'https://github.com/google/glog',
    'graphviz': 'https://github.com/xflr6/graphviz',
    'grpc': 'https://github.com/grpc/grpc',
    'json': 'https://github.com/nlohmann/json',
    'librdkafka': 'https://github.com/confluentinc/librdkafka',
    'libwebp': 'https://github.com/webmproject/libwebp',
    'mlflow': 'https://github.com/mlflow/mlflow',
    'networkx': 'https://github.com/networkx/networkx',
    'numpydoc': 'https://github.com/numpy/numpydoc',
    'nv-RxCpp': 'https://github.com/mdemoret-nv/RxCpp',
    'pip': 'https://github.com/pypa/pip',
    'pluggy': 'https://github.com/pytest-dev/pluggy',
    'protobuf': 'https://github.com/protocolbuffers/protobuf',
    'pybind11': 'https://github.com/pybind/pybind11',
    'pybind11-stubgen': 'https://github.com/sizmailov/pybind11-stubgen',
    'pydantic': 'https://github.com/pydantic/pydantic',
    'pymilvus': 'https://github.com/milvus-io/pymilvus',
    'python-versioneer': 'https://github.com/python-versioneer/python-versioneer',
    'rapidjson': 'https://github.com/Tencent/rapidjson',
    'rdma-core': 'https://github.com/linux-rdma/rdma-core',
    'requests': 'https://github.com/psf/requests',
    'requests-cache': 'https://github.com/requests-cache/requests-cache',
    'RxCpp': 'https://github.com/ReactiveX/RxCpp',
    'scikit-learn': 'https://github.com/scikit-learn/scikit-learn',
    'sqlalchemy': 'https://github.com/sqlalchemy/sqlalchemy',
    'pytorch': 'https://github.com/pytorch/pytorch',
    'tqdm': 'https://github.com/tqdm/tqdm',
    'typing_utils': 'https://github.com/bojiang/typing_utils',
    'versioneer-518': 'https://github.com/python-versioneer/versioneer-518',
    'watchdog': 'https://github.com/gorakhargosh/watchdog',
    'websockets': 'https://github.com/python-websockets/websockets',
    'zlib': 'https://github.com/madler/zlib',
}

KNOWN_GITLAB_URLS = {
    'pkg-config': 'https://gitlab.freedesktop.org/pkg-config/pkg-config',
}

OTHER_REPOS: dict[str, PACKAGE_TO_URL_FN_T] = {
    # While boost is available on GitHub, the sub-libraries are in separate repos.
    'beautifulsoup4':
        lambda name,
        ver: ("https://www.crummy.com/software/BeautifulSoup/bs4/download/"
              f"{'.'.join(ver.split('.')[:-1])}/{name}-{ver}.tar.gz"),
}

# Please keep sorted
KNOWN_FIRST_PARTY = frozenset(
    {'cuda-cudart', 'cuda-nvrtc', 'cuda-nvtx', 'cuda-version', 'cudf', 'mrc', 'rapids-dask-dependency', 'tritonclient'})

# Some of these packages are installed via CPM (pybind11), others are transitive deps who's version is determined by
# other packages but we use directly (glog), while others exist in the build environment and are statically linked
# (zlib) and not specified in the runtime environment.
# Unfortunately this means these versions will need to be updated manually, although any that exist in the resolved
# environment will have their versions updated to match the resolved environment.
KNOWN_NON_CONDA_DEPS = [
    ('c-ares', '1.32.3'),
    ('dfencoder', '0.0.37'),
    ('gflags', '2.2.2'),
    ('glog', '0.7.1'),
    ('nlohmann_json', '3.11.3'),
    ('nv-RxCpp', '4.1.1.2'),
    ('librdkafka', '1.6.2'),
    ('pkg-config', '0.29.2'),
    ('protobuf', '4.25.3'),
    ('pybind11', '2.8.1'),
    ('pybind11-stubgen', '0.10.5'),
    ('python-versioneer', '0.22'),
    ('rapidjson', '1.1.0'),
    ('rdma-core', '54.0'),
    ('RxCpp', '4.1.1'),
    ('versioneer', '0.18'),
    ('versioneer-518', '0.19'),  # Conda has a version, but the git repo is unversioned
    ('zlib', '1.3.1'),
]


GIT_TAG_FORMAT = {  # any packages not in this dict are assumned to have the TAG_V_PREFIX
    'appdirs': TAG_BARE,
    'click': TAG_BARE,
    'databricks-cli': TAG_BARE,
    'dill': TAG_NAME_DASH_BARE,
    'docker-py': TAG_BARE,
    'feedparser': TAG_BARE,
    'graphviz': TAG_BARE,
    'networkx': TAG_NAME_DASH_BARE,
    'pip': TAG_BARE,
    'pkg-config': TAG_NAME_DASH_BARE,
    'pluggy': TAG_BARE,
    'pybind11-stubgen': TAG_BARE,
    'python-versioneer': TAG_BARE,
    'scikit-learn': TAG_BARE,
    'sqlalchemy': lambda ver: f"rel_{ver.replace('.', '_')}",
    'websockets': TAG_BARE,
}

logger = logging.getLogger(__file__)


def main():
    args = parse_args(conda_json_cmd=CONDA_JSON_CMD,
                      default_conda_yaml=os.path.join(PROJ_ROOT,
                                                      "conda/environments/runtime_cuda-125_arch-x86_64.yaml"),
                      default_conda_json=os.path.join(PROJ_ROOT, ".tmp/container_pkgs.json"))
    log_level = logging._nameToLevel[args.log_level.upper()]
    logging.basicConfig(level=log_level, format="%(message)s")

    # Set the log level for requests and urllib3
    logging.getLogger('requests').setLevel(args.http_log_level)
    logging.getLogger("urllib3").setLevel(args.http_log_level)

    needs_cleanup = False
    download_dir: str | None = args.download_dir
    if download_dir is None:
        if args.download:
            download_dir = tempfile.mkdtemp(prefix="morpheus_deps_download_")
            logger.info("Created temporary download directory: %s", download_dir)
            needs_cleanup = True
        elif args.extract:
            logger.error("--extract requires either --download or --download_dir to be set.")
            sys.exit(1)

    extract_dir: str | None = args.extract_dir
    if extract_dir is None and args.extract:
        extract_dir = tempfile.mkdtemp(prefix="morpheus_deps_extract_")
        logger.info("Created temporary extract directory: %s", extract_dir)

    num_missing_packages = download_source_deps(conda_yaml=args.conda_yaml,
                                                conda_json=args.conda_json,
                                                package_aliases=PACKAGE_ALIASES,
                                                known_github_urls=KNOWN_GITHUB_URLS,
                                                known_gitlab_urls=KNOWN_GITLAB_URLS,
                                                other_repos=OTHER_REPOS,
                                                known_first_party=KNOWN_FIRST_PARTY,
                                                git_tag_format=GIT_TAG_FORMAT,
                                                known_non_conda_deps=KNOWN_NON_CONDA_DEPS,
                                                dry_run=args.dry_run,
                                                verify_urls=args.verify_urls,
                                                download=args.download,
                                                download_dir=download_dir,
                                                extract=args.extract,
                                                extract_dir=extract_dir)

    if needs_cleanup and not args.no_clean and download_dir is not None:
        logger.info("Removing temporary download directory: %s", download_dir)
        shutil.rmtree(download_dir)

    if num_missing_packages > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
