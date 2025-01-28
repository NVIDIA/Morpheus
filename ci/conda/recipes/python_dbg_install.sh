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

set -e

function usage {
  echo "$0 -s [CPYTHON_SOURCE] -p [CONDA_INSTALL_FILE_PATH] -i [SOURCE_INSTALL_PATH] -g [GDBINIT_INSTALL_PATH]"

  exit 0
}

function help {
echo <<-DOC
Usage: $0 [OPTION]...
Install debug version of cpython, into the current conda environment, which has previously been downloaded/built.

Arguments:
-s            Path to cpython (s)ource tarball, source files will be extracted and installed to path specified by '-i'.
-c            Path to cpython (c)onda tarball, this is what will be installed into the conda environment.
-i            Path where cpython source files will be installed.
              Default: ci/conda/recipes/python-dbg/source
-g            Path to install point for cpython 'gdbinit' file. Ignored if empty.
              Note: Requires cpython source install point '-i' to be specified so that we can locate the gdb macros.
DOC

exit 0
}

function set_py_ver {
    ## Get the version information
    PYDEBUG_VERSION=$(python --version | cut -d ' ' -f2 )
    PYDEBUG_VERSION_MAJ_MIN=$(echo "${PYDEBUG_VERSION}" | awk '{split($0,ver,"."); print ver[1] "." ver[2]}')
}

PYDEBUG_CONDA_INSTALL_FILE=""
PYDEBUG_INSTALL_GDB_PATH=""
PYDEBUG_INSTALL_GDB_PATH=""
PYDEBUG_INSTALL_PATH=${PWD}/ci/conda/recipes/python-dbg/source
PYDEBUG_VERSION=""
PYDEBUG_VERSION_MAJ_MIN=""

while getopts "s:c:i:gh" opt; do
  case ${opt} in
    s) PYDEBUG_SOURCE=${OPTARG};;
    c) PYDEBUG_CONDA_INSTALL_FILE=${OPTARG};;
    i) PYDEBUG_INSTALL_PATH=${OPTARG};;
    g) PYDEBUG_INSTALL_GDB_PATH=${OPTARG};;
    h) help;;
    *) usage;;
  esac
done

# Install conda package
if [ -n "${PYDEBUG_CONDA_INSTALL_FILE}" ]; then
  if [ ! -f "${PYDEBUG_CONDA_INSTALL_FILE}" ]; then
    echo "Conda install file does not exist or is inaccessible: ${PYDEBUG_CONDA_INSTALL_FILE}"
    exit 1
  else
    echo "Installing cpython debug build: ${PYDEBUG_CONDA_INSTALL_FILE}"
    mamba install --use-local "${PYDEBUG_CONDA_INSTALL_FILE}"

    set_py_ver
    ## Conda package will install python source to python3.xd (for development), which the CMake configure files won't find
    ## Copy the includes to the python3.x folder so CMake can find them.
    cp -R /opt/conda/envs/morpheus/include/python${PYDEBUG_VERSION_MAJ_MIN}d/* \
          /opt/conda/envs/morpheus/include/python${PYDEBUG_VERSION_MAJ_MIN}
  fi
else
    echo "No Conda install file specified, skipping..."
    set_py_ver
fi

# Install cpython source files
if [[ -n ${PYDEBUG_SOURCE} && -n ${PYDEBUG_INSTALL_PATH} ]]; then
  if [[ ! -f ${PYDEBUG_SOURCE} ]]; then
    echo "Cpython source file does not exist or is inaccessible: ${PYDEBUG_SOURCE}"
    exit 1
  fi

  if [[ ! -f ${PYDEBUG_INSTALL_PATH} ]]; then
    mkdir -p "${PYDEBUG_INSTALL_PATH}"
  fi

  # Extract cpython source to /workspace \
  for src_dir in Include Misc Modules Objects Python; do
      tar --strip-components=1 --extract --wildcards --file="${PYDEBUG_SOURCE}" Python-${PYDEBUG_VERSION}/${src_dir}/*
      mv ./${src_dir} "${PYDEBUG_INSTALL_PATH}/${src_dir}"
  done
else
    echo "Missing cpython source tarball or install path, skipping..."
fi

# Install GDB init macros
if [[ -f ${PYDEBUG_INSTALL_PATH} ]]; then
  # Move cpython gdb helper macros to ${HOME}/.gdbinit \
  # See: https://github.com/python/cpython/blob/main/Misc/gdbinit
  if [[ "${PYDEBUG_INSTALL_GDB_PATH}" != "" ]]; then
    GDB_SRC_PATH="${PYDEBUG_INSTALL_PATH}/Misc/gdbinit"
    if [[ ! -f "${GDB_SRC_PATH}" ]]; then
      echo "gdbinit path does not exist or is inaccessible: ${GDB_SRC_PATH}"
      exit 1
    fi

    cp "${GDB_SRC_PATH}" "${PYDEBUG_INSTALL_GDB_PATH}"
  fi
fi
