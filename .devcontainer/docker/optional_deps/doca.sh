#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

MORPHEUS_SUPPORT_DOCA=${MORPHEUS_SUPPORT_DOCA:-OFF}
LINUX_DISTRO=${LINUX_DISTRO:-ubuntu}
LINUX_VER=${LINUX_VER:-22.04}
DOCA_VERSION=${DOCA_VERSION:-2.7.0}
PKG_ARCH=${PKG_ARCH:-$(dpkg --print-architecture)}

# Exit early if nothing to do
if [[ ${MORPHEUS_SUPPORT_DOCA} != @(TRUE|ON) ]]; then
   exit 0
fi

WORKING_DIR=$1

echo "Installing DOCA using directory: ${WORKING_DIR}"

DEB_DIR=${WORKING_DIR}/deb

mkdir -p ${DEB_DIR}

DOCA_OS_VERSION="ubuntu2204"
DOCA_PKG_LINK="https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host_${DOCA_VERSION}-204000-24.04-${DOCA_OS_VERSION}_${PKG_ARCH}.deb"

# Upgrade the base packages (diff between image and Canonical upstream repo)
apt update -y
apt upgrade -y

# Install wget
apt install -y --no-install-recommends wget

wget -qO - ${DOCA_PKG_LINK} -O doca-host.deb
apt install ./doca-host.deb
apt update
apt install -y doca-all
apt install -y doca-gpu doca-gpu-dev

# Now install the gdrcopy library according to: https://github.com/NVIDIA/gdrcopy
GDRCOPY_DIR=${WORKING_DIR}/gdrcopy

if [[ ! -d "${GDRCOPY_DIR}" ]] ; then
    git clone https://github.com/NVIDIA/gdrcopy.git ${GDRCOPY_DIR}
    cd ${GDRCOPY_DIR}
else
    cd ${GDRCOPY_DIR}
    git pull https://github.com/NVIDIA/gdrcopy.git
fi

make lib lib_install
