/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "morpheus/doca/rte_context.hpp"

#include "morpheus/doca/error.hpp"

#include <glog/logging.h>
#include <rte_eal.h>

#include <array>
#include <ostream>
#include <vector>

namespace morpheus::doca {

RTEContext::RTEContext()
{
    auto argv = std::vector<char*>();

    std::array<char, 1> fake_program = {""};
    argv.push_back(fake_program.begin());

    std::array<char, 3> a_flag           = {"-a"};
    std::array<char, 8> fake_pci_address = {"00:00.0"};
    argv.push_back(a_flag.begin());
    argv.push_back(fake_pci_address.begin());

    RTE_TRY(rte_eal_init(argv.size(), argv.data()));
}

RTEContext::~RTEContext()
{
    auto eal_ret = rte_eal_cleanup();

    if (eal_ret < 0)
    {
        LOG(WARNING) << "EAL cleanup failed (" << eal_ret << ")" << std::endl;
    }
}

}  // namespace morpheus::doca
