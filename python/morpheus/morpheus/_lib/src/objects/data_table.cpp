/*
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

#include "morpheus/objects/data_table.hpp"

#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/table_info_data.hpp"

#include <glog/logging.h>
#include <pybind11/gil.h>

#include <mutex>
#include <ostream>
#include <shared_mutex>
#include <utility>

namespace morpheus {

TableInfo IDataTable::get_info() const
{
    CHECK_EQ(PyGILState_Check(), 0)
        << "Cannot hold the Python GIL when accessing `get_info()`. Please release it first or deadlocks may occur.";

    // Get a shared lock while we get the table info (prevents mutation)
    std::shared_lock lock(m_mutex);

    // Get the table info data
    auto table_info_data = this->get_table_data();

    // From this, create a new TableInfo
    return {this->shared_from_this(), std::move(lock), std::move(table_info_data)};
}

MutableTableInfo IDataTable::get_mutable_info() const
{
    CHECK_EQ(PyGILState_Check(), 0) << "Cannot hold the Python GIL when accessing `get_mutable_info()`. Please release "
                                       "it first or deadlocks may occur.";

    // Get a unique lock while we get the table info (prevents mutation)
    std::unique_lock lock(m_mutex);

    // Get the table info data
    auto table_info_data = this->get_table_data();

    // From this, create a new TableInfo
    return {this->shared_from_this(), std::move(lock), std::move(table_info_data)};
}

}  // namespace morpheus
