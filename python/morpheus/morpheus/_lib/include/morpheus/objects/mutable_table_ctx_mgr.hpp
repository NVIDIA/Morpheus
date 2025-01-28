/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "morpheus/export.h"
#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/table_info.hpp"

#include <pybind11/pytypes.h>  // for object

#include <memory>  // for unique_ptr

namespace morpheus {
/**
 * @addtogroup objects
 * @{
 * @file
 */

class MORPHEUS_EXPORT MutableTableCtxMgr
{
  public:
    MutableTableCtxMgr(MessageMeta& meta_msg);
    pybind11::object enter();
    void exit(const pybind11::object& type, const pybind11::object& value, const pybind11::object& traceback);

    // Throws a useful exception when a user attempts to use this object as if it were the dataframe itself
    void throw_usage_error(pybind11::args args, const pybind11::kwargs& kwargs);

  private:
    MessageMeta& m_meta_msg;
    std::unique_ptr<MutableTableInfo> m_table;
    std::unique_ptr<pybind11::object> m_py_table;
};
/** @} */  // end of group
}  // namespace morpheus
