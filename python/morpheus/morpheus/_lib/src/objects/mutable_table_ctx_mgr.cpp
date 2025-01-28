/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/mutable_table_ctx_mgr.hpp"

#include "morpheus/utilities/string_util.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for attribute_error
#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {

namespace py = pybind11;

/********** MutableTableCtxMgr **********/
MutableTableCtxMgr::MutableTableCtxMgr(MessageMeta& meta_msg) :
  m_meta_msg{meta_msg},
  m_table{nullptr},
  m_py_table{nullptr} {};

py::object MutableTableCtxMgr::enter()
{
    // Release the GIL
    py::gil_scoped_release no_gil;
    m_table    = std::make_unique<MutableTableInfo>(std::move(m_meta_msg.get_mutable_info()));
    m_py_table = m_table->checkout_obj();
    return *m_py_table;
}

void MutableTableCtxMgr::exit(const py::object& type, const py::object& value, const py::object& traceback)
{
    std::unique_ptr<pybind11::object> ptr{nullptr};
    m_py_table.swap(ptr);

    m_table->return_obj(std::move(ptr));
    m_table.reset(nullptr);
}

void MutableTableCtxMgr::throw_usage_error(pybind11::args args, const pybind11::kwargs& kwargs)
{
    throw py::attribute_error(
        MORPHEUS_CONCAT_STR("Error attempting to use mutable_dataframe outside of context manager. Intended usage :\n"
                            "with message_meta.mutable_dataframe() as df:\n"
                            "    df['col'] = 5"));
}

}  // namespace morpheus
