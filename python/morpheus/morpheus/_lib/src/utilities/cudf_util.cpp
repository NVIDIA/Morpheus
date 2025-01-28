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

#include "morpheus/utilities/cudf_util.hpp"

#include "morpheus/objects/data_table.hpp"
#include "morpheus/objects/table_info.hpp"

#include <cudf/table/table.hpp>  // IWYU pragma: keep
#include <glog/logging.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <cstdlib>  // for getenv
#include <memory>
#include <ostream>  // Needed for logging
#include <utility>  // for move
/*
 * **************This needs to come last.********************
 * A note to posterity: We only ever want to have a single place where cudf_helpers_api.h is included in any
 * translation unit.
 */
#include "cudf_helpers_api.h"

namespace morpheus {

CudfHelper::CudfHelper()
{
    // Avoid loading cudf_helpers if we are in a sphinx build
    if (std::getenv("MORPHEUS_IN_SPHINX_BUILD") == nullptr)
    {
        if (import_morpheus___lib__cudf_helpers() != 0)
        {
            pybind11::error_already_set ex;

            LOG(ERROR) << "Could not load cudf_helpers library: " << ex.what();
            throw ex;
        }
    }
}

void CudfHelper::load()
{
    static CudfHelper s;
}

pybind11::object proxy_table_from_table_with_metadata(cudf::io::table_with_metadata&& table, int index_col_count)
{
    CudfHelper::load();
    return pybind11::reinterpret_steal<pybind11::object>(
        (PyObject*)make_table_from_table_with_metadata(std::move(table), index_col_count));
}

morpheus::TableInfoData proxy_table_info_data_from_table(pybind11::object table)
{
    CudfHelper::load();
    return make_table_info_data_from_table(table.ptr());
}

pybind11::object CudfHelper::table_from_table_with_metadata(cudf::io::table_with_metadata&& table, int index_col_count)
{
    return proxy_table_from_table_with_metadata(std::move(table), index_col_count);
}

pybind11::object CudfHelper::table_from_table_info(const TableInfoBase& table_info)
{
    CudfHelper::load();
    // Get the table info data from the table_into
    auto table_info_data = table_info.get_data();

    auto py_object_parent = table_info.get_parent()->get_py_object();

    // Need to guarantee that we have the gil here
    pybind11::gil_scoped_acquire gil;

    return pybind11::reinterpret_steal<pybind11::object>(
        (PyObject*)make_table_from_table_info_data(std::move(table_info_data), py_object_parent.ptr()));
}

TableInfoData CudfHelper::table_info_data_from_table(pybind11::object table)
{
    return proxy_table_info_data_from_table(std::move(table));
}

}  // namespace morpheus
