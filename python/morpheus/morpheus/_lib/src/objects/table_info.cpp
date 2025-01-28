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

#include "morpheus/objects/table_info.hpp"

#include "morpheus/objects/dtype.hpp"

#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/gil.h>       // for gil_scoped_acquire
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <algorithm>  // for find, transform
#include <cstddef>    // for size_t
#include <iterator>   // for back_insert_iterator, back_inserter
#include <memory>
#include <optional>
#include <ostream>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
// IWYU pragma: no_include <pybind11/cast.h>

namespace morpheus {

namespace py = pybind11;
using namespace py::literals;

/**
 * @brief Helper function for calculating the slice of a `TableInfoData` object
 *
 * @param table Object to slice
 * @param start Start row for the slice. Relative to the "offset" in `table`
 * @param stop Stop row for the slice. Relative to the "offset" in `table`
 * @param column_names Columns to filter by
 * @return TableInfoData
 */
TableInfoData get_table_info_data_slice(const TableInfoData& table,
                                        cudf::size_type start,
                                        cudf::size_type stop,
                                        std::vector<std::string> column_names)
{
    CHECK_GE(start, 0) << "Start must be >= 0";

    if (stop < 0)
    {
        stop = table.table_view.num_rows();
    }

    CHECK_GT(stop, 0) << "Stop must be > 0";
    CHECK_LE(stop, table.table_view.num_rows()) << "Stop must be less than the number of rows";
    CHECK_LE(start, stop) << "Start must be less than stop";

    if (column_names.empty())
    {
        column_names = table.column_names;
    }

    // Start with our table view
    auto table_view_out = table.table_view;

    std::vector<cudf::size_type> col_indices;
    std::vector<cudf::size_type> col_indices_mapppings;

    // If the columns are different, calculate the new slice
    if (column_names != table.column_names)
    {
        // Append the indices column idx by default
        for (cudf::size_type i = 0; i < table.index_names.size(); ++i)
        {
            col_indices.push_back(i);
        }

        std::transform(column_names.begin(),
                       column_names.end(),
                       std::back_inserter(col_indices),
                       [&table, &col_indices_mapppings](const std::string& c) {
                           auto found_col = std::find(table.column_names.begin(), table.column_names.end(), c);

                           if (found_col == table.column_names.end())
                           {
                               throw py::key_error("Unknown column: " + c);
                           }

                           auto idx = (found_col - table.column_names.begin());

                           col_indices_mapppings.push_back(idx);

                           return idx + table.index_names.size();
                       });

        table_view_out = table_view_out.select(col_indices);
    }

    // If the start/stop is different, then perform the slice
    if (start != 0 || stop != table.table_view.num_rows())
    {
        table_view_out = cudf::slice(table_view_out, {start, stop})[0];
    }

    if (col_indices_mapppings.size() > 0)
    {
        // Create a new TableInfoData
        return {table_view_out, table.index_names, column_names, col_indices_mapppings};
    }

    // Create a new TableInfoData
    return {table_view_out, table.index_names, column_names};
}

/****** Component public implementations *******************/
/****** TableInfoBase****************************************/
TableInfoBase::TableInfoBase(std::shared_ptr<const IDataTable> parent, TableInfoData data) :
  m_parent(std::move(parent)),
  m_data(std::move(data))
{}

const cudf::table_view& TableInfoBase::get_view() const
{
    return m_data.table_view;
}

std::vector<std::string> TableInfoBase::get_index_names() const
{
    return m_data.index_names;
}

std::vector<std::string> TableInfoBase::get_column_names() const
{
    return m_data.column_names;
}

cudf::size_type TableInfoBase::num_indices() const
{
    return this->get_index_names().size();
}

cudf::size_type TableInfoBase::num_columns() const
{
    return this->get_column_names().size();
}

cudf::size_type TableInfoBase::num_rows() const
{
    return m_data.table_view.num_rows();
}

const cudf::column_view& TableInfoBase::get_column(cudf::size_type idx) const
{
    if (idx < 0 || idx >= this->m_data.table_view.num_columns())
    {
        throw std::invalid_argument("idx must satisfy 0 <= idx < num_columns()");
    }

    return this->m_data.table_view.column(this->m_data.index_names.size() + idx);
}

const std::shared_ptr<const IDataTable>& TableInfoBase::get_parent() const
{
    return m_parent;
}

TableInfoData& TableInfoBase::get_data()
{
    return m_data;
}

const TableInfoData& TableInfoBase::get_data() const
{
    return m_data;
}

bool TableInfoBase::has_sliceable_index() const
{
    py::gil_scoped_acquire gil;
    auto df    = m_parent->get_py_object();
    auto index = df.attr("index");

    auto is_unique               = index.attr("is_unique").cast<bool>();
    auto is_monotonic_increasing = index.attr("is_monotonic_increasing").cast<bool>();
    auto is_monotonic_decreasing = index.attr("is_monotonic_decreasing").cast<bool>();

    // Must be either increasing or decreasing with unique values to slice
    return is_unique && (is_monotonic_increasing || is_monotonic_decreasing);
}

TableInfo::TableInfo(std::shared_ptr<const IDataTable> parent,
                     std::shared_lock<std::shared_mutex> lock,
                     TableInfoData data) :
  TableInfoBase(parent, std::move(data)),
  m_lock(std::move(lock))
{}

TableInfo TableInfo::get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names) const
{
    // Create a new Table info, (cloning the shared_lock)
    return {this->get_parent(),
            std::shared_lock<std::shared_mutex>(*m_lock.mutex()),
            get_table_info_data_slice(this->get_data(), start, stop, column_names)};
}

MutableTableInfo::MutableTableInfo(std::shared_ptr<const IDataTable> parent,
                                   std::unique_lock<std::shared_mutex> lock,
                                   TableInfoData data) :
  TableInfoBase(parent, std::move(data)),
  m_lock(std::move(lock))
{}

MutableTableInfo::~MutableTableInfo()
{
    if (m_checked_out_ref_count >= 0)
    {
        LOG(ERROR) << "Checked out python object was not returned before MutableTableInfo went out of scope";
    }
}

MutableTableInfo MutableTableInfo::get_slice(cudf::size_type start,
                                             cudf::size_type stop,
                                             std::vector<std::string> column_names) &&
{
    // Create a new Table info, (moving the unique_lock)
    return {
        this->get_parent(), std::move(m_lock), get_table_info_data_slice(this->get_data(), start, stop, column_names)};
}

void MutableTableInfo::insert_columns(const std::vector<std::tuple<std::string, morpheus::DType>>& columns)
{
    const auto num_existing_cols = this->get_data().column_names.size();
    const auto num_rows          = this->get_data().table_view.num_rows();

    // TODO(mdemoret): figure out how to do this without the gil
    {
        py::gil_scoped_acquire gil;
        py::object cudf_scalar = py::module_::import("cudf").attr("Scalar");

        auto table = this->get_parent()->get_py_object();

        for (std::size_t i = 0; i < columns.size(); ++i)
        {
            auto scalar = cudf_scalar(0, std::get<1>(columns[i]).type_str());
            table.attr("insert")(num_existing_cols + i, std::get<0>(columns[i]), scalar);
            this->get_data().column_names.push_back(std::get<0>(columns[i]));
        }
    }
}

void MutableTableInfo::insert_missing_columns(const std::vector<std::tuple<std::string, morpheus::DType>>& columns)
{
    std::vector<std::tuple<std::string, morpheus::DType>> missing_columns;
    for (const auto& column : columns)
    {
        if (std::find(this->get_data().column_names.begin(),
                      this->get_data().column_names.end(),
                      std::get<0>(column)) == this->get_data().column_names.end())
        {
            missing_columns.push_back(column);
        }
    }

    if (!missing_columns.empty())
    {
        insert_columns(missing_columns);
    }
}

std::unique_ptr<pybind11::object> MutableTableInfo::checkout_obj()
{
    // Get a copy increasing the ref count
    py::object checked_out_obj = this->get_parent()->get_py_object();

    m_checked_out_ref_count = checked_out_obj.ref_count();

    auto ptr = std::make_unique<py::object>(std::move(checked_out_obj));

    return ptr;
}

void MutableTableInfo::return_obj(std::unique_ptr<pybind11::object>&& obj)
{
    obj.reset(nullptr);
    m_checked_out_ref_count = -1;
}

std::optional<std::string> MutableTableInfo::ensure_sliceable_index()
{
    std::optional<std::string> old_index_col_name{"_index_"};
    auto ptr_df = this->checkout_obj();
    {
        py::gil_scoped_acquire gil;
        auto& py_df   = *ptr_df;
        auto df_index = py_df.attr("index");

        // Check to see if we actually need the change
        if (df_index.attr("is_unique").cast<bool>() && (df_index.attr("is_monotonic_increasing").cast<bool>() ||
                                                        df_index.attr("is_monotonic_decreasing").cast<bool>()))
        {
            // Set the outputname to nullopt
            old_index_col_name = std::nullopt;
        }
        else
        {
            auto index_name = df_index.attr("name");

            if (!index_name.is_none())
            {
                old_index_col_name = *old_index_col_name + index_name.cast<std::string>();
            }

            df_index.attr("name") = py::str(*old_index_col_name);

            py_df.attr("reset_index")("inplace"_a = true);
        }
    }

    this->return_obj(std::move(ptr_df));

    // If we made a change, update the index and column list
    if (old_index_col_name.has_value())
    {
        auto& tbl_data = this->get_data();
        tbl_data.column_names.insert(tbl_data.column_names.begin(), *old_index_col_name);
        tbl_data.index_names.clear();
        tbl_data.index_names.emplace_back("");
    }

    return old_index_col_name;
}

}  // namespace morpheus
