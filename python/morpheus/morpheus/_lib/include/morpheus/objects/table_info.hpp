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

#pragma once

#include "morpheus/export.h"
#include "morpheus/objects/data_table.hpp"
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/table_info_data.hpp"

#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>      // for size_type
#include <pybind11/pytypes.h>  // for object

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <vector>

namespace morpheus {

struct CudfHelper;

/**
 * @addtogroup objects
 * @{
 * @file
 */
/****** Component public implementations *******************/
/****** TableInfo******************************************/
struct MORPHEUS_EXPORT TableInfoBase
{
    /**
     * @brief Get reference of a cudf table view
     *
     * @return cudf::table_view
     */
    const cudf::table_view& get_view() const;

    /**
     * @brief Get index names of a data table
     *
     * @return std::vector<std::string>
     */
    std::vector<std::string> get_index_names() const;

    /**
     * @brief Get column names of a data table
     *
     * @return std::vector<std::string>
     */
    std::vector<std::string> get_column_names() const;

    /**
     * @brief Get the number of indices in a data table
     *
     * @return cudf::size_type
     */
    cudf::size_type num_indices() const;

    /**
     * @brief Get columns count in a data table
     *
     * @return cudf::size_type
     */
    cudf::size_type num_columns() const;

    /**
     * @brief Get rows count in a data table
     *
     * @return cudf::size_type
     */
    cudf::size_type num_rows() const;

    /**
     * @brief Returns a reference to the view of the specified column
     *
     * @throws std::out_of_range
     * If `column_index` is out of the range [0, num_columns)
     *
     * @param idx : The index of the desired column
     * @return cudf::column_view : A reference to the desired column
     */
    const cudf::column_view& get_column(cudf::size_type idx) const;

    /**
     * @brief Returns true if the underlying dataframe as a unique index.
     *
     * @return bool
     */
    bool has_sliceable_index() const;

  protected:
    TableInfoBase() = default;

    TableInfoBase(std::shared_ptr<const IDataTable> parent, TableInfoData data);

    const std::shared_ptr<const IDataTable>& get_parent() const;

    TableInfoData& get_data();
    const TableInfoData& get_data() const;

  private:
    std::shared_ptr<const IDataTable> m_parent;
    TableInfoData m_data;

    // Give access to internal m_parent and m_data for converting to cudf dataframe
    friend CudfHelper;
};

struct MORPHEUS_EXPORT TableInfo : public TableInfoBase
{
  public:
    TableInfo() = default;
    TableInfo(std::shared_ptr<const IDataTable> parent, std::shared_lock<std::shared_mutex> lock, TableInfoData data);

    /**
     * @brief Get slice of a data table info based on the start and stop offset address
     *
     * @param start : Start offset address (inclusive)
     * @param stop : Stop offset address (exclusive)
     * @param column_names : Columns of interest
     * @return TableInfo
     */
    TableInfo get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names = {}) const;

  private:
    // We use a shared_lock to allow multiple `TableInfo` to be in use at the same time.
    std::shared_lock<std::shared_mutex> m_lock;
};

struct MORPHEUS_EXPORT MutableTableInfo : public TableInfoBase
{
  public:
    MutableTableInfo(std::shared_ptr<const IDataTable> parent,
                     std::unique_lock<std::shared_mutex> lock,
                     TableInfoData data);

    MutableTableInfo(MutableTableInfo&& other) = default;

    ~MutableTableInfo();

    /**
     * @brief Get slice of a data table info based on the start and stop offset address
     *
     * @param start : Start offset address (inclusive)
     * @param stop : Stop offset address (exclusive)
     * @param column_names : Columns of interest
     * @return TableInfo
     */
    MutableTableInfo get_slice(cudf::size_type start,
                               cudf::size_type stop,
                               std::vector<std::string> column_names = {}) &&;

    /**
     * TODO(Documentation)
     */
    void insert_columns(const std::vector<std::tuple<std::string, morpheus::DType>>& columns);

    /**
     * TODO(Documentation)
     */
    void insert_missing_columns(const std::vector<std::tuple<std::string, morpheus::DType>>& columns);

    /**
     * @brief Allows the python object to be "checked out" which gives exclusive access to the python object during the
     * lifetime of `MutableTableInfo`. Use this method when it is necessary to make changes to the python object using
     * the python API. The python object must be returned via `return_obj` before `MutableTableInfo` goes out of scope.
     *
     * @return std::unique_ptr<pybind11::object>
     */
    std::unique_ptr<pybind11::object> checkout_obj();

    /**
     * @brief Returns the checked out python object from `checkout_obj`. Each call to `checkout_obj` needs a
     * coresponding `return_obj` call.
     *
     * @param obj
     */
    void return_obj(std::unique_ptr<pybind11::object>&& obj);

    /**
     * @brief Replaces the index in the underlying dataframe if the existing one is not unique and monotonic. The old
     * index will be preserved in a column named `_index_{old_index.name}`. If `has_sliceable_index() == true`, this is
     * a no-op.
     *
     */
    std::optional<std::string> ensure_sliceable_index();

  private:
    // We use a unique_lock here to enforce exclusive access
    std::unique_lock<std::shared_mutex> m_lock;

    mutable int m_checked_out_ref_count{-1};
};

/** @} */  // end of group
}  // namespace morpheus
