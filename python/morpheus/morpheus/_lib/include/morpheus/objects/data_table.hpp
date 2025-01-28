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

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <memory>
#include <shared_mutex>

namespace morpheus {

struct TableInfoData;
struct TableInfo;
struct MutableTableInfo;

/****** Component public implementations *******************/
/****** IDataTable******************************************/

/**
 * @addtogroup objects
 * @{
 * @file
 */

/**
 * @brief Owning object which owns a unique_ptr<cudf::table>, table_metadata, and index information
 * Why this doesn't exist in cudf is beyond me
 */
struct IDataTable : public std::enable_shared_from_this<IDataTable>
{
    /**
     * @brief Construct a new IDataTable object
     *
     */
    IDataTable()          = default;
    virtual ~IDataTable() = default;

    /**
     * @brief cuDF dataframe rows count.
     *
     * @return cudf::size_type
     */
    virtual cudf::size_type count() const = 0;

    ///@{
    /**
     * @brief Gets a read-only instance of `TableInfo` which can be used to query and update the table from both C++ and
     * Python. This will block calls to `get_mutable_info` until all `TableInfo` object have been destroyed.
     *
     * @return TableInfo
     *
     * @note Read-only refers to changes made to the structure of a DataFrame. i.e. Adding/Removing columns, changing
     * column, types, adding/removing rows, etc. It's possible to update an existing range of data in a column with
     * `TableInfo`.
     */
    TableInfo get_info() const;
    ///@}

    ///@{
    /**
     * @brief Gets a writable instance of `MutableTableInfo` which can be used to modify the structure of the table from
     * both C++ and Python. This requires exclusive access to the underlying `IDataTable` and will block until all
     * `TableInfo` and `MutableTableInfo` objects have been destroyed. This class also provides direct access to the
     * underlying python object.
     *
     * @return MutableTableInfo
     *
     * @note Read-only refers to changes made to the structure of a DataFrame. i.e. Adding/Removing columns, changing
     * column, types, adding/removing rows, etc. It's possible to update an existing range of data in a column with
     * `TableInfo`.
     */
    MutableTableInfo get_mutable_info() const;
    ///@}

    /**
     * @brief Direct access to the underlying python object. Use only when absolutely necessary. `get_mutable_info()`
     * provides better checking when using the python object directly.
     *
     * @return const pybind11::object&
     */
    virtual const pybind11::object& get_py_object() const = 0;

  private:
    /**
     * @brief Gets the necessary information to build a `TableInfo` object from this interface. Must be implemented by
     * derived classes.
     *
     * @return TableInfoData
     */
    virtual TableInfoData get_table_data() const = 0;

    // Used to prevent locking to shared resources. Will need to be a boost fibers
    // supported mutex if we support C++ nodes with Fiber runables in the future
    mutable std::shared_mutex m_mutex{};
};
/** @} */  // end of group
}  // namespace morpheus
