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
#include "morpheus/objects/data_table.hpp"  // for IDataTable
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for TensorIndex

#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace morpheus {

/****** Component public implementations ******************/
/****** MessageMeta****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

class MORPHEUS_EXPORT MutableTableCtxMgr;

/**
 * @brief Container for class holding a data table, in practice a cudf DataFrame, with the ability to return both
 * Python and C++ representations of the table
 *
 */
class MORPHEUS_EXPORT MessageMeta
{
  public:
    /**
     * @brief Get the row count of the underlying DataFrame
     *
     * @return TensorIndex
     */
    virtual TensorIndex count() const;

    /**
     * @brief Get the info object
     *
     * @return TableInfo
     */
    virtual TableInfo get_info() const;

    /**
     * @brief Get the info object for a specific column
     *
     * @param col_name The name of the column to slice
     * @return TableInfo The table info containing only the column specified
     */
    virtual TableInfo get_info(const std::string& col_name) const;

    /**
     * @brief Get the info object for a specific set of columns
     *
     * @param column_names The names of the columns to slice
     * @return TableInfo The table info containing only the columns specified, in the order specified
     */
    virtual TableInfo get_info(const std::vector<std::string>& column_names) const;

    /**
     * @brief Set the data for a single column from a TensorObject
     *
     * @param col_name The name of the column to set
     * @param tensor The tensor to set the column to
     */
    virtual void set_data(const std::string& col_name, TensorObject tensor);

    /**
     * @brief Set the data for multiple columns from a vector of TensorObjects
     *
     * @param column_names The names of the columns to set
     * @param tensors The tensors to set the columns to
     */
    virtual void set_data(const std::vector<std::string>& column_names, const std::vector<TensorObject>& tensors);

    /**
     * TODO(Documentation)
     */
    virtual MutableTableInfo get_mutable_info() const;

    std::vector<std::string> get_column_names() const;

    /**
     * @brief Returns true if the underlying DataFrame's index is unique and monotonic. Sliceable indices have better
     * performance since a range of rows can be specified by a start and stop index instead of requiring boolean masks.
     *
     * @return bool
     */
    bool has_sliceable_index() const;

    /**
     * @brief Replaces the index in the underlying dataframe if the existing one is not unique and monotonic. The old
     * index will be preserved in a column named `_index_{old_index.name}`. If `has_sliceable_index() == true`, this is
     * a no-op.
     *
     * @return std::string The name of the column with the old index or nullopt if no changes were made.
     */
    virtual std::optional<std::string> ensure_sliceable_index();

    /**
     * @brief Creates a deep copy of DataFrame with the specified ranges.
     *
     * @param ranges the tensor index ranges to copy
     * @return std::shared_ptr<MessageMeta> the deep copy of the specified ranges
     */
    virtual std::shared_ptr<MessageMeta> copy_ranges(const std::vector<RangeType>& ranges) const;

    /**
     * @brief Get a slice of the underlying DataFrame by creating a deep copy
     *
     * @param start the tensor index of the start of the copy
     * @param stop the tensor index of the end of the copy
     * @return std::shared_ptr<MessageMeta> the deep copy of the speicifed slice
     */
    virtual std::shared_ptr<MessageMeta> get_slice(TensorIndex start, TensorIndex stop) const;

    /**
     * @brief Create MessageMeta cpp object from a python object
     *
     * @param data_table
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> create_from_python(pybind11::object&& data_table);

    /**
     * @brief Create MessageMeta cpp object from a cpp object, used internally by `create_from_cpp`
     *
     * @param data_table
     * @param index_col_count
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                        int index_col_count = 0);

  protected:
    MessageMeta(std::shared_ptr<IDataTable> data);

    /**
     * @brief Create MessageMeta python object from a cpp object
     *
     * @param table
     * @param index_col_count
     * @return pybind11::object
     */
    static pybind11::object cpp_to_py(cudf::io::table_with_metadata&& table, int index_col_count = 0);

    std::shared_ptr<IDataTable> m_data;
};

/**
 * @brief Operates similarly to MessageMeta, except it applies a filter on the columns and rows. Used by Serialization
 * to filter columns without copying the entire DataFrame
 *
 */
class MORPHEUS_EXPORT SlicedMessageMeta : public MessageMeta
{
  public:
    SlicedMessageMeta(std::shared_ptr<MessageMeta> other,
                      TensorIndex start                = 0,
                      TensorIndex stop                 = -1,
                      std::vector<std::string> columns = {});

    TensorIndex count() const override;

    TableInfo get_info() const override;

    TableInfo get_info(const std::string& col_name) const override;

    TableInfo get_info(const std::vector<std::string>& column_names) const override;

    MutableTableInfo get_mutable_info() const override;

    std::optional<std::string> ensure_sliceable_index() override;

  private:
    TensorIndex m_start{0};
    TensorIndex m_stop{-1};
    std::vector<std::string> m_column_names;
};

/****** Python Interface **************************/
/****** MessageMetaInterfaceProxy**************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT MessageMetaInterfaceProxy
{
    /**
     * @brief Initialize MessageMeta cpp object with the given filename
     *
     * @param filename : Filename for loading the data on to MessageMeta
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> init_cpp(const std::string& filename);

    /**
     * @brief Initialize MessageMeta cpp object with a given dataframe and returns shared pointer as the result
     *
     * @param data_frame : Dataframe that contains the data
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> init_python(pybind11::object&& data_frame);

    /**
     * @brief Initialize MessageMeta cpp object with a given a MessageMeta python objectand returns shared pointer as
     * the result
     *
     * @param meta : Python MesageMeta object
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> init_python_meta(const pybind11::object& meta);

    /**
     * @brief Get messages count
     *
     * @param self
     * @return TensorIndex
     */
    static TensorIndex count(MessageMeta& self);

    /**
     * @brief Gets a DataFrame for all columns
     *
     * @param self The MessageMeta instance
     * @return pybind11::object A python DataFrame containing the info for all columns
     */
    static pybind11::object get_data(MessageMeta& self);

    /**
     * @brief Get a Series for a single column
     *
     * @param self The MessageMeta instance
     * @param col_name The name of the column to get
     * @return pybind11::object A python Series containing the info for the specified column
     */
    static pybind11::object get_data(MessageMeta& self, std::string col_name);

    /**
     * @brief Get a DataFrame for a set of columns
     *
     * @param self The MessageMeta instance
     * @param columns The names of the columns to get
     * @return pybind11::object A python DataFrame containing the info for the specified columns, in the order specified
     */
    static pybind11::object get_data(MessageMeta& self, std::vector<std::string> columns);

    /**
     * @brief Gets a DataFrame for all columns. This is only used for overload resolution from python
     *
     * @param self The MessageMeta instance
     * @param none_obj An object of None
     * @return pybind11::object A python DataFrame containing the info for all columns
     */
    static pybind11::object get_data(MessageMeta& self, pybind11::none none_obj);

    /**
     * @brief Set the values for one or more columns from a python object
     *
     * @param self The MessageMeta instance
     * @param columns The names of the columns to set
     * @param value The value to set the columns to. This can be a scalar, a list, a numpy array, a Series, or a
     * DataFrame. The dimension must match the number of columns according to DataFrame broadcasting rules.
     */
    static void set_data(MessageMeta& self, pybind11::object columns, pybind11::object value);

    static std::vector<std::string> get_column_names(MessageMeta& self);

    /**
     * @brief Get a copy of the data frame object as a python object
     *
     * @param self The MessageMeta instance
     * @return pybind11::object A `DataFrame` object
     */
    static pybind11::object get_data_frame(MessageMeta& self);

    static pybind11::object df_property(MessageMeta& self);

    static MutableTableCtxMgr mutable_dataframe(MessageMeta& self);

    /**
     * @brief Returns true if the underlying DataFrame's index is unique and monotonic. Sliceable indices have better
     * performance since a range of rows can be specified by a start and stop index instead of requiring boolean masks.
     *
     * @return bool
     */
    static bool has_sliceable_index(MessageMeta& self);

    /**
     * @brief Replaces the index in the underlying dataframe if the existing one is not unique and monotonic. The old
     * index will be preserved in a column named `_index_{old_index.name}`. If `has_sliceable_index() == true`, this is
     * a no-op.
     *
     * @return std::string The name of the column with the old index or nullopt if no changes were made.
     */
    static std::optional<std::string> ensure_sliceable_index(MessageMeta& self);

    /**
     * @brief Creates a deep copy of DataFrame with the specified ranges.
     *
     * @param ranges the tensor index ranges to copy
     * @return std::shared_ptr<MessageMeta> the deep copy of the specified ranges
     */
    static std::shared_ptr<MessageMeta> copy_ranges(MessageMeta& self, const std::vector<RangeType>& ranges);

    /**
     * @brief Get a slice of the underlying DataFrame by creating a deep copy
     *
     * @param start the tensor index of the start of the copy
     * @param stop the tensor index of the end of the copy
     * @return std::shared_ptr<MessageMeta> the deep copy of the speicifed slice
     */
    static std::shared_ptr<MessageMeta> get_slice(MessageMeta& self, TensorIndex start, TensorIndex stop);
};
/** @} */  // end of group
}  // namespace morpheus
