/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/data_table.hpp"  // for IDataTable
#include "morpheus/objects/table_info.hpp"
#include "morpheus/types.hpp"  // for TensorIndex

#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace morpheus {
#pragma GCC visibility push(default)
/****** Component public implementations ******************/
/****** MessageMeta****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

class MutableTableCtxMgr;

/**
 * @brief Container for class holding a data table, in practice a cudf DataFrame, with the ability to return both
 * Python and C++ representations of the table
 *
 */
class MessageMeta
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
     * TODO(Documentation)
     */
    virtual MutableTableInfo get_mutable_info() const;

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

    std::shared_ptr<IDataTable> m_data;

    MessageMeta(std::shared_ptr<IDataTable> data);

  protected:
    /**
     * @brief Create MessageMeta python object from a cpp object
     *
     * @param table
     * @param index_col_count
     * @return pybind11::object
     */
    static pybind11::object cpp_to_py(cudf::io::table_with_metadata&& table, int index_col_count = 0);
};

/**
 * @brief Operates similarly to MessageMeta, except it applies a filter on the columns and rows. Used by Serialization
 * to filter columns without copying the entire DataFrame
 *
 */
class SlicedMessageMeta : public MessageMeta
{
  public:
    SlicedMessageMeta(std::shared_ptr<MessageMeta> other,
                      TensorIndex start                = 0,
                      TensorIndex stop                 = -1,
                      std::vector<std::string> columns = {});

    TensorIndex count() const override;

    TableInfo get_info() const override;

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
struct MessageMetaInterfaceProxy
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
     * @brief Get messages count
     *
     * @param self
     * @return TensorIndex
     */
    static TensorIndex count(MessageMeta& self);

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

    static pybind11::dict pickle_dump(const MessageMeta& meta);

    static MessageMeta pickle_load(pybind11::dict pickle);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
