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

#include "morpheus/messages/meta.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/objects/mutable_table_ctx_mgr.hpp"
#include "morpheus/objects/python_data_table.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <cudf/io/types.hpp>
#include <glog/logging.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pyerrors.h>  // for PyExc_DeprecationWarning
#include <warnings.h>  // for PyErr_WarnEx

#include <memory>
#include <optional>
#include <ostream>    // for operator<< needed by glog
#include <stdexcept>  // for runtime_error
#include <utility>
// We're already including pybind11.h and don't need to include cast.
// For some reason IWYU also thinks we need array for the `isinsance` call.
// IWYU pragma: no_include <pybind11/cast.h>
// IWYU pragma: no_include <array>

namespace morpheus {

namespace py = pybind11;

/****** Component public implementations *******************/
/****** MessageMeta ****************************************/

TensorIndex MessageMeta::count() const
{
    return m_data->count();
}

TableInfo MessageMeta::get_info() const
{
    return this->m_data->get_info();
}

MutableTableInfo MessageMeta::get_mutable_info() const
{
    return this->m_data->get_mutable_info();
}

std::shared_ptr<MessageMeta> MessageMeta::create_from_python(py::object&& data_table)
{
    auto data = std::make_unique<PyDataTable>(std::move(data_table));

    return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data)));
}

std::shared_ptr<MessageMeta> MessageMeta::create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                          int index_col_count)
{
    // Convert to py first
    py::object py_dt = cpp_to_py(std::move(data_table), index_col_count);

    auto data = std::make_unique<PyDataTable>(std::move(py_dt));

    return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data)));
}

MessageMeta::MessageMeta(std::shared_ptr<IDataTable> data) : m_data(std::move(data)) {}

py::object MessageMeta::cpp_to_py(cudf::io::table_with_metadata&& table, int index_col_count)
{
    py::gil_scoped_acquire gil;

    // Now convert to a python TableInfo object
    auto converted_table = CudfHelper::table_from_table_with_metadata(std::move(table), index_col_count);

    // VLOG(10) << "Table. Num Col: " << converted_table.attr("_num_columns").str().cast<std::string>()
    //          << ", Num Ind: " << converted_table.attr("_num_columns").cast<std::string>()
    //          << ", Rows: " << converted_table.attr("_num_rows").cast<std::string>();
    // py::print("Table Created. Num Rows: {}, Num Cols: {}, Num Ind: {}",
    //           converted_table.attr("_num_rows"),
    //           converted_table.attr("_num_columns"),
    //           converted_table.attr("_num_indices"));

    return converted_table;
}

bool MessageMeta::has_sliceable_index() const
{
    const auto table = get_info();
    return table.has_sliceable_index();
}

std::optional<std::string> MessageMeta::ensure_sliceable_index()
{
    auto table = this->get_mutable_info();

    // Check to ensure we do (or still do) have a non-unique index. Presumably the caller already made a call to
    // `has_sliceable_index` but there could have been a race condition between the first call to has_sliceable_index
    // and the acquisition of the mutex. Re-check here to ensure some other thread didn't already fix the index
    if (!table.has_sliceable_index())
    {
        LOG(WARNING) << "Non unique index found in dataframe, generating new index.";
        return table.ensure_sliceable_index();
    }

    return std::nullopt;
}

/********** MessageMetaInterfaceProxy **********/
std::shared_ptr<MessageMeta> MessageMetaInterfaceProxy::init_python(py::object&& data_frame)
{
    // ensure we have a cudf DF and not a pandas DF
    auto cudf_df_cls = py::module_::import("cudf").attr("DataFrame");
    if (!py::isinstance(data_frame, cudf_df_cls))
    {
        // Convert to cudf if it's a Pandas DF, thrown an error otherwise
        auto pd_df_cls = py::module_::import("pandas").attr("DataFrame");
        if (py::isinstance(data_frame, pd_df_cls))
        {
            LOG(WARNING) << "Dataframe is not a cudf dataframe, converting to cudf dataframe";
            data_frame = cudf_df_cls(std::move(data_frame));
        }
        else
        {
            throw pybind11::value_error("Dataframe is not a cudf or pandas dataframe");
        }
    }

    return MessageMeta::create_from_python(std::move(data_frame));
}

TensorIndex MessageMetaInterfaceProxy::count(MessageMeta& self)
{
    return self.count();
}

py::object MessageMetaInterfaceProxy::get_data_frame(MessageMeta& self)
{
    TableInfo info;

    {
        // Need to release the GIL before calling `get_meta()`
        pybind11::gil_scoped_release no_gil;

        // Get the column and convert to cudf
        info = self.get_info();
    }

    return CudfHelper::table_from_table_info(info);
}

py::object MessageMetaInterfaceProxy::df_property(MessageMeta& self)
{
    PyErr_WarnEx(
        PyExc_DeprecationWarning,
        "Warning the df property returns a copy, please use the copy_dataframe method or the mutable_dataframe "
        "context manager to modify the DataFrame in-place instead.",
        1);

    return MessageMetaInterfaceProxy::get_data_frame(self);
}

MutableTableCtxMgr MessageMetaInterfaceProxy::mutable_dataframe(MessageMeta& self)
{
    // Release any GIL
    py::gil_scoped_release no_gil;
    return {self};
}

std::shared_ptr<MessageMeta> MessageMetaInterfaceProxy::init_cpp(const std::string& filename)
{
    // Load the file
    auto df_with_meta   = load_table_from_file(filename);
    int index_col_count = prepare_df_index(df_with_meta);

    return MessageMeta::create_from_cpp(std::move(df_with_meta), index_col_count);
}

bool MessageMetaInterfaceProxy::has_sliceable_index(MessageMeta& self)
{
    // Release the GIL
    py::gil_scoped_release no_gil;
    return self.has_sliceable_index();
}

std::optional<std::string> MessageMetaInterfaceProxy::ensure_sliceable_index(MessageMeta& self)
{
    // Release the GIL
    py::gil_scoped_release no_gil;
    return self.ensure_sliceable_index();
}

pybind11::dict MessageMetaInterfaceProxy::pickle_dump(const MessageMeta& meta)
{
    auto pickle   = pybind11::dict();
    pickle["obj"] = meta.m_data->get_py_object();
    return pickle;
}

MessageMeta MessageMetaInterfaceProxy::pickle_load(pybind11::dict pickle)
{
    auto obj   = pickle["obj"].cast<py::object>();
    auto table = std::make_shared<PyDataTable>(std::move(obj));
    return {std::move(table)};
}

SlicedMessageMeta::SlicedMessageMeta(std::shared_ptr<MessageMeta> other,
                                     TensorIndex start,
                                     TensorIndex stop,
                                     std::vector<std::string> columns) :
  MessageMeta(*other),
  m_start(start),
  m_stop(stop),
  m_column_names(std::move(columns))
{}

TensorIndex SlicedMessageMeta::count() const
{
    return m_stop - m_start;
}

TableInfo SlicedMessageMeta::get_info() const
{
    return this->m_data->get_info().get_slice(m_start, m_stop, m_column_names);
}

MutableTableInfo SlicedMessageMeta::get_mutable_info() const
{
    return this->m_data->get_mutable_info().get_slice(m_start, m_stop, m_column_names);
}

std::optional<std::string> SlicedMessageMeta::ensure_sliceable_index()
{
    throw std::runtime_error{"Unable to set a new index on the DataFrame from a partial view of the columns/rows."};
}

}  // namespace morpheus
