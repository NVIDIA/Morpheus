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

#include "morpheus/messages/meta.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/objects/dtype.hpp"  // for DType
#include "morpheus/objects/mutable_table_ctx_mgr.hpp"
#include "morpheus/objects/python_data_table.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <cuda_runtime.h>               // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyKind
#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>  // for table_view
#include <cudf/types.hpp>             // for type_id, data_type, size_type
#include <glog/logging.h>
#include <mrc/cuda/common.hpp>  // for __check_cuda_errors, MRC_CHECK_CUDA
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pyerrors.h>  // for PyExc_DeprecationWarning
#include <warnings.h>  // for PyErr_WarnEx

#include <cstddef>  // for size_t
#include <cstdint>  // for uint8_t
#include <memory>
#include <optional>
#include <ostream>        // for operator<< needed by glog
#include <stdexcept>      // for runtime_error
#include <tuple>          // for make_tuple, tuple
#include <unordered_map>  // for unordered_map
#include <utility>
// We're already including pybind11.h and don't need to include cast.
// For some reason IWYU also thinks we need array for the `isinsance` call.
// IWYU pragma: no_include <pybind11/cast.h>
// IWYU pragma: no_include <array>

namespace morpheus {

namespace py = pybind11;
using namespace py::literals;

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

TableInfo MessageMeta::get_info(const std::string& col_name) const
{
    auto full_info = this->m_data->get_info();

    return full_info.get_slice(0, full_info.num_rows(), {col_name});
}

TableInfo MessageMeta::get_info(const std::vector<std::string>& column_names) const
{
    auto full_info = this->m_data->get_info();

    return full_info.get_slice(0, full_info.num_rows(), column_names);
}

void MessageMeta::set_data(const std::string& col_name, TensorObject tensor)
{
    this->set_data({col_name}, std::vector<TensorObject>{tensor});
}

void MessageMeta::set_data(const std::vector<std::string>& column_names, const std::vector<TensorObject>& tensors)
{
    CHECK_EQ(column_names.size(), tensors.size()) << "Column names and tensors must be the same size";

    TableInfo table_meta;
    try
    {
        table_meta = this->get_info(column_names);
    } catch (const std::runtime_error& e)
    {
        std::ostringstream err_msg;
        err_msg << e.what() << " Ensure that the stage that needs this column has populated the '_needed_columns' "
                << "attribute and that at least one stage in the current segment is using the PreallocatorMixin to "
                << "ensure all needed columns have been allocated.";
        throw std::runtime_error(err_msg.str());
    }

    for (std::size_t i = 0; i < tensors.size(); ++i)
    {
        const auto& cv            = table_meta.get_column(i);
        const auto table_type_id  = cv.type().id();
        const auto tensor_type    = DType(tensors[i].dtype());
        const auto tensor_type_id = tensor_type.cudf_type_id();
        const auto row_stride     = tensors[i].stride(0);
        CHECK(tensors[i].count() == cv.size() &&
              (table_type_id == tensor_type_id ||
               (table_type_id == cudf::type_id::BOOL8 && tensor_type_id == cudf::type_id::UINT8)));
        const auto item_size = tensors[i].dtype().item_size();

        // Dont use cv.data<>() here since that does not account for the size of each element
        auto data_start = const_cast<uint8_t*>(cv.head<uint8_t>()) + cv.offset() * item_size;
        if (row_stride == 1)
        {
            // column major just use cudaMemcpy
            MRC_CHECK_CUDA(cudaMemcpy(data_start, tensors[i].data(), tensors[i].bytes(), cudaMemcpyDeviceToDevice));
        }
        else
        {
            MRC_CHECK_CUDA(cudaMemcpy2D(data_start,
                                        item_size,
                                        tensors[i].data(),
                                        row_stride * item_size,
                                        item_size,
                                        cv.size(),
                                        cudaMemcpyDeviceToDevice));
        }
    }
}

MutableTableInfo MessageMeta::get_mutable_info() const
{
    return this->m_data->get_mutable_info();
}

std::vector<std::string> MessageMeta::get_column_names() const
{
    return m_data->get_info().get_column_names();
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

std::shared_ptr<MessageMeta> MessageMeta::copy_ranges(const std::vector<RangeType>& ranges) const
{
    // copy ranges into a sequntial list of values
    // https://github.com/rapidsai/cudf/issues/11223
    std::vector<TensorIndex> cudf_ranges;
    for (const auto& p : ranges)
    {
        // Append the message offset to the range here
        cudf_ranges.push_back(p.first);
        cudf_ranges.push_back(p.second);
    }
    auto table_info   = this->get_info();
    auto column_names = table_info.get_column_names();
    auto metadata     = cudf::io::table_metadata{};

    metadata.schema_info.reserve(column_names.size() + 1);
    metadata.schema_info.emplace_back("");

    for (auto column_name : column_names)
    {
        metadata.schema_info.emplace_back(column_name);
    }

    auto table_view                     = table_info.get_view();
    auto sliced_views                   = cudf::slice(table_view, cudf_ranges);
    cudf::io::table_with_metadata table = {cudf::concatenate(sliced_views), std::move(metadata)};

    return MessageMeta::create_from_cpp(std::move(table), 1);
}

std::shared_ptr<MessageMeta> MessageMeta::get_slice(TensorIndex start, TensorIndex stop) const
{
    return this->copy_ranges({{start, stop}});
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
        // Check if we received a Pandas DF or the Python impl of MessageMeta, throw an error otherwise
        auto pd_df_cls = py::module_::import("pandas").attr("DataFrame");
        if (py::isinstance(data_frame, pd_df_cls))
        {
            LOG(WARNING) << "Dataframe is not a cudf dataframe, converting to cudf dataframe";
            data_frame = cudf_df_cls(std::move(data_frame));
        }
        else
        {
            // check to see if its a Python MessageMeta object
            auto msg_meta_cls = py::module_::import("morpheus.messages").attr("MessageMeta");
            if (py::isinstance(data_frame, msg_meta_cls))
            {
                DVLOG(10) << "Converting from a Python impl of MessageMeta to C++ impl";
                return init_python_meta(data_frame);
            }
            else
            {
                throw pybind11::value_error("Dataframe is not a cudf or pandas dataframe");
            }
        }
    }

    return MessageMeta::create_from_python(std::move(data_frame));
}

std::shared_ptr<MessageMeta> MessageMetaInterfaceProxy::init_python_meta(const py::object& meta)
{
    // check to see if its a Python MessageMeta object
    auto msg_meta_cls = py::module_::import("morpheus.messages").attr("MessageMeta");
    if (py::isinstance(meta, msg_meta_cls))
    {
        DVLOG(10) << "Converting Python impl of MessageMeta to C++ impl";
        return init_python(meta.attr("copy_dataframe")());
    }
    else
    {
        throw pybind11::value_error("meta is not a Python instance of MessageMeta");
    }
}

TensorIndex MessageMetaInterfaceProxy::count(MessageMeta& self)
{
    return self.count();
}

pybind11::object MessageMetaInterfaceProxy::get_data(MessageMeta& self)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_info();

    // Convert to a python datatable. Automatically gets the GIL
    return CudfHelper::table_from_table_info(info);
}

pybind11::object MessageMetaInterfaceProxy::get_data(MessageMeta& self, std::string col_name)
{
    TableInfo info;

    {
        // Need to release the GIL before calling `get_meta()`
        pybind11::gil_scoped_release no_gil;

        // Get the column and convert to cudf
        info = self.get_info(col_name);
    }

    auto py_table = CudfHelper::table_from_table_info(info);

    // Now convert it to a series by selecting only the column
    return py_table[col_name.c_str()];
}

pybind11::object MessageMetaInterfaceProxy::get_data(MessageMeta& self, std::vector<std::string> columns)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_info(columns);

    // Convert to a python datatable. Automatically gets the GIL
    return CudfHelper::table_from_table_info(info);
}

pybind11::object MessageMetaInterfaceProxy::get_data(MessageMeta& self, pybind11::none none_obj)
{
    // Just offload to the overload without columns. This overload is needed to match the python interface
    return MessageMetaInterfaceProxy::get_data(self);
}

std::tuple<py::object, py::object> get_indexers(MessageMeta& self,
                                                py::object df,
                                                py::object columns,
                                                cudf::size_type num_rows)
{
    auto row_indexer = pybind11::slice(pybind11::int_(0), pybind11::int_(num_rows), pybind11::none());

    if (columns.is_none())
    {
        columns = df.attr("columns").attr("to_list")();
    }
    else if (pybind11::isinstance<pybind11::str>(columns))
    {
        // Convert a single string into a list so all versions return tables, not series
        pybind11::list col_list;

        col_list.append(columns);

        columns = std::move(col_list);
    }

    auto column_indexer = df.attr("columns").attr("get_indexer_for")(columns);

    return std::make_tuple(row_indexer, column_indexer);
}

void MessageMetaInterfaceProxy::set_data(MessageMeta& self, pybind11::object columns, pybind11::object value)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    auto mutable_info = self.get_mutable_info();
    auto num_rows     = mutable_info.num_rows();

    // Need the GIL for the remainder
    pybind11::gil_scoped_acquire gil;

    auto pdf = mutable_info.checkout_obj();
    auto& df = *pdf;

    auto [row_indexer, column_indexer] = get_indexers(self, df, columns, num_rows);

    // Check to see if this is adding a column. If so, we need to use .loc instead of .iloc
    if (column_indexer.contains(-1))
    {
        // cudf is really bad at adding new columns. Need to use loc with a unique and monotonic index
        py::object saved_index = df.attr("index");

        // Check to see if we can use slices
        if (!(saved_index.attr("is_unique").cast<bool>() && (saved_index.attr("is_monotonic_increasing").cast<bool>() ||
                                                             saved_index.attr("is_monotonic_decreasing").cast<bool>())))
        {
            df.attr("reset_index")("drop"_a = true, "inplace"_a = true);
        }
        else
        {
            // Erase the saved index so we dont reset it
            saved_index = py::none();
        }

        // Perform the update via slices
        df.attr("loc")[pybind11::make_tuple(df.attr("index")[row_indexer], columns)] = value;

        // Reset the index if we changed it
        if (!saved_index.is_none())
        {
            df.attr("set_index")(saved_index, "inplace"_a = true);
        }
    }
    else
    {
        // If we only have one column, convert it to a series (broadcasts work with more types on a series)
        if (pybind11::len(column_indexer) == 1)
        {
            column_indexer = column_indexer.cast<py::list>()[0];
        }

        try
        {
            // Use iloc
            df.attr("iloc")[pybind11::make_tuple(row_indexer, column_indexer)] = value;
        } catch (py::error_already_set)
        {
            // Try this as a fallback. Works better for strings. See issue #286
            df[columns].attr("iloc")[row_indexer] = value;
        }
    }

    mutable_info.return_obj(std::move(pdf));
}

std::vector<std::string> MessageMetaInterfaceProxy::get_column_names(MessageMeta& self)
{
    pybind11::gil_scoped_release no_gil;
    return self.get_column_names();
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

std::shared_ptr<MessageMeta> MessageMetaInterfaceProxy::copy_ranges(MessageMeta& self,
                                                                    const std::vector<RangeType>& ranges)
{
    pybind11::gil_scoped_release no_gil;

    return self.copy_ranges(ranges);
}

std::shared_ptr<MessageMeta> MessageMetaInterfaceProxy::get_slice(MessageMeta& self,
                                                                  TensorIndex start,
                                                                  TensorIndex stop)
{
    pybind11::gil_scoped_release no_gil;

    return self.get_slice(start, stop);
}

SlicedMessageMeta::SlicedMessageMeta(std::shared_ptr<MessageMeta> other,
                                     TensorIndex start,
                                     TensorIndex stop,
                                     std::vector<std::string> columns) :
  MessageMeta(*other),
  m_start(start),
  m_stop(stop),
  m_column_names(std::move(columns))
{
    auto sliced_other = std::dynamic_pointer_cast<SlicedMessageMeta>(other);
    if (sliced_other)
    {
        m_start += sliced_other->m_start;
        m_stop += sliced_other->m_start;
    }
}

TensorIndex SlicedMessageMeta::count() const
{
    return m_stop - m_start;
}

TableInfo SlicedMessageMeta::get_info() const
{
    return get_info(m_column_names);
}

TableInfo SlicedMessageMeta::get_info(const std::string& col_name) const
{
    return get_info(std::vector<std::string>{{col_name}});
}

TableInfo SlicedMessageMeta::get_info(const std::vector<std::string>& column_names) const
{
    auto full_info = this->m_data->get_info();

    return full_info.get_slice(m_start, m_stop, column_names);
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
