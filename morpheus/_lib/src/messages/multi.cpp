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

#include "morpheus/messages/multi.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/dtype.hpp"  // for TypeId, DType
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <cuda_runtime.h>               // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyDeviceToDevice
#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>       // for CHECK
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rmm/mr/device/per_device_resource.hpp>  // for get_current_device_resource

#include <algorithm>  // for transform
#include <array>      // needed for pybind11::make_tuple
#include <cstddef>    // for size_t
#include <cstdint>    // for uint8_t
#include <sstream>
#include <stdexcept>  // for runtime_error
#include <tuple>
#include <type_traits>
#include <utility>
// IWYU pragma: no_include <unordered_map>

namespace morpheus {

namespace py = pybind11;
using namespace py::literals;

/****** Component public implementations *******************/
/****** MultiMessage****************************************/
MultiMessage::MultiMessage(std::shared_ptr<MessageMeta> meta, TensorIndex offset, TensorIndex count) :
  meta(std::move(meta)),
  mess_offset(offset)
{
    if (!this->meta)
    {
        throw std::invalid_argument("Must define `meta` when creating MultiMessage");
    }

    // Default to using the count from the meta if it is unset
    if (count == -1)
    {
        count = this->meta->count() - offset;
    }

    this->mess_count = count;

    if (this->mess_offset < 0 || this->mess_offset >= this->meta->count())
    {
        throw std::invalid_argument("Invalid message offset value");
    }
    if (this->mess_count <= 0 || (this->mess_offset + this->mess_count > this->meta->count()))
    {
        throw std::invalid_argument("Invalid message count value");
    }
}

TableInfo MultiMessage::get_meta()
{
    auto table_info = this->get_meta(std::vector<std::string>{});

    return table_info;
}

TableInfo MultiMessage::get_meta(const std::string& col_name)
{
    auto table_view = this->get_meta(std::vector<std::string>{col_name});

    return table_view;
}

TableInfo MultiMessage::get_meta(const std::vector<std::string>& column_names)
{
    TableInfo info = this->meta->get_info();

    TableInfo sliced_info = info.get_slice(this->mess_offset,
                                           this->mess_offset + this->mess_count,
                                           column_names.empty() ? info.get_column_names() : column_names);

    return sliced_info;
}

void MultiMessage::get_slice_impl(std::shared_ptr<MultiMessage> new_message, TensorIndex start, TensorIndex stop) const
{
    // Start must be between [0, mess_count)
    if (start < 0 || start >= this->mess_count)
    {
        throw std::out_of_range("Invalid `start` argument");
    }

    // Stop must be between (start, mess_count]
    if (stop <= start or stop > this->mess_count)
    {
        throw std::out_of_range("Invalid `stop` argument");
    }

    new_message->mess_offset = this->mess_offset + start;
    new_message->mess_count  = this->mess_offset + stop - new_message->mess_offset;
}

void MultiMessage::copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                    const std::vector<RangeType>& ranges,
                                    TensorIndex num_selected_rows) const
{
    new_message->mess_offset = 0;
    new_message->mess_count  = num_selected_rows;
    new_message->meta        = copy_meta_ranges(ranges);
}

std::shared_ptr<MessageMeta> MultiMessage::copy_meta_ranges(const std::vector<RangeType>& ranges) const
{
    // copy ranges into a sequntial list of values
    // https://github.com/rapidsai/cudf/issues/11223
    std::vector<TensorIndex> cudf_ranges;
    for (const auto& p : ranges)
    {
        // Append the message offset to the range here
        cudf_ranges.push_back(p.first + this->mess_offset);
        cudf_ranges.push_back(p.second + this->mess_offset);
    }

    auto table_info   = this->meta->get_info();
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
    cudf::io::table_with_metadata table = {cudf::concatenate(sliced_views, rmm::mr::get_current_device_resource()),
                                           std::move(metadata)};

    return MessageMeta::create_from_cpp(std::move(table), 1);
}

void MultiMessage::set_meta(const std::string& col_name, TensorObject tensor)
{
    set_meta(std::vector<std::string>{col_name}, std::vector<TensorObject>{tensor});
}

void MultiMessage::set_meta(const std::vector<std::string>& column_names, const std::vector<TensorObject>& tensors)
{
    TableInfo table_meta;
    try
    {
        table_meta = this->get_meta(column_names);
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

std::vector<RangeType> MultiMessage::apply_offset_to_ranges(TensorIndex offset,
                                                            const std::vector<RangeType>& ranges) const
{
    std::vector<RangeType> offset_ranges(ranges.size());
    std::transform(ranges.cbegin(), ranges.cend(), offset_ranges.begin(), [offset](const RangeType range) {
        return std::pair{offset + range.first, offset + range.second};
    });

    return offset_ranges;
}

/****** MultiMessageInterfaceProxy *************************/
std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::init(std::shared_ptr<MessageMeta> meta,
                                                               TensorIndex mess_offset,
                                                               TensorIndex mess_count)
{
    return std::make_shared<MultiMessage>(std::move(meta), mess_offset, mess_count);
}

std::shared_ptr<morpheus::MessageMeta> MultiMessageInterfaceProxy::meta(const MultiMessage& self)
{
    return self.meta;
}

TensorIndex MultiMessageInterfaceProxy::mess_offset(const MultiMessage& self)
{
    return self.mess_offset;
}

TensorIndex MultiMessageInterfaceProxy::mess_count(const MultiMessage& self)
{
    return self.mess_count;
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage& self)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_meta();

    // Convert to a python datatable. Automatically gets the GIL
    return CudfHelper::table_from_table_info(info);
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage& self, std::string col_name)
{
    TableInfo info;

    {
        // Need to release the GIL before calling `get_meta()`
        pybind11::gil_scoped_release no_gil;

        // Get the column and convert to cudf
        info = self.get_meta();
    }

    auto py_table = CudfHelper::table_from_table_info(info);

    // Now convert it to a series by selecting only the column
    return py_table[col_name.c_str()];
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage& self, std::vector<std::string> columns)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_meta(columns);

    // Convert to a python datatable. Automatically gets the GIL
    return CudfHelper::table_from_table_info(info);
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage& self, pybind11::none none_obj)
{
    // Just offload to the overload without columns. This overload is needed to match the python interface
    return MultiMessageInterfaceProxy::get_meta(self);
}

pybind11::object MultiMessageInterfaceProxy::get_meta_list(MultiMessage& self, pybind11::object col_name)
{
    std::vector<std::string> column_names;
    if (!col_name.is_none())
    {
        column_names.emplace_back(col_name.cast<std::string>());
    }

    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    auto info = self.get_meta(column_names);

    // Need the GIL for the remainder
    pybind11::gil_scoped_acquire gil;

    auto meta = CudfHelper::table_from_table_info(info);

    if (!col_name.is_none())
    {  // needed to slice off the id column
        meta = meta[col_name];
    }

    auto arrow_tbl           = meta.attr("to_arrow")();
    pybind11::object py_list = arrow_tbl.attr("to_pylist")();

    return py_list;
}

std::tuple<py::object, py::object> get_indexers(MultiMessage& self, py::object df, py::object columns)
{
    auto row_indexer = pybind11::slice(
        pybind11::int_(self.mess_offset), pybind11::int_(self.mess_offset + self.mess_count), pybind11::none());

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

void MultiMessageInterfaceProxy::set_meta(MultiMessage& self, pybind11::object columns, pybind11::object value)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    auto mutable_info = self.meta->get_mutable_info();

    // Need the GIL for the remainder
    pybind11::gil_scoped_acquire gil;

    auto df = mutable_info.checkout_obj();

    auto [row_indexer, column_indexer] = get_indexers(self, df, columns);

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

    mutable_info.return_obj(std::move(df));
}

std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::get_slice(MultiMessage& self,
                                                                    TensorIndex start,
                                                                    TensorIndex stop)
{
    if (start < 0)
    {
        throw std::out_of_range("Invalid message `start` argument");
    }

    if (stop < 0)
    {
        throw std::out_of_range("Invalid message `stop` argument");
    }

    // Need to drop the GIL before calling any methods on the C++ object
    pybind11::gil_scoped_release no_gil;

    // Returns shared_ptr
    return self.get_slice(start, stop);
}

std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::copy_ranges(MultiMessage& self,
                                                                      const std::vector<RangeType>& ranges,
                                                                      pybind11::object num_selected_rows)
{
    TensorIndex num_rows = 0;
    if (num_selected_rows.is_none())
    {
        for (const auto& range : ranges)
        {
            num_rows += range.second - range.first;
        }
    }
    else
    {
        num_rows = num_selected_rows.cast<TensorIndex>();
    }

    // Need to drop the GIL before calling any methods on the C++ object
    pybind11::gil_scoped_release no_gil;

    return self.copy_ranges(ranges, num_rows);
}

pybind11::dict MultiMessageInterfaceProxy::pickle_dump(const MultiMessage& message)
{
    auto pickle           = pybind11::dict();
    pickle["meta"]        = MessageMetaInterfaceProxy::pickle_dump(*message.meta);
    pickle["mess_offset"] = message.mess_offset;
    pickle["mess_count"]  = message.mess_count;
    return pickle;
}

MultiMessage MultiMessageInterfaceProxy::pickle_load(pybind11::dict pickle)
{
    auto mess_count  = pickle["mess_count"].cast<TensorIndex>();
    auto mess_offset = pickle["mess_offset"].cast<TensorIndex>();
    auto meta        = MessageMetaInterfaceProxy::pickle_load(pickle["meta"]);
    auto meta_shared = std::make_shared<MessageMeta>(std::move(meta));
    return {std::move(meta_shared), mess_offset, mess_count};
}

}  // namespace morpheus
