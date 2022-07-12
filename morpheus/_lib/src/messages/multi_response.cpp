/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/messages/multi_response.hpp>

#include <morpheus/messages/memory/response_memory.hpp>
#include <morpheus/messages/meta.hpp>
#include <morpheus/messages/multi.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/utilities/cupy_util.hpp>
#include <morpheus/utilities/tensor_util.hpp>

#include <cudf/types.hpp>

#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include "morpheus/objects/tensor_object.hpp"

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseMessage****************************************/
MultiResponseMessage::MultiResponseMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                                           std::size_t mess_offset,
                                           std::size_t mess_count,
                                           std::shared_ptr<ResponseMemory> memory,
                                           std::size_t offset,
                                           std::size_t count) :
  MultiMessage(meta, mess_offset, mess_count),
  memory(std::move(memory)),
  offset(offset),
  count(count)
{}

TensorObject MultiResponseMessage::get_output(const std::string &name)
{
    CHECK(this->memory->has_output(name)) << "Could not find output: " << name;

    // check if we are getting the entire input
    if (this->offset == 0 && this->count == this->memory->count)
    {
        return this->memory->outputs[name];
    }

    // TODO(MDD): This really needs to return the slice of the tensor
    return this->memory->outputs[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                             {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

const TensorObject MultiResponseMessage::get_output(const std::string &name) const
{
    CHECK(this->memory->has_output(name)) << "Could not find output: " << name;

    // check if we are getting the entire input
    if (this->offset == 0 && this->count == this->memory->count)
    {
        return this->memory->outputs[name];
    }

    // TODO(MDD): This really needs to return the slice of the tensor
    return this->memory->outputs[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                             {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

const void MultiResponseMessage::set_output(const std::string &name, const TensorObject &value)
{
    // Get the input slice first
    auto slice = this->get_output(name);

    // Set the value to use assignment
    slice = value;
}

std::shared_ptr<MultiResponseMessage> MultiResponseMessage::get_slice(std::size_t start, std::size_t stop) const
{
    // This can only cast down
    return std::static_pointer_cast<MultiResponseMessage>(this->internal_get_slice(start, stop));
}

std::shared_ptr<MultiMessage> MultiResponseMessage::internal_get_slice(std::size_t start, std::size_t stop) const
{
    CHECK(this->mess_count == this->count) << "At this time, mess_count and count must be the same for slicing";

    auto mess_start = this->mess_offset + start;
    auto mess_stop  = this->mess_offset + stop;

    return std::make_shared<MultiResponseMessage>(
        this->meta, mess_start, mess_stop - mess_start, this->memory, start, stop - start);
}

std::shared_ptr<MultiResponseMessage> MultiResponseMessage::copy_ranges(
    const std::vector<std::pair<size_t, size_t>> &ranges, size_t num_selected_rows) const
{
    return std::static_pointer_cast<MultiResponseMessage>(this->internal_copy_ranges(ranges, num_selected_rows));
}

std::shared_ptr<MultiMessage> MultiResponseMessage::internal_copy_ranges(
    const std::vector<std::pair<size_t, size_t>> &ranges, size_t num_selected_rows) const
{
    auto msg_meta = copy_meta_ranges(ranges);
    auto mem      = copy_output_ranges(ranges, num_selected_rows);
    return std::make_shared<MultiResponseMessage>(msg_meta, 0, num_selected_rows, mem, 0, num_selected_rows);
}

std::shared_ptr<ResponseMemory> MultiResponseMessage::copy_output_ranges(
    const std::vector<std::pair<size_t, size_t>> &ranges, size_t num_selected_rows) const
{
    const auto offset = static_cast<TensorIndex>(this->offset);
    std::vector<std::pair<TensorIndex, TensorIndex>> offset_ranges(ranges.size());
    std::transform(
        ranges.cbegin(), ranges.cend(), offset_ranges.begin(), [offset](const std::pair<size_t, size_t> range) {
            return std::pair{offset + range.first, offset + range.second};
        });

    std::map<std::string, TensorObject> output_tensors;
    for (const auto &mem_output : memory->outputs)
    {
        // A little confusing here, but the response outputs are the inputs for this method
        const std::string &output_name   = mem_output.first;
        const TensorObject &input_tensor = mem_output.second;

        output_tensors.insert(std::pair{output_name, input_tensor.copy_rows(offset_ranges, num_selected_rows)});
    }

    return std::make_shared<ResponseMemory>(num_selected_rows, std::move(output_tensors));
}

/****** MultiResponseMessageInterfaceProxy *************************/
std::shared_ptr<MultiResponseMessage> MultiResponseMessageInterfaceProxy::init(std::shared_ptr<MessageMeta> meta,
                                                                               cudf::size_type mess_offset,
                                                                               cudf::size_type mess_count,
                                                                               std::shared_ptr<ResponseMemory> memory,
                                                                               cudf::size_type offset,
                                                                               cudf::size_type count)
{
    return std::make_shared<MultiResponseMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

std::shared_ptr<morpheus::ResponseMemory> MultiResponseMessageInterfaceProxy::memory(MultiResponseMessage &self)
{
    return self.memory;
}

std::size_t MultiResponseMessageInterfaceProxy::offset(MultiResponseMessage &self)
{
    return self.offset;
}

std::size_t MultiResponseMessageInterfaceProxy::count(MultiResponseMessage &self)
{
    return self.count;
}

pybind11::object MultiResponseMessageInterfaceProxy::get_output(MultiResponseMessage &self, const std::string &name)
{
    auto tensor = self.get_output(name);

    return CupyUtil::tensor_to_cupy(tensor);
}
}  // namespace morpheus
