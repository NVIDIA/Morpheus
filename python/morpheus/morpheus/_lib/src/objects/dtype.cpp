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

#include "morpheus/objects/dtype.hpp"

#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR

#include <cudf/types.hpp>

#include <map>
#include <sstream>  // Needed by MORPHEUS_CONCAT_STR
#include <stdexcept>
#include <string>
#include <utility>

namespace {
const std::map<char, std::map<size_t, morpheus::TypeId>> STR_TO_TYPE_ID = {
    {'b', {{1, morpheus::TypeId::BOOL8}}},

    {'i',
     {{1, morpheus::TypeId::INT8},
      {2, morpheus::TypeId::INT16},
      {4, morpheus::TypeId::INT32},
      {8, morpheus::TypeId::INT64}}},

    {'u',
     {
         {1, morpheus::TypeId::UINT8},
         {2, morpheus::TypeId::UINT16},
         {4, morpheus::TypeId::UINT32},
         {8, morpheus::TypeId::UINT64},
     }},

    {'f', {{4, morpheus::TypeId::FLOAT32}, {8, morpheus::TypeId::FLOAT64}}},

    {'O', {{1, morpheus::TypeId::STRING}}}};
}  // namespace

namespace morpheus {

DType::DType(TypeId tid) : m_type_id(tid) {}

bool DType::operator==(const DType& other) const
{
    return m_type_id == other.m_type_id;
}

TypeId DType::type_id() const
{
    return m_type_id;
}

size_t DType::item_size() const
{
    switch (m_type_id)
    {
    case TypeId::BOOL8:
    case TypeId::INT8:
    case TypeId::STRING:  // not sure, but size of individual char
    case TypeId::UINT8:
        return 1;
    case TypeId::INT16:
    case TypeId::UINT16:
        return 2;
    case TypeId::INT32:
    case TypeId::UINT32:
    case TypeId::FLOAT32:
        return 4;
    case TypeId::INT64:
    case TypeId::UINT64:
    case TypeId::FLOAT64:
        return 8;
    case TypeId::NUM_TYPE_IDS:
    case TypeId::EMPTY:
    default:
        throw std::invalid_argument("Unknown datatype");
    }
}

std::string DType::name() const
{
    // TODO(MDD): Replace this with a better version. For now, follow type_str
    return this->type_str();
}

std::string DType::type_str() const
{
    return MORPHEUS_CONCAT_STR(this->byte_order_char() << this->type_char() << this->item_size());
}

// Cudf representation
cudf::type_id DType::cudf_type_id() const
{
    switch (m_type_id)
    {
    case TypeId::INT8:
        return cudf::type_id::INT8;
    case TypeId::INT16:
        return cudf::type_id::INT16;
    case TypeId::INT32:
        return cudf::type_id::INT32;
    case TypeId::INT64:
        return cudf::type_id::INT64;
    case TypeId::UINT8:
        return cudf::type_id::UINT8;
    case TypeId::UINT16:
        return cudf::type_id::UINT16;
    case TypeId::UINT32:
        return cudf::type_id::UINT32;
    case TypeId::UINT64:
        return cudf::type_id::UINT64;
    case TypeId::FLOAT32:
        return cudf::type_id::FLOAT32;
    case TypeId::FLOAT64:
        return cudf::type_id::FLOAT64;
    case TypeId::BOOL8:
        return cudf::type_id::BOOL8;
    case TypeId::STRING:
        return cudf::type_id::STRING;
    case TypeId::EMPTY:
    case TypeId::NUM_TYPE_IDS:
    default:
        throw std::runtime_error("Not supported");
    }
}

// Returns the triton string representation
std::string DType::triton_str() const
{
    // Triton doesn't have any definitions or enums. Wow
    switch (m_type_id)
    {
    case TypeId::INT8:
        return "INT8";
    case TypeId::INT16:
        return "INT16";
    case TypeId::INT32:
        return "INT32";
    case TypeId::INT64:
        return "INT64";
    case TypeId::UINT8:
        return "UINT8";
    case TypeId::UINT16:
        return "UINT16";
    case TypeId::UINT32:
        return "UINT32";
    case TypeId::UINT64:
        return "UINT64";
    case TypeId::FLOAT32:
        return "FP32";
    case TypeId::FLOAT64:
        return "FP64";
    case TypeId::BOOL8:
        return "BOOL";
    case TypeId::EMPTY:
    case TypeId::NUM_TYPE_IDS:
    case TypeId::STRING:
    default:
        throw std::runtime_error("Not supported");
    }
}

// From cudf
DType DType::from_cudf(cudf::type_id tid)
{
    switch (tid)
    {
    case cudf::type_id::INT8:
        return {TypeId::INT8};
    case cudf::type_id::INT16:
        return {TypeId::INT16};
    case cudf::type_id::INT32:
        return {TypeId::INT32};
    case cudf::type_id::INT64:
        return {TypeId::INT64};
    case cudf::type_id::UINT8:
        return {TypeId::UINT8};
    case cudf::type_id::UINT16:
        return {TypeId::UINT16};
    case cudf::type_id::UINT32:
        return {TypeId::UINT32};
    case cudf::type_id::UINT64:
        return {TypeId::UINT64};
    case cudf::type_id::FLOAT32:
        return {TypeId::FLOAT32};
    case cudf::type_id::FLOAT64:
        return {TypeId::FLOAT64};
    case cudf::type_id::BOOL8:
        return {TypeId::BOOL8};
    case cudf::type_id::STRING:
        return {TypeId::STRING};
    case cudf::type_id::EMPTY:
    case cudf::type_id::NUM_TYPE_IDS:
    default:
        throw std::invalid_argument("Not supported");
    }
}

DType DType::from_numpy(const std::string& numpy_str)
{
    if (numpy_str.empty())
    {
        throw std::invalid_argument("Cannot create DataType from empty string");
    }

    char type_char    = numpy_str[0];
    size_t size_start = 1;

    // Can start with <, >, | or none
    if (numpy_str[0] == '<' || numpy_str[0] == '>' || numpy_str[0] == '|')
    {
        type_char  = numpy_str[1];
        size_start = 2;
    }

    int dtype_size = 1;
    if (numpy_str.size() > 1)
    {
        dtype_size = std::stoi(numpy_str.substr(size_start));
    }

    // Now lookup in the map
    auto found_type = STR_TO_TYPE_ID.find(type_char);

    if (found_type == STR_TO_TYPE_ID.end())
    {
        throw std::invalid_argument(MORPHEUS_CONCAT_STR("Type char '" << type_char << "' not supported"));
    }

    auto found_enum = found_type->second.find(dtype_size);

    if (found_enum == found_type->second.end())
    {
        throw std::invalid_argument(MORPHEUS_CONCAT_STR("Type str '" << type_char << dtype_size << "' not supported"));
    }

    return {found_enum->second};
}

// From triton
DType DType::from_triton(const std::string& type_str)
{
    if (type_str == "INT8")
    {
        return {TypeId::INT8};
    }
    else if (type_str == "INT16")
    {
        return {TypeId::INT16};
    }
    else if (type_str == "INT32")
    {
        return {TypeId::INT32};
    }
    else if (type_str == "INT64")
    {
        return {TypeId::INT64};
    }
    else if (type_str == "UINT8")
    {
        return {TypeId::UINT8};
    }
    else if (type_str == "UINT16")
    {
        return {TypeId::UINT16};
    }
    else if (type_str == "UINT32")
    {
        return {TypeId::UINT32};
    }
    else if (type_str == "UINT64")
    {
        return {TypeId::UINT64};
    }
    else if (type_str == "FP32")
    {
        return {TypeId::FLOAT32};
    }
    else if (type_str == "FP64")
    {
        return {TypeId::FLOAT64};
    }
    else if (type_str == "BOOL")
    {
        return {TypeId::BOOL8};
    }
    else
    {
        throw std::invalid_argument("Not supported");
    }
}

char DType::byte_order_char() const
{
    switch (m_type_id)
    {
    case TypeId::BOOL8:
    case TypeId::INT8:
    case TypeId::UINT8:
        return '|';
    case TypeId::INT16:
    case TypeId::UINT16:
    case TypeId::INT32:
    case TypeId::UINT32:
    case TypeId::INT64:
    case TypeId::UINT64:
    case TypeId::FLOAT32:
    case TypeId::FLOAT64:
        return '<';
    case TypeId::EMPTY:
    case TypeId::NUM_TYPE_IDS:
    case TypeId::STRING:
    default:
        throw std::runtime_error("Not supported");
    }
}

char DType::type_char() const
{
    switch (m_type_id)
    {
    case TypeId::INT8:
    case TypeId::INT16:
    case TypeId::INT32:
    case TypeId::INT64:
        return 'i';
    case TypeId::UINT8:
    case TypeId::UINT16:
    case TypeId::UINT32:
    case TypeId::UINT64:
        return 'u';
    case TypeId::BOOL8:
        return 'b';
    case TypeId::FLOAT32:
    case TypeId::FLOAT64:
        return 'f';
    case TypeId::STRING:
        return 'O';
    case TypeId::NUM_TYPE_IDS:
    case TypeId::EMPTY:
    default:
        throw std::invalid_argument("Unknown datatype");
    }
}

bool DType::is_fully_supported() const
{
    try
    {
        byte_order_char();
        cudf_type_id();
        item_size();
        triton_str();
        type_char();
    } catch (...)
    {
        return false;
    }

    return true;
}

}  // namespace morpheus
