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

#include <cudf/types.hpp>

#include <climits>  // for CHAR_BIT
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <string>   // for string

namespace morpheus {

/**
 * @addtogroup objects
 * @{
 * @file
 */

// Pulled from cuDF

/**
 * @brief Template function to calculate the size in bits of a given type.
 *
 * @tparam T The type to calculate the size for.
 * @return The size in bits of the given type.
 */
template <typename T>
constexpr std::size_t size_in_bits()
{
    static_assert(CHAR_BIT == 8, "Size of a byte must be 8 bits.");
    return sizeof(T) * CHAR_BIT;
}

/**
 * @brief Enum class for representing data types used in Tensors and DataFrame columns.
 */
enum class MORPHEUS_EXPORT TypeId : int32_t
{
    EMPTY,    ///< Always null with no underlying data
    INT8,     ///< 1 byte signed integer
    INT16,    ///< 2 byte signed integer
    INT32,    ///< 4 byte signed integer
    INT64,    ///< 8 byte signed integer
    UINT8,    ///< 1 byte unsigned integer
    UINT16,   ///< 2 byte unsigned integer
    UINT32,   ///< 4 byte unsigned integer
    UINT64,   ///< 8 byte unsigned integer
    FLOAT32,  ///< 4 byte floating point
    FLOAT64,  ///< 8 byte floating point
    BOOL8,    ///< Boolean using one byte per value, 0 == false, else true
    STRING,   ///< String elements, not supported by cupy

    //   TIMESTAMP_DAYS,          ///< point in time in days since Unix Epoch in int32
    //   TIMESTAMP_SECONDS,       ///< point in time in seconds since Unix Epoch in int64
    //   TIMESTAMP_MILLISECONDS,  ///< point in time in milliseconds since Unix Epoch in int64
    //   TIMESTAMP_MICROSECONDS,  ///< point in time in microseconds since Unix Epoch in int64
    //   TIMESTAMP_NANOSECONDS,   ///< point in time in nanoseconds since Unix Epoch in int64
    //   DURATION_DAYS,           ///< time interval of days in int32
    //   DURATION_SECONDS,        ///< time interval of seconds in int64
    //   DURATION_MILLISECONDS,   ///< time interval of milliseconds in int64
    //   DURATION_MICROSECONDS,   ///< time interval of microseconds in int64
    //   DURATION_NANOSECONDS,    ///< time interval of nanoseconds in int64
    //   DICTIONARY32,            ///< Dictionary type using int32 indices
    //   LIST,                    ///< List elements
    //   DECIMAL32,               ///< Fixed-point type with int32_t
    //   DECIMAL64,               ///< Fixed-point type with int64_t
    //   STRUCT,                  ///< Struct elements

    // `NUM_TYPE_IDS` must be last!
    NUM_TYPE_IDS  ///< Total number of type ids
};

/**
 * @class DType
 * @brief This class represents a data type specified by a TypeId.
 */
struct MORPHEUS_EXPORT DType
{
    /**
     * @brief Construct a DType for a given type specified by a TypeId.
     *
     * @param tid The TypeId to initialize the DType object with.
     */
    DType(TypeId tid);

    /**
     * @brief Copy constructor.
     *
     * @param dtype The DType object to copy from.
     */
    DType(const DType& dtype) = default;

    /**
     * @brief Equality operator.
     *
     * @param other The DType object to compare with.
     * @return True if the two DType objects represent the same TypeId, false otherwise.
     */
    bool operator==(const DType& other) const;

    /**
     * @brief Get the TypeId of the DType object.
     *
     * @return The TypeId of the DType object.
     */
    TypeId type_id() const;

    /**
     * @brief Get the number of bytes per item.
     *
     * @return The number of bytes per item.
     */
    size_t item_size() const;

    /**
     * @brief Get the name of the DType object.
     *
     * @return The name of the DType object.
     */
    std::string name() const;

    /**
     * @brief Get the numpy string representation of the DType object.
     *
     * @return The numpy string representation of the DType object.
     */
    std::string type_str() const;

    /**
     * @brief Get the cudf type id of the DType object.
     *
     * @return The cudf type id of the DType object.
     */
    cudf::type_id cudf_type_id() const;

    /**
     * @brief Get the triton string representation of the DType object.
     *
     * @return The triton string representation of the DType object.
     */
    std::string triton_str() const;

    /**
     * @brief Create a DType object from a cudf type id.
     *
     * @param id The cudf type id.
     * @return A DType object.
     */
    static DType from_cudf(cudf::type_id tid);

    /**
     * @brief Create a DType object from a numpy type string.
     *
     * @param type_str The numpy type string.
     * @return A DType object.
     */
    static DType from_numpy(const std::string& numpy_str);

    /**
     * @brief Create a DType object from a triton type string.
     *
     * @param type_str The triton type string.
     * @return A DType object.
     */
    static DType from_triton(const std::string& type_str);

    /**
     * @brief Check if the DType object is fully supported.
     *
     * @return True if the DType object is fully supported, false otherwise.
     */
    bool is_fully_supported() const;

    /**
     * @brief Construct a DType object from a C++ type.
     *
     * @return A DType object.
     */
    template <typename T>
    static DType create()
    {
        if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 8)
        {
            return {TypeId::INT8};
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 16)
        {
            return {TypeId::INT16};
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 32)
        {
            return {TypeId::INT32};
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 64)
        {
            return {TypeId::INT64};
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 8)
        {
            return {TypeId::UINT8};
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 16)
        {
            return {TypeId::UINT16};
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 32)
        {
            return {TypeId::UINT32};
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 64)
        {
            return {TypeId::UINT64};
        }
        else if constexpr (std::is_floating_point_v<T> && size_in_bits<T>() == 32)
        {
            return {TypeId::FLOAT32};
        }
        else if constexpr (std::is_floating_point_v<T> && size_in_bits<T>() == 64)
        {
            return {TypeId::FLOAT64};
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return {TypeId::BOOL8};
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            return {TypeId::STRING};
        }
        else
        {
            static_assert(!sizeof(T), "Type not implemented");
        }

        // To hide compiler warnings
        return {TypeId::EMPTY};
    }

  private:
    char byte_order_char() const;
    char type_char() const;

    TypeId m_type_id;
};
/** @} */  // end of group
}  // namespace morpheus
