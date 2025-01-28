/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <filesystem>  // for path
#include <ostream>
#include <stdexcept>
#include <string>

namespace morpheus {

/**
 * @addtogroup objects
 * @{
 * @file
 */

enum class MORPHEUS_EXPORT FileTypes : int32_t
{
    Auto,
    JSON,
    CSV,
    PARQUET
};

/**
 * @brief Converts a `FilesTypes` enum to a string
 *
 * @param f
 * @return std::string
 */
inline std::string MORPHEUS_EXPORT filetypes_to_str(const FileTypes& f)
{
    switch (f)
    {
    case FileTypes::Auto:
        return "Auto";
    case FileTypes::JSON:
        return "JSON";
    case FileTypes::CSV:
        return "CSV";
    case FileTypes::PARQUET:
        return "PARQUET";
    default:
        throw std::logic_error("Unsupported FileTypes enum. Was a new value added recently?");
    }
}

/**
 * @brief Stream operator for `FileTypes`
 *
 * @param os
 * @param f
 * @return std::ostream&
 */
static inline std::ostream& operator<<(std::ostream& os, const FileTypes& f)
{
    os << filetypes_to_str(f);
    return os;
}

/**
 * @brief Determines the file type from a filename based on extension. For example, my_file.json would return
 * `FileTypes::JSON`.
 *
 * @param filename String to a file. Does not need to exist
 * @return FileTypes
 */
FileTypes MORPHEUS_EXPORT determine_file_type(const std::string& filename);

/**
 * @brief Determines the file type from a filename based on extension. For example, my_file.json would return
 * `FileTypes::JSON`.
 *
 * @param filename path to a file. Does not need to exist
 * @return FileTypes
 */
FileTypes MORPHEUS_EXPORT determine_file_type(const std::filesystem::path& filename);
/** @} */  // end of group
}  // namespace morpheus
