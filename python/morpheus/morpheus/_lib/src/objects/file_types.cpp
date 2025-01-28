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

#include "morpheus/objects/file_types.hpp"

#include "morpheus/utilities/string_util.hpp"

#include <filesystem>
#include <sstream>  // ostringstream needed by MORPHEUS_CONCAT_STR
#include <stdexcept>

namespace morpheus {

FileTypes determine_file_type(const std::string& filename)
{
    auto filename_path = std::filesystem::path(filename);

    if (filename_path.extension() == ".json" || filename_path.extension() == ".jsonlines")
    {
        return FileTypes::JSON;
    }
    else if (filename_path.extension() == ".csv")
    {
        return FileTypes::CSV;
    }
    else if (filename_path.extension() == ".parquet")
    {
        return FileTypes::PARQUET;
    }
    else
    {
        throw std::runtime_error(MORPHEUS_CONCAT_STR("Unsupported extension '"
                                                     << filename_path.extension()
                                                     << "' with 'auto' type. 'auto' only works with: csv, json"));
    }
}

FileTypes determine_file_type(const std::filesystem::path& filename)
{
    return determine_file_type(filename.string());
}

}  // namespace morpheus
