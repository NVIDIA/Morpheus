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

#include <sstream>
#include <string>

namespace morpheus {

/**
 * @addtogroup utilities
 * @{
 * @file
 */

/**
 * @brief Concats multiple strings together using ostringstream. Use with MORPHEUS_CONCAT_STR("Start [" << my_int <<
 * "]")
 *
 */
#define MORPHEUS_CONCAT_STR(strs) ((std::ostringstream&)(std::ostringstream() << strs)).str()

/****** Component public implementations *******************/
/****** StringUtil ****************************************/

/**
 * @brief A struct that encapsulates string utilities.
 */
struct StringUtil
{
    /**
     * @brief Concatenate a sequence of values with a separator string.
     *
     * This method takes a pair of iterators `begin` and `end` that define a sequence of values, and
     * a separator string. It returns a string that concatenates all the values in the sequence with
     * the separator string between each pair of values.
     *
     * @tparam IterT A template parameter representing the iterator type.
     * @param begin An iterator pointing to the beginning of the sequence.
     * @param end An iterator pointing to the end of the sequence.
     * @param separator A string to insert between each pair of values.
     * @return A string containing the concatenated values.
     */
    template <typename IterT>
    static std::string join(IterT begin, IterT end, std::string const& separator)
    {
        std::ostringstream result;
        if (begin != end)
            result << *begin++;
        while (begin != end)
            result << separator << *begin++;
        return result.str();
    }

    /**
     * @brief Convert a sequence of values to a string representation.
     *
     * This method takes a pair of iterators `begin` and `end` that define a sequence of values. It
     * returns a string that represents the sequence as a comma-separated list enclosed in square
     * brackets.
     *
     * @tparam IterT A template parameter representing the iterator type.
     * @param begin An iterator pointing to the beginning of the sequence.
     * @param end An iterator pointing to the end of the sequence.
     * @return A string containing the string representation of the sequence.
     */
    template <typename IterT>
    static std::string array_to_str(IterT begin, IterT end)
    {
        return MORPHEUS_CONCAT_STR("[" << join(begin, end, ", ") << "]");
    }

    /**
     * @brief Generates a string representation of a std::map in the form "{key1: 'value1', key2: 'value2'}"
     *
     * @tparam IterT Deduced iterator type
     * @param begin Start iterator. Use `myMap.begin()`
     * @param end End iterator. Use `myMap.end()`
     * @return std::string
     */
    template <typename IterT>
    static std::string map_to_str(IterT begin, IterT end)
    {
        std::ostringstream ss;

        ss << "{";

        if (begin != end)
        {
            ss << begin->first << ": '" << begin->second << "'";
            ++begin;
        }
        while (begin != end)
        {
            ss << ", " << begin->first << ": '" << begin->second << "'";
            ++begin;
        }

        ss << "}";

        return ss.str();
    }

    /**
     * @brief Check if a string contains a substring.
     *
     * This method takes two string arguments, `str` and `search_str`. It returns true if `str`
     * contains `search_str`, and false otherwise.
     *
     * @param str The string to search in.
     * @param search_str The string to search for.
     * @return True if `str` contains `search_str`, false otherwise.
     */
    static bool str_contains(const std::string& str, const std::string& search_str);
};
/** @} */  // end of group
}  // namespace morpheus
