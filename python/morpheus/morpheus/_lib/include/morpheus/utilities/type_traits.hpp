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

#include <type_traits>

namespace morpheus::utilities {

// NOLINTBEGIN(readability-identifier-naming)

/**@{*/
/**
 * @brief Tests whether or not a type is a specialization of a template. i.e. `is_specialization<std::atomic<int>,
 * std::atomic> == true`
 *
 * @tparam T The type to test
 * @tparam TemplateT The base template to test against
 */
template <class T, template <class...> class TemplateT>
struct is_specialization : std::false_type
{};

template <template <class...> class TemplateT, class... ArgsT>
struct is_specialization<TemplateT<ArgsT...>, TemplateT> : std::true_type
{};

template <class T, template <class...> class TemplateT>
using is_specialization_v = typename is_specialization<T, TemplateT>::value;
/**@}*/

/**@{*/
/**
 * @brief Extracts the value type from a templated type. i.e. `extract_value_type<std::vector<int>>::type == int`
 *
 * @tparam T The type to extract the value from
 */
template <typename T>
struct extract_value_type
{
    using type = T;
};

template <template <typename, typename...> class ClassT, typename T, typename... ArgsT>
struct extract_value_type<ClassT<T, ArgsT...>>
{
    using type = T;
};

template <template <typename, typename...> class ClassT, typename T, typename... ArgsT>
using extract_value_type_t = typename extract_value_type<ClassT<T, ArgsT...>>::type;
/**@}*/

// NOLINTEND(readability-identifier-naming)

}  // namespace morpheus::utilities
