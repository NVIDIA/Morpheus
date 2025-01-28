/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/controllers/monitor_controller.hpp"  // for MonitorController
#include "morpheus/export.h"                            // for MORPHEUS_EXPORT

#include <indicators/progress_bar.hpp>
#include <mrc/segment/builder.hpp>  // for Builder
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for trace_activity, decay_t, from

#include <optional>
#include <string>

namespace morpheus {
/*************** Component public implementations ***************/
/******************** MonitorStage ********************/

/**
 * @addtogroup controllers
 * @{
 * @file
 */

/**
 * @brief Displays descriptive progress bars including throughput metrics for the messages passing through the pipeline.
 */
template <typename MessageT>
class MORPHEUS_EXPORT MonitorStage : public mrc::pymrc::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Monitor Stage object
     *
     * @param description : A text label displayed on the left side of the progress bars
     * @param unit : the unit of message count
     * @param determine_count_fn : A function that computes the count for each incoming message
     */
    MonitorStage(const std::string& description,
                 const std::string& unit                                              = "messages",
                 indicators::Color                                                    = indicators::Color::cyan,
                 indicators::FontStyle font_style                                     = indicators::FontStyle::bold,
                 std::optional<std::function<size_t(sink_type_t)>> determine_count_fn = std::nullopt);

  private:
    subscribe_fn_t build_operator();

    MonitorController<sink_type_t> m_monitor_controller;
};

template <typename MessageT>
MonitorStage<MessageT>::MonitorStage(const std::string& description,
                                     const std::string& unit,
                                     indicators::Color text_color,
                                     indicators::FontStyle font_style,
                                     std::optional<std::function<size_t(sink_type_t)>> determine_count_fn) :
  base_t(base_t::op_factory_from_sub_fn(build_operator())),
  m_monitor_controller(MonitorController<sink_type_t>(description, unit, text_color, font_style, determine_count_fn))
{}

template <typename MessageT>
MonitorStage<MessageT>::subscribe_fn_t MonitorStage<MessageT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t msg) {
                m_monitor_controller.progress_sink(msg);
                output.on_next(std::move(msg));
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                m_monitor_controller.sink_on_completed();
                output.on_completed();
            }));
    };
}

/****** MonitorStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
template <typename MessageT>
struct MORPHEUS_EXPORT MonitorStageInterfaceProxy
{
    /**
     * @brief Create and initialize a MonitorStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param description : A text label displayed on the left side of the progress bars
     * @param unit : the unit of message count
     * @param text_color : the text color of progress bars
     * @param font_style : the font style of progress bars
     * @param determine_count_fn : A function that computes the count for each incoming message
     * @return std::shared_ptr<mrc::segment::Object<MonitorStage<MessageT>>>
     */
    static std::shared_ptr<mrc::segment::Object<MonitorStage<MessageT>>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        const std::string& description,
        const std::string& unit,
        indicators::Color color          = indicators::Color::cyan,
        indicators::FontStyle font_style = indicators::FontStyle::bold,
        std::optional<std::function<size_t(typename MonitorStage<MessageT>::sink_type_t)>> determine_count_fn =
            std::nullopt);
};

template <typename MessageT>
std::shared_ptr<mrc::segment::Object<MonitorStage<MessageT>>> MonitorStageInterfaceProxy<MessageT>::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    const std::string& description,
    const std::string& unit,
    indicators::Color text_color,
    indicators::FontStyle font_style,
    std::optional<std::function<size_t(typename MonitorStage<MessageT>::sink_type_t)>> determine_count_fn)
{
    auto stage = builder.construct_object<MonitorStage<MessageT>>(
        name, description, unit, text_color, font_style, determine_count_fn);

    return stage;
}
/** @} */  // end of group
}  // namespace morpheus
