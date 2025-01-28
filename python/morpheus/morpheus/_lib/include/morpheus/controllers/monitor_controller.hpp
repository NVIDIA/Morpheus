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
#include "indicators/setting.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/line.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <indicators/color.hpp>
#include <indicators/dynamic_progress.hpp>
#include <indicators/font_style.hpp>
#include <indicators/indeterminate_progress_bar.hpp>
#include <indicators/progress_bar.hpp>
#include <mrc/segment/builder.hpp>  // for Builder
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for trace_activity, decay_t, from

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace morpheus {
/******************* MonitorController**********************/

/**
 * @addtogroup controllers
 * @{
 * @file
 */

// Workaround to display progress bars and other logs in the same terminal:
// https://github.com/p-ranav/indicators/issues/107
// Adding "\n" (new line) "\033[A" (move cursor up) and "\033[1L" (insert line) before each log output line.
// This keeps the progress bars always display as the last line of console output
struct LineInsertingFilter : boost::iostreams::line_filter
{
    std::string do_filter(const std::string& line) override
    {
        return "\n\033[A\033[1L" + line;
    }
};

// A singleton manager class that manages the lifetime of progress bars related to any MonitorController<T> instances
// Meyer's singleton is guaranteed thread-safe
class ProgressBarContextManager
{
  public:
    static ProgressBarContextManager& get_instance()
    {
        static ProgressBarContextManager instance;
        return instance;
    }
    ProgressBarContextManager(ProgressBarContextManager const&)            = delete;
    ProgressBarContextManager& operator=(ProgressBarContextManager const&) = delete;

    size_t add_progress_bar(const std::string& description,
                            indicators::Color text_color     = indicators::Color::cyan,
                            indicators::FontStyle font_style = indicators::FontStyle::bold)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_progress_bars.push_back(std::move(initialize_progress_bar(description, text_color, font_style)));

        // DynamicProgress should take ownership over progressbars: https://github.com/p-ranav/indicators/issues/134
        // The fix to this issue is not yet released, so we need to:
        //     - Maintain the lifetime of the progress bar in m_progress_bars
        //     - Push the underlying progress bar object to the DynamicProgress container, since it accepts
        //        Indicator &bar rather than std::unique_ptr<Indicator> bar before the fix
        return m_dynamic_progress_bars.push_back(*m_progress_bars.back());
    }

    std::vector<std::unique_ptr<indicators::IndeterminateProgressBar>>& progress_bars()
    {
        return m_progress_bars;
    }

    void display_all()
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // To avoid display_all() being executed after calling mark_pbar_as_completed() in some race conditions
        if (m_is_completed)
        {
            return;
        }

        // A bit of hack here to make the font settings work. Indicators enables the font options only if the bars are
        // output to standard streams (see is_colorized() in <indicators/termcolor.hpp>), but since we are still using
        // the ostream (m_stdout_os) that is connected to the console terminal, the font options should be enabled.
        // The internal function here is used to manually enable the font display.
        m_stdout_os.iword(termcolor::_internal::colorize_index()) = 1;

        for (auto& pbar : m_progress_bars)
        {
            pbar->print_progress(true);
            m_stdout_os << termcolor::reset;  // The font option only works for the current bar
            m_stdout_os << std::endl;
        }

        // After each round of display, move cursor up ("\033[A") to the beginning of the first bar
        m_stdout_os << "\033[" << m_progress_bars.size() << "A" << std::flush;
    }

    void mark_pbar_as_completed(size_t bar_id)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_progress_bars[bar_id]->mark_as_completed();

        if (!m_is_completed)
        {
            bool all_pbars_completed = true;
            for (auto& pbar : m_progress_bars)
            {
                if (!pbar->is_completed())
                {
                    all_pbars_completed = false;
                    break;
                }
            }
            if (all_pbars_completed)
            {
                // Move the cursor down to the bottom of the last progress bar
                // Doing this here instead of the destructor to avoid a race condition with the pipeline's
                // "====Pipeline Complete====" log message.
                // Using a string stream to ensure other logs are not interleaved.
                std::ostringstream new_lines;
                for (std::size_t i = 0; i < m_progress_bars.size(); ++i)
                {
                    new_lines << "\n";
                }

                m_stdout_os << new_lines.str() << std::flush;
                m_is_completed = true;
            }
        }
    }

  private:
    ProgressBarContextManager() : m_stdout_streambuf(std::cout.rdbuf()), m_stdout_os(m_stdout_streambuf)
    {
        init_log_streambuf();
    }

    ~ProgressBarContextManager()
    {
        // Reset std::cout to use the normal streambuf when exit
        std::cout.rdbuf(m_stdout_streambuf);
    }

    void init_log_streambuf()
    {
        // Configure all std::cout output to use LineInsertingFilter
        m_log_streambuf.push(LineInsertingFilter());
        m_log_streambuf.push(*m_stdout_streambuf);
        std::cout.rdbuf(&m_log_streambuf);
    }

    std::unique_ptr<indicators::IndeterminateProgressBar> initialize_progress_bar(const std::string& description,
                                                                                  indicators::Color text_color,
                                                                                  indicators::FontStyle font_style)
    {
        auto progress_bar = std::make_unique<indicators::IndeterminateProgressBar>(
            indicators::option::BarWidth{10},
            indicators::option::Start{"["},
            indicators::option::Fill{" "},
            indicators::option::Lead{"#"},
            indicators::option::End("]"),
            indicators::option::PrefixText{description},
            indicators::option::ForegroundColor{text_color},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{font_style}},
            indicators::option::Stream{m_stdout_os});

        return std::move(progress_bar);
    }

    indicators::DynamicProgress<indicators::IndeterminateProgressBar> m_dynamic_progress_bars;
    std::vector<std::unique_ptr<indicators::IndeterminateProgressBar>> m_progress_bars;
    std::mutex m_mutex;

    // To ensure progress bars are displayed alongside other log outputs, we use two distinct stream buffers:
    //  - Progress bars are redirected to m_stdout_os, which points to the original standard output stream.
    //  - All std::cout output is redirected to m_log_streambuf, which incorporates a LineInsertingFilter to continually
    //  shift the progress bar display downward.
    std::streambuf* m_stdout_streambuf;  // Stores the original std::cout.rdbuf()
    std::ostream m_stdout_os;
    boost::iostreams::filtering_ostreambuf m_log_streambuf;
    bool m_is_completed{false};
};

/**
 * @brief A controller class that manages the display of progress bars that used by MonitorStage.
 */
template <typename MessageT>
class MonitorController
{
  public:
    /**
     * @brief Construct a new Monitor Controller object
     *
     * @param description : A text label displayed on the left side of the progress bars
     * @param unit : the unit of message count
     * @param determine_count_fn : A function that computes the count for each incoming message
     */
    MonitorController(const std::string& description,
                      std::string unit                                                  = "messages",
                      indicators::Color text_color                                      = indicators::Color::cyan,
                      indicators::FontStyle font_style                                  = indicators::FontStyle::bold,
                      std::optional<std::function<size_t(MessageT)>> determine_count_fn = std::nullopt);

    auto auto_count_fn() -> std::optional<std::function<size_t(MessageT)>>;

    MessageT progress_sink(MessageT msg);
    void sink_on_completed();

  private:
    static std::string format_duration(std::chrono::seconds duration);
    static std::string format_throughput(std::chrono::seconds duration, size_t count, const std::string& unit);

    size_t m_bar_id;
    const std::string m_unit;
    std::optional<std::function<size_t(MessageT)>> m_determine_count_fn;
    size_t m_count{0};
    time_point_t m_start_time;
    bool m_is_started{false};  // Set to true after the first call to progress_sink()
};

template <typename MessageT>
MonitorController<MessageT>::MonitorController(const std::string& description,
                                               std::string unit,
                                               indicators::Color text_color,
                                               indicators::FontStyle font_style,
                                               std::optional<std::function<size_t(MessageT)>> determine_count_fn) :
  m_unit(std::move(unit)),
  m_determine_count_fn(determine_count_fn)
{
    if (!m_determine_count_fn)
    {
        m_determine_count_fn = auto_count_fn();
        if (!m_determine_count_fn)
        {
            throw std::runtime_error("No count function provided and no default count function available");
        }
    }

    m_bar_id = ProgressBarContextManager::get_instance().add_progress_bar(description, text_color, font_style);
}

template <typename MessageT>
MessageT MonitorController<MessageT>::progress_sink(MessageT msg)
{
    if (!m_is_started)
    {
        m_start_time = std::chrono::system_clock::now();
        m_is_started = true;
    }
    m_count += (*m_determine_count_fn)(msg);
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - m_start_time);

    auto& manager = ProgressBarContextManager::get_instance();
    auto& pbar    = manager.progress_bars()[m_bar_id];

    // Update the progress bar
    pbar->set_option(indicators::option::PostfixText{format_throughput(duration, m_count, m_unit)});
    pbar->tick();

    manager.display_all();

    return msg;
}

template <typename MessageT>
void MonitorController<MessageT>::sink_on_completed()
{
    auto& manager = ProgressBarContextManager::get_instance();
    manager.mark_pbar_as_completed(m_bar_id);
}

template <typename MessageT>
std::string MonitorController<MessageT>::format_duration(std::chrono::seconds duration)
{
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    auto seconds = duration - minutes;

    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << minutes.count() << "m:" << std::setw(2) << std::setfill('0')
        << seconds.count() << "s";
    return oss.str();
}

template <typename MessageT>
std::string MonitorController<MessageT>::format_throughput(std::chrono::seconds duration,
                                                           size_t count,
                                                           const std::string& unit)
{
    double throughput = static_cast<double>(count) / duration.count();
    std::ostringstream oss;
    oss << count << " " << unit << " in " << format_duration(duration) << ", "
        << "Throughput: " << std::fixed << std::setprecision(2) << throughput << " " << unit << "/s";
    return oss.str();
}

template <typename MessageT>
auto MonitorController<MessageT>::auto_count_fn() -> std::optional<std::function<size_t(MessageT)>>
{
    if constexpr (std::is_same_v<MessageT, std::shared_ptr<MessageMeta>>)
    {
        return [](std::shared_ptr<MessageMeta> msg) {
            return msg->count();
        };
    }

    if constexpr (std::is_same_v<MessageT, std::shared_ptr<ControlMessage>>)
    {
        return [](std::shared_ptr<ControlMessage> msg) {
            if (!msg->payload())
            {
                return 0;
            }
            return msg->payload()->count();
        };
    }

    return std::nullopt;
}

/** @} */  // end of group
}  // namespace morpheus
