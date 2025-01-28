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
#include "morpheus/messages/meta.hpp"
#include "morpheus/types.hpp"

#include <boost/fiber/context.hpp>
#include <cudf/io/types.hpp>
#include <librdkafka/rdkafkacpp.h>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pybind11/pytypes.h>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity

#include <cstddef>  // for size_t
#include <cstdint>  // for uuint32_t
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace morpheus {

/****** Component public implementations *******************/
/****** KafkaSourceStage************************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

class MORPHEUS_EXPORT KafkaOAuthCallback : public RdKafka::OAuthBearerTokenRefreshCb
{
  public:
    KafkaOAuthCallback(const std::function<std::map<std::string, std::string>()>& oauth_callback);

    void oauthbearer_token_refresh_cb(RdKafka::Handle* handle, const std::string& oauthbearer_config) override;

  private:
    const std::function<std::map<std::string, std::string>()>& m_oauth_callback;
};
/**
 * This class loads messages from the Kafka cluster by serving as a Kafka consumer.
 */
class MORPHEUS_EXPORT KafkaSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    /**
     * @brief Construct a new Kafka Source Stage object
     *
     * @param max_batch_size : The maximum batch size for the messages batch.
     * @param topic : Input kafka topic.
     * @param batch_timeout_ms : Frequency of the poll in ms.
     * @param config : Kafka consumer configuration.
     * @param disable_commit : Enabling this option will skip committing messages as they are pulled off the server.
     * This is only useful for debugging, allowing the user to process the same messages multiple times
     * @param disable_pre_filtering : Enabling this option will skip pre-filtering of json messages.
     * This is only useful when inputs are known to be valid json.
     * @param stop_after : Stops ingesting after emitting `stop_after` records (rows in the table).
     * Useful for testing. Disabled if `0`
     * @param async_commits : Asynchronously acknowledge consuming Kafka messages
     */
    KafkaSourceStage(TensorIndex max_batch_size,
                     std::string topic,
                     uint32_t batch_timeout_ms,
                     std::map<std::string, std::string> config,
                     bool disable_commit                                = false,
                     bool disable_pre_filtering                         = false,
                     std::size_t stop_after                             = 0,
                     bool async_commits                                 = true,
                     std::unique_ptr<KafkaOAuthCallback> oauth_callback = nullptr);

    /**
     * @brief Construct a new Kafka Source Stage object
     *
     * @param max_batch_size : The maximum batch size for the messages batch.
     * @param topics : Input kafka topics.
     * @param batch_timeout_ms : Frequency of the poll in ms.
     * @param config : Kafka consumer configuration.
     * @param disable_commit : Enabling this option will skip committing messages as they are pulled off the server.
     * This is only useful for debugging, allowing the user to process the same messages multiple times
     * @param disable_pre_filtering : Enabling this option will skip pre-filtering of json messages.
     * This is only useful when inputs are known to be valid json.
     * @param stop_after : Stops ingesting after emitting `stop_after` records (rows in the table).
     * Useful for testing. Disabled if `0`
     * @param async_commits : Asynchronously acknowledge consuming Kafka messages
     */
    KafkaSourceStage(TensorIndex max_batch_size,
                     std::vector<std::string> topics,
                     uint32_t batch_timeout_ms,
                     std::map<std::string, std::string> config,
                     bool disable_commit                                = false,
                     bool disable_pre_filtering                         = false,
                     std::size_t stop_after                             = 0,
                     bool async_commits                                 = true,
                     std::unique_ptr<KafkaOAuthCallback> oauth_callback = nullptr);

    ~KafkaSourceStage() override = default;

    /**
     * @return maximum batch size for KafkaSource.
     */
    TensorIndex max_batch_size();

    /**
     * @return batch timeout in ms.
     */
    uint32_t batch_timeout_ms();

  private:
    /**
     * TODO(Documentation)
     */
    subscriber_fn_t build();

    /**
     * @brief Create kafka consumer configuration and returns unique pointer to the result.
     *
     * @param config_in : Configuration map contains Kafka consumer properties.
     * @return std::unique_ptr<RdKafka::Conf>
     */
    std::unique_ptr<RdKafka::Conf> build_kafka_conf(const std::map<std::string, std::string>& config_in);

    /**
     * @brief Creates Kafka consumer instance.
     *
     * @param rebalancer : Group rebalance callback for use with RdKafka::KafkaConsumer.
     * @return std::unique_ptr<RdKafka::KafkaConsumer>
     */
    std::unique_ptr<RdKafka::KafkaConsumer> create_consumer(RdKafka::RebalanceCb& rebalancer);

    /**
     * @brief Load messages from a buffer/file to a cuDF table.
     *
     * @param buffer : Reference of a messages buffer
     * @return cudf::io::table_with_metadata
     */
    cudf::io::table_with_metadata load_table(const std::string& buffer);

    /**
     * @brief This function combines JSON messages from Kafka, parses them, then loads them onto a MessageMeta.
     * and returns the shared pointer as a result.
     *
     * @param message_batch : Reference of a message batch that needs to be processed.
     * @return std::shared_ptr<morpheus::MessageMeta>
     */
    std::shared_ptr<morpheus::MessageMeta> process_batch(
        std::vector<std::unique_ptr<RdKafka::Message>>&& message_batch);

    TensorIndex m_max_batch_size{128};
    uint32_t m_batch_timeout_ms{100};

    std::vector<std::string> m_topics;
    std::map<std::string, std::string> m_config;

    bool m_disable_commit{false};
    bool m_disable_pre_filtering{false};
    bool m_requires_commit{false};  // Whether or not manual committing is required
    bool m_async_commits{true};
    std::size_t m_stop_after{0};

    void* m_rebalancer;

    std::unique_ptr<KafkaOAuthCallback> m_oauth_callback;
};

/****** KafkaSourceStageInferenceProxy**********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT KafkaSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a KafkaSourceStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param max_batch_size : The maximum batch size for the messages batch.
     * @param topic : Input kafka topic.
     * @param batch_timeout_ms : Frequency of the poll in ms.
     * @param config : Kafka consumer configuration.
     * @param disable_commit : Enabling this option will skip committing messages as they are pulled off the server.
     * This is only useful for debugging, allowing the user to process the same messages multiple times
     * @param disable_pre_filtering : Enabling this option will skip pre-filtering of json messages.
     * This is only useful when inputs are known to be valid json.
     * @param stop_after : Stops ingesting after emitting `stop_after` records (rows in the table).
     * Useful for testing. Disabled if `0`
     * @param async_commits : Asynchronously acknowledge consuming Kafka messages
     * @param oauth_callback : Callback used when an OAuth token needs to be generated.
     */
    static std::shared_ptr<mrc::segment::Object<KafkaSourceStage>> init_with_single_topic(
        mrc::segment::Builder& builder,
        const std::string& name,
        TensorIndex max_batch_size,
        std::string topic,
        uint32_t batch_timeout_ms,
        std::map<std::string, std::string> config,
        bool disable_commit,
        bool disable_pre_filtering,
        std::size_t stop_after                           = 0,
        bool async_commits                               = true,
        std::optional<pybind11::function> oauth_callback = std::nullopt);

    /**
     * @brief Create and initialize a KafkaSourceStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param max_batch_size : The maximum batch size for the messages batch.
     * @param topics : Input kafka topics.
     * @param batch_timeout_ms : Frequency of the poll in ms.
     * @param config : Kafka consumer configuration.
     * @param disable_commit : Enabling this option will skip committing messages as they are pulled off the server.
     * This is only useful for debugging, allowing the user to process the same messages multiple times
     * @param disable_pre_filtering : Enabling this option will skip pre-filtering of json messages.
     * This is only useful when inputs are known to be valid json.
     * @param stop_after : Stops ingesting after emitting `stop_after` records (rows in the table).
     * Useful for testing. Disabled if `0`
     * @param async_commits : Asynchronously acknowledge consuming Kafka messages
     * @param oauth_callback : Callback used when an OAuth token needs to be generated.
     */
    static std::shared_ptr<mrc::segment::Object<KafkaSourceStage>> init_with_multiple_topics(
        mrc::segment::Builder& builder,
        const std::string& name,
        TensorIndex max_batch_size,
        std::vector<std::string> topics,
        uint32_t batch_timeout_ms,
        std::map<std::string, std::string> config,
        bool disable_commit,
        bool disable_pre_filtering,
        std::size_t stop_after                           = 0,
        bool async_commits                               = true,
        std::optional<pybind11::function> oauth_callback = std::nullopt);

  private:
    /**
     * @brief Create a KafkaOAuthCallback or return nullptr. If oauth_callback is std::nullopt,
     * returns nullptr, otherwise wraps the callback in a KafkaOAuthCallback such that the values
     * returned from the python callback are converted for use in c++.
     * @param oauth_callback : The callback to wrap, if any.
     */
    static std::unique_ptr<KafkaOAuthCallback> make_kafka_oauth_callback(
        std::optional<pybind11::function>&& oauth_callback);
};
/** @} */  // end of group
}  // namespace morpheus
