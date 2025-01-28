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

#include "morpheus/stages/kafka_source.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/utilities/stage_util.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <boost/fiber/operations.hpp>  // for sleep_for, yield
#include <boost/fiber/recursive_mutex.hpp>
#include <cudf/io/json.hpp>
#include <glog/logging.h>
#include <librdkafka/rdkafkacpp.h>
#include <mrc/runnable/context.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/types.hpp>  // for SharedFuture
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/node.hpp>
#include <pymrc/utilities/function_wrappers.hpp>  // for PyFuncWrapper

#include <algorithm>  // for find, min, transform
#include <chrono>
#include <compare>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>  // for back_insert_iterator, back_inserter
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
// IWYU thinks we need atomic for vector.emplace_back of a unique_ptr
// and __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <atomic>
// IWYU pragma: no_include <ext/alloc_traits.h>

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
    /**
     * @brief Checks the error code returned by an RDKafka expression (`command`) against an `expected` code
     * (usually `RdKafka::ERR_NO_ERROR`), and logs an error otherwise.
     *
     */
    #define CHECK_KAFKA(command, expected, msg)                                                                    \
        {                                                                                                          \
            RdKafka::ErrorCode __code = command;                                                                   \
            if (__code != expected)                                                                                \
            {                                                                                                      \
                LOG(ERROR) << msg << ". Received unexpected ErrorCode. Expected: " << #expected << "(" << expected \
                           << "), Received: " << __code << ", Msg: " << RdKafka::err2str(__code);                  \
            }                                                                                                      \
        };
#endif  // DOXYGEN_SHOULD_SKIP_THIS

namespace morpheus {

KafkaOAuthCallback::KafkaOAuthCallback(const std::function<std::map<std::string, std::string>()>& oauth_callback) :
  m_oauth_callback(oauth_callback)
{}

void KafkaOAuthCallback::oauthbearer_token_refresh_cb(RdKafka::Handle* handle, const std::string& oauthbearer_config)
{
    try
    {
        auto response = m_oauth_callback();
        // Build parameters to pass to librdkafka
        std::string token         = response["token"];
        int64_t token_lifetime_ms = std::stoll(response["token_expiration_in_epoch"]);
        std::list<std::string> extensions;  // currently not supported
        std::string errstr;
        auto result = handle->oauthbearer_set_token(token, token_lifetime_ms, "kafka", extensions, errstr);
        CHECK(result == RdKafka::ErrorCode::ERR_NO_ERROR) << "Error occurred while setting the oauthbearer token";
    } catch (std::exception ex)
    {
        LOG(FATAL) << "Exception occured oauth refresh: " << ex.what();
    }
}

// Component-private classes.
// ************ KafkaSourceStage__UnsubscribedException**************//
class KafkaSourceStageUnsubscribedException : public std::exception
{};

class KafkaSourceStageStopAfter : public std::exception
{};

// ************ KafkaSourceStage__Rebalancer *************************//
class KafkaSourceStage__Rebalancer : public RdKafka::RebalanceCb  // NOLINT
{
  public:
    KafkaSourceStage__Rebalancer(std::function<uint32_t()> batch_timeout_fn,
                                 std::function<TensorIndex()> max_batch_size_fn,
                                 std::function<std::string(std::string)> display_str_fn,
                                 std::function<bool(std::vector<std::unique_ptr<RdKafka::Message>>&)> process_fn);

    void rebalance_cb(RdKafka::KafkaConsumer* consumer,
                      RdKafka::ErrorCode err,
                      std::vector<RdKafka::TopicPartition*>& partitions) override;

    void rebalance_loop(RdKafka::KafkaConsumer* consumer);

    bool is_rebalanced();

    std::vector<std::unique_ptr<RdKafka::Message>> partition_progress_step(RdKafka::KafkaConsumer* consumer)
    {
        // auto batch_timeout = std::chrono::milliseconds(m_parent.batch_timeout_ms());
        auto batch_timeout = std::chrono::milliseconds(m_batch_timeout_fn());

        std::vector<std::unique_ptr<RdKafka::Message>> messages;

        auto now       = std::chrono::high_resolution_clock::now();
        auto batch_end = now + batch_timeout;

        do
        {
            auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - now).count();

            DCHECK(remaining_ms >= 0) << "Cant have negative reminaing time";

            std::unique_ptr<RdKafka::Message> msg{consumer->consume(std::min(10L, remaining_ms))};

            switch (msg->err())
            {
            case RdKafka::ERR__TIMED_OUT:
                // Yield on a timeout
                boost::this_fiber::yield();
                break;
            case RdKafka::ERR_NO_ERROR:

                messages.emplace_back(std::move(msg));
                break;
            case RdKafka::ERR__PARTITION_EOF:
                VLOG_EVERY_N(10, 10) << "Hit EOF for partition";
                // Hit the end, sleep for 100 ms
                boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
                break;
            default:
                /* Errors */
                LOG(ERROR) << "Consume failed: " << msg->errstr();
            }

            // Update now
            now = std::chrono::high_resolution_clock::now();
        } while (messages.size() < m_max_batch_size_fn() && now < batch_end);

        return std::move(messages);
    }

    bool process_messages(std::vector<std::unique_ptr<RdKafka::Message>>& messages)
    {
        return m_process_fn(messages);
    }

  private:
    bool m_is_rebalanced{false};

    std::function<uint32_t()> m_batch_timeout_fn;
    std::function<TensorIndex()> m_max_batch_size_fn;
    std::function<std::string(std::string)> m_display_str_fn;
    std::function<bool(std::vector<std::unique_ptr<RdKafka::Message>>&)> m_process_fn;

    boost::fibers::recursive_mutex m_mutex;
    mrc::SharedFuture<bool> m_partition_future;
};

KafkaSourceStage__Rebalancer::KafkaSourceStage__Rebalancer(
    std::function<uint32_t()> batch_timeout_fn,
    std::function<TensorIndex()> max_batch_size_fn,
    std::function<std::string(std::string)> display_str_fn,
    std::function<bool(std::vector<std::unique_ptr<RdKafka::Message>>&)> process_fn) :
  m_batch_timeout_fn(std::move(batch_timeout_fn)),
  m_max_batch_size_fn(std::move(max_batch_size_fn)),
  m_display_str_fn(std::move(display_str_fn)),
  m_process_fn(std::move(process_fn))
{}

void KafkaSourceStage__Rebalancer::rebalance_cb(RdKafka::KafkaConsumer* consumer,
                                                RdKafka::ErrorCode err,
                                                std::vector<RdKafka::TopicPartition*>& partitions)
{
    std::unique_lock<boost::fibers::recursive_mutex> lock(m_mutex);

    std::vector<RdKafka::TopicPartition*> current_assignment;
    CHECK_KAFKA(consumer->assignment(current_assignment), RdKafka::ERR_NO_ERROR, "Error retrieving current assignment");

    auto old_partition_ids = foreach_map(current_assignment, [](const auto& x) {
        return x->partition();
    });
    auto new_partition_ids = foreach_map(partitions, [](const auto& x) {
        return x->partition();
    });

    if (err == RdKafka::ERR__ASSIGN_PARTITIONS)
    {
        VLOG(10) << m_display_str_fn(MORPHEUS_CONCAT_STR(
            "Rebalance: Assign Partitions. Current Partitions: << "
            << StringUtil::array_to_str(old_partition_ids.begin(), old_partition_ids.end())
            << ". Assigning: " << StringUtil::array_to_str(new_partition_ids.begin(), new_partition_ids.end())));

        // application may load offets from arbitrary external storage here and update \p partitions
        if (consumer->rebalance_protocol() == "COOPERATIVE")
        {
            CHECK_KAFKA(std::unique_ptr<RdKafka::Error>(consumer->incremental_assign(partitions))->code(),
                        RdKafka::ERR_NO_ERROR,
                        "Error during incremental assign");
        }
        else
        {
            CHECK_KAFKA(consumer->assign(partitions), RdKafka::ERR_NO_ERROR, "Error during assign");
        }
    }
    else if (err == RdKafka::ERR__REVOKE_PARTITIONS)
    {
        VLOG(10) << m_display_str_fn(MORPHEUS_CONCAT_STR(
            "Rebalance: Revoke Partitions. Current Partitions: << "
            << StringUtil::array_to_str(old_partition_ids.begin(), old_partition_ids.end())
            << ". Revoking: " << StringUtil::array_to_str(new_partition_ids.begin(), new_partition_ids.end())));

        // Application may commit offsets manually here if auto.commit.enable=false
        if (consumer->rebalance_protocol() == "COOPERATIVE")
        {
            CHECK_KAFKA(std::unique_ptr<RdKafka::Error>(consumer->incremental_unassign(partitions))->code(),
                        RdKafka::ERR_NO_ERROR,
                        "Error during incremental unassign");
        }
        else
        {
            CHECK_KAFKA(consumer->unassign(), RdKafka::ERR_NO_ERROR, "Error during unassign");
        }
    }
    else
    {
        LOG(ERROR) << "Rebalancing error: " << RdKafka::err2str(err) << std::endl;
        CHECK_KAFKA(consumer->unassign(), RdKafka::ERR_NO_ERROR, "Error during unassign");
    }
}

void KafkaSourceStage__Rebalancer::rebalance_loop(RdKafka::KafkaConsumer* consumer)
{
    do
    {
        // Poll until we are rebalanced
        while (!this->is_rebalanced())
        {
            VLOG(10) << m_display_str_fn("Rebalance: Calling poll to trigger rebalance");
            consumer->poll(500);
        }
    } while (m_partition_future.get());
}

bool KafkaSourceStage__Rebalancer::is_rebalanced()
{
    std::unique_lock<boost::fibers::recursive_mutex> lock(m_mutex);

    return m_is_rebalanced;
}

class KafkaRebalancer : public RdKafka::RebalanceCb
{
  private:
    std::unique_ptr<RdKafka::KafkaConsumer> m_consumer;
};

// Component public implementations
// ************ KafkaStage ************************* //
KafkaSourceStage::KafkaSourceStage(TensorIndex max_batch_size,
                                   std::string topic,
                                   uint32_t batch_timeout_ms,
                                   std::map<std::string, std::string> config,
                                   bool disable_commit,
                                   bool disable_pre_filtering,
                                   std::size_t stop_after,
                                   bool async_commits,
                                   std::unique_ptr<KafkaOAuthCallback> oauth_callback) :
  PythonSource(build()),
  m_max_batch_size(max_batch_size),
  m_topics(std::vector<std::string>{std::move(topic)}),
  m_batch_timeout_ms(batch_timeout_ms),
  m_config(std::move(config)),
  m_disable_commit(disable_commit),
  m_disable_pre_filtering(disable_pre_filtering),
  m_stop_after{stop_after},
  m_async_commits(async_commits),
  m_oauth_callback(std::move(oauth_callback))
{}

KafkaSourceStage::KafkaSourceStage(TensorIndex max_batch_size,
                                   std::vector<std::string> topics,
                                   uint32_t batch_timeout_ms,
                                   std::map<std::string, std::string> config,
                                   bool disable_commit,
                                   bool disable_pre_filtering,
                                   std::size_t stop_after,
                                   bool async_commits,
                                   std::unique_ptr<KafkaOAuthCallback> oauth_callback) :
  PythonSource(build()),
  m_max_batch_size(max_batch_size),
  m_topics(std::move(topics)),
  m_batch_timeout_ms(batch_timeout_ms),
  m_config(std::move(config)),
  m_disable_commit(disable_commit),
  m_disable_pre_filtering(disable_pre_filtering),
  m_stop_after{stop_after},
  m_async_commits(async_commits),
  m_oauth_callback(std::move(oauth_callback))
{}

KafkaSourceStage::subscriber_fn_t KafkaSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> sub) -> void {
        std::size_t records_emitted = 0;
        // Build rebalancer
        KafkaSourceStage__Rebalancer rebalancer(
            [this]() {
                return this->batch_timeout_ms();
            },
            [this]() {
                return this->max_batch_size();
            },
            [this](const std::string str_to_display) {
                auto& ctx = mrc::runnable::Context::get_runtime_context();
                return MORPHEUS_CONCAT_STR(ctx.info() << " " << str_to_display);
            },
            [sub, &records_emitted, this](std::vector<std::unique_ptr<RdKafka::Message>>& message_batch) {
                // If we are unsubscribed, throw an error to break the loops
                if (!sub.is_subscribed())
                {
                    throw KafkaSourceStageUnsubscribedException();
                }
                else if (m_stop_after > 0 && records_emitted >= m_stop_after)
                {
                    throw KafkaSourceStageStopAfter();
                }

                if (message_batch.empty())
                {
                    return false;
                }

                std::shared_ptr<morpheus::MessageMeta> batch;

                try
                {
                    batch = std::move(this->process_batch(std::move(message_batch)));
                } catch (std::exception& ex)
                {
                    LOG(ERROR) << "Exception in process_batch. Msg: " << ex.what();

                    return false;
                }

                auto num_records = batch->count();
                sub.on_next(std::move(batch));
                records_emitted += num_records;
                return m_requires_commit;
            });

        auto& context = mrc::runnable::Context::get_runtime_context();

        // Build consumer
        auto consumer = this->create_consumer(rebalancer);

        // Wait for all to connect
        context.barrier();

        try
        {
            while (sub.is_subscribed())
            {
                std::vector<std::unique_ptr<RdKafka::Message>> message_batch =
                    rebalancer.partition_progress_step(consumer.get());

                // Process the messages. Returns true if we need to commit
                auto should_commit = rebalancer.process_messages(message_batch);

                if (should_commit)
                {
                    if (m_async_commits)
                    {
                        CHECK_KAFKA(consumer->commitAsync(), RdKafka::ERR_NO_ERROR, "Error during commitAsync");
                    }
                    else
                    {
                        CHECK_KAFKA(consumer->commitSync(), RdKafka::ERR_NO_ERROR, "Error during commit");
                    }
                }
            }

        } catch (KafkaSourceStageStopAfter)
        {
            DLOG(INFO) << "Completed after emitting " << records_emitted << " records";
        } catch (std::exception& ex)
        {
            LOG(ERROR) << "Exception in rebalance_loop. Msg: " << ex.what();
        }

        consumer->unsubscribe();
        consumer->close();
        consumer.reset();

        m_rebalancer = nullptr;

        sub.on_completed();
    };
}

TensorIndex KafkaSourceStage::max_batch_size()
{
    return m_max_batch_size;
}

uint32_t KafkaSourceStage::batch_timeout_ms()
{
    return m_batch_timeout_ms;
}

std::unique_ptr<RdKafka::Conf> KafkaSourceStage::build_kafka_conf(const std::map<std::string, std::string>& config_in)
{
    // Copy the config
    std::map<std::string, std::string> config_out(config_in);

    std::map<std::string, std::string> defaults{{"session.timeout.ms", "60000"},
                                                {"enable.auto.commit", "false"},
                                                {"auto.offset.reset", "latest"},
                                                {"enable.partition.eof", "true"}};

    // Set some defaults if they dont exist
    config_out.merge(defaults);

    m_requires_commit = config_out["enable.auto.commit"] == "false";

    if (m_requires_commit && m_disable_commit)
    {
        LOG(WARNING) << "KafkaSourceStage: Commits have been disabled for this Kafka consumer. This should only be "
                        "used in a debug environment";
        m_requires_commit = false;
    }
    else if (!m_requires_commit && m_disable_commit)
    {
        // User has auto-commit on and disable commit at same time
        LOG(WARNING) << "KafkaSourceStage: The config option 'enable.auto.commit' was set to True but commits have "
                        "been disabled for this Kafka consumer. This should only be used in a debug environment";
    }

    // Make the kafka_conf and set all properties
    auto kafka_conf = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

    for (auto const& key_value : config_out)
    {
        std::string error_string;
        if (RdKafka::Conf::ConfResult::CONF_OK != kafka_conf->set(key_value.first, key_value.second, error_string))
        {
            LOG(ERROR) << "Error occurred while setting Kafka configuration. Error: " << error_string;
        }
    }

    return std::move(kafka_conf);
}

std::unique_ptr<RdKafka::KafkaConsumer> KafkaSourceStage::create_consumer(RdKafka::RebalanceCb& rebalancer)
{
    auto kafka_conf = this->build_kafka_conf(m_config);
    std::string errstr;

    if (RdKafka::Conf::ConfResult::CONF_OK != kafka_conf->set("rebalance_cb", &rebalancer, errstr))
    {
        LOG(FATAL) << "Error occurred while setting Kafka rebalance function. Error: " << errstr;
    }

    if (m_oauth_callback != nullptr)
    {
        if (RdKafka::Conf::ConfResult::CONF_OK !=
            kafka_conf->set("oauthbearer_token_refresh_cb", m_oauth_callback.get(), errstr))
        {
            LOG(FATAL) << "Error occurred while setting Kafka OAuth Callback function. Error: " << errstr;
        }
    }

    auto consumer = std::unique_ptr<RdKafka::KafkaConsumer>(RdKafka::KafkaConsumer::create(kafka_conf.get(), errstr));

    if (!consumer)
    {
        LOG(FATAL) << "Error occurred creating Kafka consumer. Error: " << errstr;
    }

    // Subscribe to the topics. Uses the default rebalancer
    CHECK_KAFKA(consumer->subscribe(m_topics), RdKafka::ERR_NO_ERROR, "Error subscribing to topics");

    // Create a vector of topic objects
    std::vector<std::unique_ptr<RdKafka::Topic>> topicObjs;

    // Create a topic object for each topic name and add it to the vector
    for (const auto& m_topic : m_topics)
    {
        auto topicObj =
            std::unique_ptr<RdKafka::Topic>(RdKafka::Topic::create(consumer.get(), m_topic, nullptr, errstr));
        if (!topicObj)
        {
            throw std::runtime_error("Failed to create Kafka topic object");
        }
        topicObjs.push_back(std::move(topicObj));
    }

    RdKafka::Metadata* md;

    for (unsigned short i = 0; i < 5; ++i)
    {
        auto err_code =
            consumer->metadata(topicObjs.empty(), topicObjs.empty() ? nullptr : topicObjs[0].get(), &md, 1000);

        if (err_code == RdKafka::ERR_NO_ERROR && md != nullptr)
        {
            break;
        }

        LOG(WARNING) << "Timed out while trying to list topics from the Kafka broker. Attempt #" << i + 1
                     << "/5. Error message: " << RdKafka::err2str(err_code);
    }

    if (md == nullptr)
    {
        throw std::runtime_error("Failed to list topics in Kafka broker after 5 attempts");
    }

    std::map<std::string, std::vector<int32_t>> topic_parts;

    auto& ctx = mrc::runnable::Context::get_runtime_context();
    VLOG(10) << ctx.info() << MORPHEUS_CONCAT_STR(" Subscribed to " << md->topics()->size() << " topics:");

    for (auto const& topic : *(md->topics()))
    {
        auto& part_ids = topic_parts[topic->topic()];

        auto const& parts = *(topic->partitions());

        std::transform(parts.cbegin(), parts.cend(), std::back_inserter(part_ids), [](auto const& part) {
            return part->id();
        });

        auto toppar_list = foreach_map(parts, [&topic](const auto& part) {
            return std::unique_ptr<RdKafka::TopicPartition>{
                RdKafka::TopicPartition::create(topic->topic(), part->id())};
        });

        std::vector<RdKafka::TopicPartition*> toppar_ptrs =
            foreach_map(toppar_list, [](const std::unique_ptr<RdKafka::TopicPartition>& x) {
                return x.get();
            });

        // Query Kafka to populate the TopicPartitions with the desired offsets
        CHECK_KAFKA(
            consumer->committed(toppar_ptrs, 2000), RdKafka::ERR_NO_ERROR, "Failed retrieve Kafka committed offsets");

        auto committed = foreach_map(toppar_list, [](const std::unique_ptr<RdKafka::TopicPartition>& x) {
            return x->offset();
        });

        // Query Kafka to populate the TopicPartitions with the desired offsets
        CHECK_KAFKA(consumer->position(toppar_ptrs), RdKafka::ERR_NO_ERROR, "Failed retrieve Kafka positions");

        auto positions = foreach_map(toppar_list, [](const std::unique_ptr<RdKafka::TopicPartition>& x) {
            return x->offset();
        });

        auto watermarks = foreach_map(toppar_list, [&consumer](const std::unique_ptr<RdKafka::TopicPartition>& x) {
            int64_t low;
            int64_t high;
            CHECK_KAFKA(consumer->query_watermark_offsets(x->topic(), x->partition(), &low, &high, 1000),
                        RdKafka::ERR_NO_ERROR,
                        "Failed retrieve Kafka watermark offsets");

            return std::make_tuple(low, high);
        });

        auto watermark_strs = foreach_map(watermarks, [](const auto& x) {
            return MORPHEUS_CONCAT_STR("(" << std::get<0>(x) << ", " << std::get<1>(x) << ")");
        });

        auto& ctx = mrc::runnable::Context::get_runtime_context();
        VLOG(10) << ctx.info()
                 << MORPHEUS_CONCAT_STR(
                        "   Topic: '" << topic->topic()
                                      << "', Parts: " << StringUtil::array_to_str(part_ids.begin(), part_ids.end())
                                      << ", Committed: " << StringUtil::array_to_str(committed.begin(), committed.end())
                                      << ", Positions: " << StringUtil::array_to_str(positions.begin(), positions.end())
                                      << ", Watermarks: "
                                      << StringUtil::array_to_str(watermark_strs.begin(), watermark_strs.end()));
    }

    return std::move(consumer);
}

cudf::io::table_with_metadata KafkaSourceStage::load_table(const std::string& buffer)
{
    auto options =
        cudf::io::json_reader_options::builder(cudf::io::source_info(buffer.c_str(), buffer.size())).lines(true);

    return cudf::io::read_json(options.build());
}

template <bool EnableFilter>
std::string concat_message_batch(std::vector<std::unique_ptr<RdKafka::Message>> const& message_batch)
{
    std::ostringstream buffer;

    for (auto& msg : message_batch)
    {
        auto s = static_cast<char*>(msg->payload());

        if constexpr (EnableFilter)
        {
            if (!nlohmann::json::accept(s))
            {
                LOG(ERROR) << "Failed to parse kafka message as json: " << s;
                continue;
            }
        }

        buffer << s << "\n";
    }

    return buffer.str();
}

std::shared_ptr<morpheus::MessageMeta> KafkaSourceStage::process_batch(
    std::vector<std::unique_ptr<RdKafka::Message>>&& message_batch)
{
    // concat the kafka json messages
    auto json_lines = !this->m_disable_pre_filtering ? concat_message_batch<true>(message_batch)
                                                     : concat_message_batch<false>(message_batch);

    // parse the json
    auto data_table = this->load_table(json_lines);

    // Next, create the message metadata. This gets reused for repeats
    return MessageMeta::create_from_cpp(std::move(data_table), 0);
}

// ************ KafkaStageInterfaceProxy ************ //
std::shared_ptr<mrc::segment::Object<KafkaSourceStage>> KafkaSourceStageInterfaceProxy::init_with_single_topic(
    mrc::segment::Builder& builder,
    const std::string& name,
    TensorIndex max_batch_size,
    std::string topic,
    uint32_t batch_timeout_ms,
    std::map<std::string, std::string> config,
    bool disable_commit,
    bool disable_pre_filtering,
    std::size_t stop_after,
    bool async_commits,
    std::optional<pybind11::function> oauth_callback)
{
    auto oauth_callback_cpp = KafkaSourceStageInterfaceProxy::make_kafka_oauth_callback(std::move(oauth_callback));

    auto stage = builder.construct_object<KafkaSourceStage>(name,
                                                            max_batch_size,
                                                            topic,
                                                            batch_timeout_ms,
                                                            config,
                                                            disable_commit,
                                                            disable_pre_filtering,
                                                            stop_after,
                                                            async_commits,
                                                            std::move(oauth_callback_cpp));

    return stage;
}

std::shared_ptr<mrc::segment::Object<KafkaSourceStage>> KafkaSourceStageInterfaceProxy::init_with_multiple_topics(
    mrc::segment::Builder& builder,
    const std::string& name,
    TensorIndex max_batch_size,
    std::vector<std::string> topics,
    uint32_t batch_timeout_ms,
    std::map<std::string, std::string> config,
    bool disable_commit,
    bool disable_pre_filtering,
    std::size_t stop_after,
    bool async_commits,
    std::optional<pybind11::function> oauth_callback)
{
    auto oauth_callback_cpp = KafkaSourceStageInterfaceProxy::make_kafka_oauth_callback(std::move(oauth_callback));

    auto stage = builder.construct_object<KafkaSourceStage>(name,
                                                            max_batch_size,
                                                            topics,
                                                            batch_timeout_ms,
                                                            config,
                                                            disable_commit,
                                                            disable_pre_filtering,
                                                            stop_after,
                                                            async_commits,
                                                            std::move(oauth_callback_cpp));

    return stage;
}

std::unique_ptr<KafkaOAuthCallback> KafkaSourceStageInterfaceProxy::make_kafka_oauth_callback(
    std::optional<pybind11::function>&& oauth_callback)
{
    if (oauth_callback == std::nullopt)
    {
        return static_cast<std::unique_ptr<KafkaOAuthCallback>>(nullptr);
    }

    auto oauth_callback_wrapped = mrc::pymrc::PyFuncWrapper(std::move(oauth_callback.value()));

    return std::make_unique<KafkaOAuthCallback>([oauth_callback_wrapped = std::move(oauth_callback_wrapped)]() {
        auto kvp_cpp = std::map<std::string, std::string>();
        auto kvp     = oauth_callback_wrapped.operator()<pybind11::dict>();
        for (auto [key, value] : kvp)
        {
            kvp_cpp[key.cast<std::string>()] = value.cast<std::string>();
        }
        return kvp_cpp;
    });
}

}  // namespace morpheus
