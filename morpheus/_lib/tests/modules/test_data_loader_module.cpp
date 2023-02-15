/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_modules.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/modules/data_loader_module.hpp"

#include <mrc/core/executor.hpp>
#include <mrc/engine/pipeline/ipipeline.hpp>
#include <mrc/modules/sample_modules.hpp>
#include <mrc/modules/segment_modules.hpp>
#include <mrc/node/rx_sink.hpp>
#include <mrc/node/rx_source.hpp>
#include <mrc/options/options.hpp>
#include <mrc/options/topology.hpp>
#include <mrc/pipeline/pipeline.hpp>
#include <mrc/segment/builder.hpp>
#include <nlohmann/json.hpp>

#include <fstream>

using namespace morpheus;
using namespace morpheus::test;

// TODO(Devin): Can't seem to get this to work, we lock up trying to grab the gil -- something going on with the fiber
// interactions.
// TEST_F(TestDataLoaderModule, EndToEndFileDataLoaderTest)
//{
//    using namespace mrc::modules;
//    using namespace mrc;
//
//    using sp_msg_meta_t = std::shared_ptr<MessageMeta>;
//    using sp_msg_ctrl_t = std::shared_ptr<MessageControl>;
//
//    std::size_t packet_count{0};
//
//    auto init_wrapper = [&packet_count](segment::Builder& builder) {
//        nlohmann::json config;
//        config["loaders"] = {"file"};
//
//        auto data_loader_module = builder.make_module<DataLoaderModule>("DataLoaderTest", config);
//
//        auto source = builder.make_source<sp_msg_ctrl_t>("source", [](rxcpp::subscriber<sp_msg_ctrl_t>& sub) {
//            std::string string_df = create_mock_dataframe({"col1", "col2", "col3"}, {"int32", "float32", "string"},
//            5);
//
//            char temp_file[] = "/tmp/morpheus_test_XXXXXXXX";
//            int fd           = mkstemp(temp_file);
//            if (fd == -1)
//            {
//                GTEST_SKIP() << "Failed to create temporary file, skipping test";
//            }
//
//            std::fstream data_file(temp_file, std::ios::out | std::ios::binary | std::ios::trunc);
//            data_file << string_df;
//            data_file.close();
//
//            nlohmann::json config;
//            config["loader_id"] = "file";
//            config["strategy"]  = "merge";
//            config["files"]     = {std::string(temp_file)};
//            if (sub.is_subscribed())
//            {
//                for (int i = 0; i < 10; i++)
//                {
//                    sub.on_next(std::make_shared<MessageControl>(config));
//                }
//            }
//
//            sub.on_completed();
//        });
//
//        builder.make_edge(source, data_loader_module->input_port("input"));
//        auto sink = builder.make_sink<sp_msg_meta_t>("sink", [&packet_count](sp_msg_meta_t input) {
//            packet_count++;
//            VLOG(30) << "Received message";
//        });
//
//        builder.make_edge(data_loader_module->output_port("output"), sink);
//    };
//
//    std::unique_ptr<pipeline::Pipeline> m_pipeline;
//    m_pipeline = pipeline::make_pipeline();
//
//    m_pipeline->make_segment("main", init_wrapper);
//
//    auto options = std::make_shared<Options>();
//    options->topology().user_cpuset("0-1");
//    options->topology().restrict_gpus(true);
//    // We're running an interpreter, and accessing python objects from multiple threads, will lock up if we use
//    // fibers.
//    options->engine_factories().set_default_engine_type(runnable::EngineType::Thread);
//
//    Executor executor(options);
//    executor.register_pipeline(std::move(m_pipeline));
//    executor.start();
//    executor.join();
//
//    EXPECT_EQ(packet_count, 10);
//}

TEST_F(TestDataLoaderModule, EndToEndGRPCDataLoaderTest)
{
    using namespace mrc::modules;
    using namespace mrc;

    using sp_msg_meta_t = std::shared_ptr<MessageMeta>;
    using sp_msg_ctrl_t = std::shared_ptr<MessageControl>;

    std::size_t packet_count{0};

    auto init_wrapper = [&packet_count](segment::Builder& builder) {
        nlohmann::json config;
        config["loaders"]       = {"grpc"};
        auto data_loader_module = builder.make_module<DataLoaderModule>("DataLoaderTest", config);

        auto source = builder.make_source<sp_msg_ctrl_t>("source", [](rxcpp::subscriber<sp_msg_ctrl_t>& sub) {
            if (sub.is_subscribed())
            {
                for (int i = 0; i < 10; i++)
                {
                    nlohmann::json config;
                    config["loader_id"] = "grpc";
                    sub.on_next(std::make_shared<MessageControl>(config));
                }
            }

            sub.on_completed();
        });

        builder.make_edge(source, data_loader_module->input_port("input"));
        auto sink = builder.make_sink<sp_msg_meta_t>("sink", [&packet_count](sp_msg_meta_t input) {
            packet_count++;
            VLOG(10) << "Received message";
        });

        builder.make_edge(data_loader_module->output_port("output"), sink);
    };

    std::unique_ptr<pipeline::Pipeline> m_pipeline;
    m_pipeline = pipeline::make_pipeline();

    m_pipeline->make_segment("main", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);
    options->engine_factories().set_default_engine_type(runnable::EngineType::Thread);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count, 10);
}

TEST_F(TestDataLoaderModule, EndToEndPayloadDataLoaderTest)
{
    using namespace mrc::modules;
    using namespace mrc;

    using sp_msg_meta_t = std::shared_ptr<MessageMeta>;
    using sp_msg_ctrl_t = std::shared_ptr<MessageControl>;

    std::size_t packet_count{0};

    auto init_wrapper = [&packet_count](segment::Builder& builder) {
        nlohmann::json config;
        config["loaders"]       = {"payload"};
        auto data_loader_module = builder.make_module<DataLoaderModule>("DataLoaderTest", config);

        auto source = builder.make_source<sp_msg_ctrl_t>("source", [](rxcpp::subscriber<sp_msg_ctrl_t>& sub) {
            if (sub.is_subscribed())
            {
                for (int i = 0; i < 10; i++)
                {
                    nlohmann::json config;
                    config["loader_id"] = "payload";
                    sub.on_next(std::make_shared<MessageControl>(config));
                }
            }

            sub.on_completed();
        });

        std::size_t x;
        builder.make_edge(source, data_loader_module->input_port("input"));
        auto sink = builder.make_sink<sp_msg_meta_t>("sink", [&packet_count](sp_msg_meta_t input) {
            packet_count++;
            VLOG(10) << "Received message";
        });

        builder.make_edge(data_loader_module->output_port("output"), sink);
    };

    std::unique_ptr<pipeline::Pipeline> m_pipeline;
    m_pipeline = pipeline::make_pipeline();

    m_pipeline->make_segment("main", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);
    options->engine_factories().set_default_engine_type(runnable::EngineType::Thread);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count, 10);
}

TEST_F(TestDataLoaderModule, EndToEndRESTDataLoaderTest)
{
    using namespace mrc::modules;
    using namespace mrc;

    using sp_msg_meta_t = std::shared_ptr<MessageMeta>;
    using sp_msg_ctrl_t = std::shared_ptr<MessageControl>;

    std::size_t packet_count{0};

    auto init_wrapper = [&packet_count](segment::Builder& builder) {
        nlohmann::json config;
        config["loaders"]       = {"rest"};
        auto data_loader_module = builder.make_module<DataLoaderModule>("DataLoaderTest", config);

        auto source = builder.make_source<sp_msg_ctrl_t>("source", [](rxcpp::subscriber<sp_msg_ctrl_t>& sub) {
            if (sub.is_subscribed())
            {
                for (int i = 0; i < 10; i++)
                {
                    nlohmann::json config;
                    config["loader_id"] = "rest";
                    sub.on_next(std::make_shared<MessageControl>(config));
                }
            }

            sub.on_completed();
        });

        builder.make_edge(source, data_loader_module->input_port("input"));
        auto sink = builder.make_sink<sp_msg_meta_t>("sink", [&packet_count](sp_msg_meta_t input) {
            packet_count++;
            VLOG(10) << "Received message";
        });

        builder.make_edge(data_loader_module->output_port("output"), sink);
    };

    std::unique_ptr<pipeline::Pipeline> m_pipeline;
    m_pipeline = pipeline::make_pipeline();

    m_pipeline->make_segment("main", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0");
    options->topology().restrict_gpus(true);
    options->engine_factories().set_default_engine_type(runnable::EngineType::Thread);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count, 10);
}