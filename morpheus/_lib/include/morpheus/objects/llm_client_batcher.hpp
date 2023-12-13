#include <glog/logging.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <iterator>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#pragma once

using namespace std::chrono_literals;

using time_point_t = std::chrono::time_point<std::chrono::system_clock>;

struct LLMClientRequest
{
    LLMClientRequest()                              = delete;
    LLMClientRequest(const LLMClientRequest& other) = delete;
    LLMClientRequest(std::vector<std::string> queries,
                     std::promise<std::vector<std::string>>&& promise,
                     time_point_t requestsed_at) :
      m_queries(std::move(queries)),
      m_promise(std::move(promise)),
      m_requested_at(requestsed_at)
    {}
    LLMClientRequest(LLMClientRequest&& other) = default;

    std::vector<std::string> m_queries;
    std::promise<std::vector<std::string>> m_promise;
    time_point_t m_requested_at;
};

class LLMClientBatcher
{
  public:
    LLMClientBatcher()
    {
        m_batching_thread = std::thread(&LLMClientBatcher::batch_processing_loop, this);
    }

    ~LLMClientBatcher()
    {
        if (m_batching_thread.joinable())
        {
            m_loop_ct = true;
            m_batching_thread.join();
        }
    }

    std::future<std::vector<std::string>> generate(std::vector<std::string> queries)
    {
        std::promise<std::vector<std::string>> response_promise;

        auto responses_future = response_promise.get_future();

        {
            auto guard   = std::lock_guard(this->m_batches_mutex);
            auto request = LLMClientRequest{
                std::move(queries),
                std::move(response_promise),
                std::chrono::system_clock::now()  //
            };
            m_requests.emplace_back(std::move(request));
        }

        return responses_future;
    }

  private:
    void batch_processing_loop()
    {
        while (not m_loop_ct)
        {
            std::this_thread::sleep_for(m_loop_wait);

            if (m_loop_ct)
            {
                break;
            }

            auto lock = std::unique_lock(this->m_batches_mutex);

            if (m_requests.size() == 0)
            {
                continue;
            }

            auto batch_size = 0;

            for (auto& request : m_requests)
            {
                batch_size += request.m_queries.size();
            }

            auto oldest_request_wait = std::chrono::system_clock::now() - m_requests.begin()->m_requested_at;

            // only go to next step if our batch size is big enough, our wait time is too long, or we're exiting.
            if (batch_size < m_min_batch_size and oldest_request_wait < m_max_wait and not m_loop_ct)
            {
                continue;
            }

            auto batch = std::move(m_requests);
            m_requests.clear();

            lock.unlock();

            LOG(INFO) << "LLMClientBatcher: processing " << batch.size() << " batches with " << batch_size
                      << " total requests";

            for (auto& request : batch)
            {
                std::vector<std::string> response{};

                for (auto& query : request.m_queries)
                {
                    response.emplace_back("cool story");
                }

                request.m_promise.set_value(std::move(response));
            }
        }

        try
        {
            throw std::runtime_error("batcher destroyed");
        } catch (...)
        {
            auto lock  = std::unique_lock(m_batches_mutex);
            auto batch = std::move(m_requests);
            m_requests.clear();
            lock.unlock();
            for (auto& request : batch)
            {
                LOG(ERROR) << "setting exception";
                request.m_promise.set_exception(std::current_exception());
            }
        }
    }

    std::vector<LLMClientRequest> m_requests{};
    std::thread m_batching_thread;
    std::mutex m_batches_mutex{};
    uint32_t m_min_batch_size                 = 10;
    std::chrono::duration<long> m_max_wait    = 1s;
    std::chrono::duration<double> m_loop_wait = 0.1s;
    std::atomic<bool> m_loop_ct               = false;
};
