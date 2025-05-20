// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <assert.h>
#include <CL/cl.h>

namespace ov {
namespace intel_gpu {
class SubMemoryManager {
public:
    using ptr = std::shared_ptr<SubMemoryManager>;
    using cptr = const std::shared_ptr<SubMemoryManager>;
    struct MemoryInfo {
        void* send_buf;
        std::shared_ptr<void> buf;
        bool flag;
        bool all_gather_flag;
        bool all_gather_copy_flag;
        bool last_used;
        std::shared_ptr<cldnn::stream> stream_ptr;
        std::vector<cldnn::memory::ptr> recv_bufs;
        std::vector<void*> remote_mems;
        std::vector<bool> recv_flag;
        std::vector<cldnn::event::ptr> events;
        cldnn::memory::ptr output;
        cldnn::layout layout;
    };

    SubMemoryManager(int num_sub_streams) {
        assert(num_sub_streams);
         _num_sub_streams = num_sub_streams;
        MemoryInfo memory_info;
        memory_info.flag = false;
        memory_info.all_gather_flag = false;
        memory_info.all_gather_copy_flag = false;
        memory_info.last_used = false;
        memory_info.layout = cldnn::layout();
        memory_info.recv_bufs.assign(_num_sub_streams, nullptr);
        memory_info.remote_mems.assign(_num_sub_streams, nullptr);
        memory_info.recv_flag.assign(_num_sub_streams, false);
        memory_info.events.assign(_num_sub_streams, nullptr);
        std::vector<MemoryInfo> memorys;
        memorys.assign(_num_sub_streams, memory_info);
        _memorys_table.assign(2, memorys);
        _use_count.assign(2, 0);
        result = nullptr;
        updated_flag = false;
        step1_copy_done.store(0);
        step2_add_done.store(0);
        step3_concat_copy_done.store(0);
        step4_concat_copy_done.store(0);
        test_done.store(0);
    }

    int get_memory_id(int sub_stream_id) {
        for (int i = 0; i < 2; i++) {
            if (!_memorys_table[i][sub_stream_id].last_used) {
                return i;
            }
        }
        return -1;
    }

    void set_memory_used(int memory_id, int sub_stream_id) {
        _memorys_table[memory_id][sub_stream_id].last_used = true;
        _memorys_table[(memory_id + 1) % 2][sub_stream_id].last_used = false;
    }

    int _num_sub_streams;
    std::vector<std::vector<MemoryInfo>> _memorys_table;
    std::vector<size_t> _use_count;
    void* result;
    bool updated_flag;
    std::mutex _flagMutex;
    std::atomic<int> step1_copy_done;
    std::atomic<int> step2_add_done;
    std::atomic<int> step3_concat_copy_done;
    std::atomic<int> step4_concat_copy_done;
    std::atomic<int> test_done;
    cl_event user_events[2];
    cl_event step1_copy_events[2];
    cl_event step2_add_events[2];
    cl_event step3_gather_copy_events[2];
    cl_event step4_gather_copy_events[2];
    cl_mem xj_test_mem_1;
    cl_mem xj_test_mem_2;

    cldnn::event::ptr step1_copy_events_ptr[2];
    cldnn::event::ptr step2_add_events_ptr[2];
    cldnn::event::ptr step3_gather_copy_events_ptr[2];
    cldnn::event::ptr step4_gather_copy_events_ptr[2];
};
}  // namespace intel_gpu
}  // namespace ov