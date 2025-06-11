// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(all_reduce_broadcast_ref)( __global INPUT0_TYPE* input,
                           __global OUTPUT_TYPE* output,
                           __global INPUT0_TYPE* tmp_buffer)
{
    const uint global_id = get_global_id(0);
    tmp_buffer[global_id] = input[global_id];
}
