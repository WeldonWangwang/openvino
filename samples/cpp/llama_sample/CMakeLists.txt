# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
              
ov_add_sample(NAME llama_sampple
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader ie_samples_utils)
