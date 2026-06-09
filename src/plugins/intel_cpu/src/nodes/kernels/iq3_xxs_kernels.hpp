// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Public IQ3_XXS dequantization + fused dot-product kernels.
//
// These kernels are the single source of truth for the IQ3_XXS compute path
// in the CPU plugin. They are shared between:
//   * node::IQ3XXSLinear  (legacy native op, GenAI bridge path)
//   * CompressedConstantFCExecutor  (new path, plugged into FullyConnected)
//
// The decode logic is a byte-for-byte translation of
//     ggml-quants.c::dequantize_row_iq3_xxs()
// from llama.cpp / ggml. Constant tables (256-entry `iq3xxs_grid` and
// 128-entry `ksigns_iq2xs`) are copied verbatim from `ggml-common.h`,
// so output is bit-exact with llama.cpp for any IQ3_XXS weight blob.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ov::intel_cpu::kernels::iq3_xxs {

// IQ3_XXS layout invariants (per ggml spec).
constexpr size_t QK_K = 256;          ///< weights per super-block
constexpr size_t BLOCK_BYTES = 98;    ///< bytes per super-block

/// Decode one IQ3_XXS weight row (`blocks_per_row * QK_K` weights) into a
/// contiguous f32 buffer.
///
/// \param w_row           Pointer to the start of the row's compressed bytes
///                        (size = `blocks_per_row * BLOCK_BYTES`).
/// \param out             Pre-allocated f32 buffer of size `blocks_per_row * QK_K`.
/// \param blocks_per_row  Number of 256-weight super-blocks in this row
///                        (== K / 256).
///
/// Bit-exact with `ggml::dequantize_row_iq3_xxs`. Thread-safe and side-effect-free.
void decode_iq3_xxs_row(const uint8_t* w_row, float* out, size_t blocks_per_row);

/// Fused IQ3_XXS dequantization + dense MatMul:  out[r, n] = sum_k act[r, k] * W[n, k]
/// where W is stored as an IQ3_XXS compressed blob.
///
/// Internally parallelized over the N (output channel) dimension with
/// `std::thread`. Each thread decodes a slab of N rows once into a small
/// per-thread buffer of K floats and reuses them across all `rows` activation
/// rows, so weight memory never grows.
///
/// \param act           Activation tensor [rows, K] in f32.
/// \param weight_blob   IQ3_XXS compressed weight blob, laid out as N rows
///                      of `blocks_per_row * BLOCK_BYTES` bytes each, with
///                      `blocks_per_row = K / QK_K`.
/// \param out           Pre-allocated output tensor [rows, N] in f32.
/// \param rows          Number of activation rows (collapsed batch * seq dims).
/// \param K             Reduction dimension. Must be a multiple of `QK_K`.
/// \param N             Output channel count.
/// \param nthreads_hint Optional thread count. 0 means use
///                      `std::thread::hardware_concurrency()`.
void iq3_xxs_fc(const float* act,
                const uint8_t* weight_blob,
                float* out,
                size_t rows,
                size_t K,
                size_t N,
                size_t nthreads_hint = 0);

}  // namespace ov::intel_cpu::kernels::iq3_xxs
