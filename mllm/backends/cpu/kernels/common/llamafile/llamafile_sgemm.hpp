// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Created by Rongjie Yi on 24-07-23.
//

#pragma once

#include <cstdint>

#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

bool llamafile_sgemm(int64_t m, int64_t n, int64_t k, const void* A, int64_t lda, const void* B, int64_t ldb, void* C,
                     int64_t ldc, int ith, int nth, DataTypes Atype, DataTypes Btype, DataTypes Ctype, void* bias = nullptr,
                     DataTypes BiasType = kFloat32);

bool check_llamafile_sgemm(int64_t m, int64_t n, int64_t k, DataTypes Atype, DataTypes Btype, DataTypes Ctype, int64_t lda,
                           int64_t ldb, int64_t ldc);

#if defined(__ARM_NEON)
// Key and query rows are contiguous, with FP32 accumulation and an N-wide output row.
void llamafile_causal_scaled_qk_f32(int64_t keys, int64_t queries, int64_t width, const float* key, const float* query,
                                    float* out, float scale, int ith, int nth);
#endif

}  // namespace mllm::cpu
