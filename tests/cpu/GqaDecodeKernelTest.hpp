// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// Focused oracle for the single-token grouped-query-attention decode kernel
// that reads native KV-head [B, H, S, D] cache views.
//
// The reference below is an independent scalar implementation that accumulates
// in double precision. It is deliberately not routed through the production
// kernel, so the vectorized QK dot product, the grouped P@V accumulation, and
// the strided cache addressing cannot validate themselves. Output tolerance is
// used for the reference comparison because the vector dot product reorders the
// float32 summation; slice equivalence and repeat stability are compared
// bitwise because the kernel promises a fixed accumulation order per head.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "mllm/backends/cpu/kernels/common/gqa_decode/fwd_bhsd.hpp"
#include "KernelTestHelper.hpp"

namespace gqa_decode_kernel_test {

using mllm::cpu::gqa_decode::BhsdStrides;
using mllm::cpu::gqa_decode::fwdBhsdFp32;

// Deterministic index-derived fill. No RNG, so every host reproduces the same
// bytes without carrying a seed through the evidence record.
inline float patternValue(std::size_t index, int salt) {
  const auto scaled = static_cast<float>((index * 37U + static_cast<unsigned>(salt) * 11U) % 251U);
  return (scaled - 125.0F) / 64.0F;
}

inline std::vector<float> makeBuffer(std::size_t count, int salt) {
  std::vector<float> buffer(count);
  for (std::size_t index = 0; index < count; ++index) { buffer[index] = patternValue(index, salt); }
  return buffer;
}

struct Geometry {
  int batch;
  int query_heads;
  int kv_heads;
  int kv_sequence;
  int qk_dim;
  int value_dim;
};

inline std::string describe(const Geometry& geometry) {
  return "B=" + std::to_string(geometry.batch) + " Hq=" + std::to_string(geometry.query_heads)
         + " Hkv=" + std::to_string(geometry.kv_heads) + " S=" + std::to_string(geometry.kv_sequence)
         + " D=" + std::to_string(geometry.qk_dim) + " Dv=" + std::to_string(geometry.value_dim);
}

// Contiguous [B, H, S, D] strides for a single-token query/output (S = 1) or a
// KV view whose sequence extent is `rows`.
inline BhsdStrides contiguousStrides(int heads, int rows, int dim) { return {heads * rows * dim, rows * dim, dim, 1}; }

inline std::size_t offset(const BhsdStrides& strides, int batch, int head, int row, int column) {
  return static_cast<std::size_t>(batch) * strides.batch + static_cast<std::size_t>(head) * strides.head
         + static_cast<std::size_t>(row) * strides.sequence + static_cast<std::size_t>(column) * strides.dimension;
}

// Independent scalar reference: per query head, softmax(q . k / sqrt(D)) @ v
// with double accumulation, honoring the same strided addressing contract.
inline void referenceDecode(const Geometry& g, const float* query, BhsdStrides qs, const float* key, BhsdStrides ks,
                            const float* value, BhsdStrides vs, std::vector<double>& output) {
  const int group_size = g.query_heads / g.kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(g.qk_dim));
  output.assign(static_cast<std::size_t>(g.batch) * g.query_heads * g.value_dim, 0.0);
  std::vector<double> scores(static_cast<std::size_t>(g.kv_sequence));
  for (int batch = 0; batch < g.batch; ++batch) {
    for (int head = 0; head < g.query_heads; ++head) {
      const int kv_head = head / group_size;
      double maximum = -std::numeric_limits<double>::infinity();
      for (int row = 0; row < g.kv_sequence; ++row) {
        double score = 0.0;
        for (int column = 0; column < g.qk_dim; ++column) {
          score += static_cast<double>(query[offset(qs, batch, head, 0, column)])
                   * static_cast<double>(key[offset(ks, batch, kv_head, row, column)]);
        }
        scores[static_cast<std::size_t>(row)] = score * scale;
        maximum = std::max(maximum, scores[static_cast<std::size_t>(row)]);
      }
      double denominator = 0.0;
      for (int row = 0; row < g.kv_sequence; ++row) {
        scores[static_cast<std::size_t>(row)] = std::exp(scores[static_cast<std::size_t>(row)] - maximum);
        denominator += scores[static_cast<std::size_t>(row)];
      }
      auto* out = output.data() + (static_cast<std::size_t>(batch) * g.query_heads + head) * g.value_dim;
      for (int row = 0; row < g.kv_sequence; ++row) {
        const double probability = scores[static_cast<std::size_t>(row)] / denominator;
        for (int column = 0; column < g.value_dim; ++column) {
          out[column] += probability * static_cast<double>(value[offset(vs, batch, kv_head, row, column)]);
        }
      }
    }
  }
}

struct Run {
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> output;
  std::vector<float> scratch;
  BhsdStrides qs{};
  BhsdStrides ks{};
  BhsdStrides vs{};
  BhsdStrides os{};
};

// Contiguous layouts; the KV view is exactly `kv_sequence` rows long.
inline Run makeContiguousRun(const Geometry& g) {
  Run run;
  run.qs = contiguousStrides(g.query_heads, 1, g.qk_dim);
  run.ks = contiguousStrides(g.kv_heads, g.kv_sequence, g.qk_dim);
  run.vs = contiguousStrides(g.kv_heads, g.kv_sequence, g.value_dim);
  run.os = contiguousStrides(g.query_heads, 1, g.value_dim);
  run.query = makeBuffer(static_cast<std::size_t>(g.batch) * g.query_heads * g.qk_dim, 1);
  run.key = makeBuffer(static_cast<std::size_t>(g.batch) * g.kv_heads * g.kv_sequence * g.qk_dim, 2);
  run.value = makeBuffer(static_cast<std::size_t>(g.batch) * g.kv_heads * g.kv_sequence * g.value_dim, 3);
  run.output.assign(static_cast<std::size_t>(g.batch) * g.query_heads * g.value_dim, 0.0F);
  run.scratch.assign(static_cast<std::size_t>(g.query_heads / g.kv_heads) * g.kv_sequence, 0.0F);
  return run;
}

inline bool invoke(const Geometry& g, Run& run) {
  return fwdBhsdFp32(g.batch, g.query_heads, g.kv_heads, g.kv_sequence, g.qk_dim, g.value_dim, run.query.data(), run.qs,
                     run.key.data(), run.ks, run.value.data(), run.vs, run.output.data(), run.os, run.scratch.data());
}

constexpr float kReferenceTolerance = 1.0e-4F;

inline void expectMatchesReference(const Geometry& g, const Run& run, const std::vector<double>& reference,
                                   const std::string& label) {
  for (int batch = 0; batch < g.batch; ++batch) {
    for (int head = 0; head < g.query_heads; ++head) {
      for (int column = 0; column < g.value_dim; ++column) {
        const auto expected = reference[(static_cast<std::size_t>(batch) * g.query_heads + head) * g.value_dim + column];
        const auto actual = run.output[offset(run.os, batch, head, 0, column)];
        ASSERT_NEAR(actual, expected, kReferenceTolerance)
            << label << " " << describe(g) << " b=" << batch << " h=" << head << " d=" << column;
      }
    }
  }
}

inline void testMatchesScalarReference(const std::vector<Geometry>& geometries) {
  for (const auto& g : geometries) {
    auto run = makeContiguousRun(g);
    ASSERT_TRUE(invoke(g, run)) << describe(g);
    std::vector<double> reference;
    referenceDecode(g, run.query.data(), run.qs, run.key.data(), run.ks, run.value.data(), run.vs, reference);
    expectMatchesReference(g, run, reference, "contiguous");
  }
}

// The production consumer hands the kernel a KV view inside a larger static
// cache ([B, Hkv, max_len, D] with only `kv_sequence` rows valid) and a query
// that is a transposed [B, 1, Hq, D] view. Both must be addressed through the
// strides rather than assumed contiguous.
inline void testNativeCacheAndTransposedQueryStrides(const Geometry& g, int cache_capacity) {
  ASSERT_GT(cache_capacity, g.kv_sequence);
  auto contiguous = makeContiguousRun(g);
  ASSERT_TRUE(invoke(g, contiguous)) << describe(g);

  Run strided;
  strided.ks = contiguousStrides(g.kv_heads, cache_capacity, g.qk_dim);
  strided.vs = contiguousStrides(g.kv_heads, cache_capacity, g.value_dim);
  // [B, 1, Hq, D] memory order: head stride D, batch stride Hq * D.
  strided.qs = {g.query_heads * g.qk_dim, g.qk_dim, g.query_heads * g.qk_dim, 1};
  strided.os = {g.query_heads * g.value_dim, g.value_dim, g.query_heads * g.value_dim, 1};
  strided.key = makeBuffer(static_cast<std::size_t>(g.batch) * g.kv_heads * cache_capacity * g.qk_dim, 7);
  strided.value = makeBuffer(static_cast<std::size_t>(g.batch) * g.kv_heads * cache_capacity * g.value_dim, 8);
  strided.query = contiguous.query;
  strided.output.assign(contiguous.output.size(), 0.0F);
  strided.scratch.assign(contiguous.scratch.size(), 0.0F);
  for (int batch = 0; batch < g.batch; ++batch) {
    for (int head = 0; head < g.kv_heads; ++head) {
      for (int row = 0; row < g.kv_sequence; ++row) {
        for (int column = 0; column < g.qk_dim; ++column) {
          strided.key[offset(strided.ks, batch, head, row, column)] =
              contiguous.key[offset(contiguous.ks, batch, head, row, column)];
        }
        for (int column = 0; column < g.value_dim; ++column) {
          strided.value[offset(strided.vs, batch, head, row, column)] =
              contiguous.value[offset(contiguous.vs, batch, head, row, column)];
        }
      }
    }
  }
  ASSERT_TRUE(invoke(g, strided)) << describe(g);

  std::vector<double> reference;
  referenceDecode(g, strided.query.data(), strided.qs, strided.key.data(), strided.ks, strided.value.data(), strided.vs,
                  reference);
  expectMatchesReference(g, strided, reference, "strided");
  for (int batch = 0; batch < g.batch; ++batch) {
    for (int head = 0; head < g.query_heads; ++head) {
      for (int column = 0; column < g.value_dim; ++column) {
        // Same bytes read in the same order: the strided view must be bitwise
        // identical to the contiguous computation, not merely close.
        ASSERT_EQ(strided.output[offset(strided.os, batch, head, 0, column)],
                  contiguous.output[offset(contiguous.os, batch, head, 0, column)])
            << describe(g) << " b=" << batch << " h=" << head << " d=" << column;
      }
    }
  }
}

// Every (batch, kv-head) group is scheduled independently and every query head
// accumulates its keys in increasing order, so a grouped call must reproduce
// the per-head single-KV-head call bitwise. This also proves the grouped
// scratch rows do not leak between heads or batches.
inline void testGroupedSlicesMatchSingleHeadCallsBitwise(const Geometry& g) {
  auto grouped = makeContiguousRun(g);
  ASSERT_TRUE(invoke(g, grouped)) << describe(g);
  const int group_size = g.query_heads / g.kv_heads;
  const Geometry single{1, 1, 1, g.kv_sequence, g.qk_dim, g.value_dim};
  for (int batch = 0; batch < g.batch; ++batch) {
    for (int head = 0; head < g.query_heads; ++head) {
      const int kv_head = head / group_size;
      Run one;
      one.qs = contiguousStrides(1, 1, g.qk_dim);
      one.ks = contiguousStrides(1, g.kv_sequence, g.qk_dim);
      one.vs = contiguousStrides(1, g.kv_sequence, g.value_dim);
      one.os = contiguousStrides(1, 1, g.value_dim);
      one.query.assign(grouped.query.begin() + static_cast<std::ptrdiff_t>(offset(grouped.qs, batch, head, 0, 0)),
                       grouped.query.begin() + static_cast<std::ptrdiff_t>(offset(grouped.qs, batch, head, 0, 0)) + g.qk_dim);
      const auto key_begin = static_cast<std::ptrdiff_t>(offset(grouped.ks, batch, kv_head, 0, 0));
      one.key.assign(grouped.key.begin() + key_begin,
                     grouped.key.begin() + key_begin + static_cast<std::ptrdiff_t>(g.kv_sequence) * g.qk_dim);
      const auto value_begin = static_cast<std::ptrdiff_t>(offset(grouped.vs, batch, kv_head, 0, 0));
      one.value.assign(grouped.value.begin() + value_begin,
                       grouped.value.begin() + value_begin + static_cast<std::ptrdiff_t>(g.kv_sequence) * g.value_dim);
      one.output.assign(static_cast<std::size_t>(g.value_dim), 0.0F);
      one.scratch.assign(static_cast<std::size_t>(g.kv_sequence), 0.0F);
      ASSERT_TRUE(invoke(single, one)) << describe(g);
      for (int column = 0; column < g.value_dim; ++column) {
        ASSERT_EQ(grouped.output[offset(grouped.os, batch, head, 0, column)], one.output[static_cast<std::size_t>(column)])
            << describe(g) << " b=" << batch << " h=" << head << " d=" << column;
      }
    }
  }
}

inline void testRepeatedCallsAreBitwiseStable(const Geometry& g, int repeats) {
  auto first = makeContiguousRun(g);
  ASSERT_TRUE(invoke(g, first)) << describe(g);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto again = makeContiguousRun(g);
    ASSERT_TRUE(invoke(g, again)) << describe(g);
    ASSERT_EQ(again.output, first.output) << describe(g) << " repeat=" << repeat;
  }
}

// Invalid geometry, null buffers, and unsupported strides must be rejected
// before any byte of the output is touched; the backend op then takes the
// reference path.
inline void testRejectsInvalidGeometryAndStrides() {
  const Geometry g{1, 4, 2, 3, 8, 8};
  auto run = makeContiguousRun(g);
  ASSERT_TRUE(invoke(g, run));

  auto expectRejected = [&](const std::string& label, auto&& mutate) {
    auto probe = makeContiguousRun(g);
    Geometry geometry = g;
    const float sentinel = 123.5F;
    probe.output.assign(probe.output.size(), sentinel);
    float* query = probe.query.data();
    float* key = probe.key.data();
    float* value = probe.value.data();
    float* output = probe.output.data();
    float* scratch = probe.scratch.data();
    mutate(geometry, probe, query, key, value, output, scratch);
    EXPECT_FALSE(fwdBhsdFp32(geometry.batch, geometry.query_heads, geometry.kv_heads, geometry.kv_sequence, geometry.qk_dim,
                             geometry.value_dim, query, probe.qs, key, probe.ks, value, probe.vs, output, probe.os, scratch))
        << label;
    for (const auto element : probe.output) { EXPECT_EQ(element, sentinel) << label << " touched the output"; }
  };
  using Mutator = void (*)(Geometry&, Run&, float*&, float*&, float*&, float*&, float*&);
  expectRejected("query heads not a multiple of kv heads",
                 static_cast<Mutator>(
                     [](Geometry& geometry, Run&, float*&, float*&, float*&, float*&, float*&) { geometry.query_heads = 3; }));
  expectRejected("zero kv sequence", static_cast<Mutator>([](Geometry& geometry, Run&, float*&, float*&, float*&, float*&,
                                                             float*&) { geometry.kv_sequence = 0; }));
  expectRejected("zero qk dim", static_cast<Mutator>([](Geometry& geometry, Run&, float*&, float*&, float*&, float*&, float*&) {
                   geometry.qk_dim = 0;
                 }));
  expectRejected("null query", static_cast<Mutator>([](Geometry&, Run&, float*& query, float*&, float*&, float*&, float*&) {
                   query = nullptr;
                 }));
  expectRejected("null key",
                 static_cast<Mutator>([](Geometry&, Run&, float*&, float*& key, float*&, float*&, float*&) { key = nullptr; }));
  expectRejected("null value", static_cast<Mutator>([](Geometry&, Run&, float*&, float*&, float*& value, float*&, float*&) {
                   value = nullptr;
                 }));
  expectRejected("null scratch", static_cast<Mutator>([](Geometry&, Run&, float*&, float*&, float*&, float*&, float*& scratch) {
                   scratch = nullptr;
                 }));
  expectRejected("non-unit key dimension stride", static_cast<Mutator>([](Geometry&, Run& probe, float*&, float*&, float*&,
                                                                          float*&, float*&) { probe.ks.dimension = 2; }));
  expectRejected("non-unit output dimension stride", static_cast<Mutator>([](Geometry&, Run& probe, float*&, float*&, float*&,
                                                                             float*&, float*&) { probe.os.dimension = 2; }));
  expectRejected("non-positive value sequence stride", static_cast<Mutator>([](Geometry&, Run& probe, float*&, float*&, float*&,
                                                                               float*&, float*&) { probe.vs.sequence = 0; }));
}

}  // namespace gqa_decode_kernel_test

class GqaDecodeKernelTest : public KernelTest {
 public:
  using Geometry = gqa_decode_kernel_test::Geometry;
  static void testMatchesScalarReference(const std::vector<Geometry>& geometries) {
    gqa_decode_kernel_test::testMatchesScalarReference(geometries);
  }
  static void testNativeCacheAndTransposedQueryStrides(const Geometry& geometry, int cache_capacity) {
    gqa_decode_kernel_test::testNativeCacheAndTransposedQueryStrides(geometry, cache_capacity);
  }
  static void testGroupedSlicesMatchSingleHeadCallsBitwise(const Geometry& geometry) {
    gqa_decode_kernel_test::testGroupedSlicesMatchSingleHeadCallsBitwise(geometry);
  }
  static void testRepeatedCallsAreBitwiseStable(const Geometry& geometry, int repeats) {
    gqa_decode_kernel_test::testRepeatedCallsAreBitwiseStable(geometry, repeats);
  }
  static void testRejectsInvalidGeometryAndStrides() { gqa_decode_kernel_test::testRejectsInvalidGeometryAndStrides(); }
};
