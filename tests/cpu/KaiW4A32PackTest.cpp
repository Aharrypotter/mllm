// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "mllm/backends/cpu/kernels/common/kai_w4a32_pack.hpp"

namespace {

constexpr auto kTileName = "qai8dxp1x8_qsi4c32p8x8_1x8x32";

TEST(KaiW4A32PackTest, PacksDeterministicallyOnHost) {
  constexpr int kOutChannels = 17;
  constexpr int kInChannels = 64;

  std::vector<float> weights(kOutChannels * kInChannels);
  for (int index = 0; index < static_cast<int>(weights.size()); ++index) {
    weights[index] =
        std::sin(static_cast<float>(index) * 0.071F) * 0.30F - std::cos(static_cast<float>(index) * 0.019F) * 0.05F;
  }

  const auto packed_size = mllm::cpu::common::kaiW4A32PackedSize(kOutChannels, kInChannels, kTileName);
  EXPECT_EQ(packed_size, 1056);

  std::vector<std::uint8_t> first(packed_size, 0xA5);
  std::vector<std::uint8_t> second(packed_size, 0x5A);
  mllm::cpu::common::kaiW4A32QuantizeAndPack(first.data(), weights.data(), nullptr, kOutChannels, kInChannels, kTileName);
  mllm::cpu::common::kaiW4A32QuantizeAndPack(second.data(), weights.data(), nullptr, kOutChannels, kInChannels, kTileName);

  EXPECT_EQ(first, second);
  EXPECT_TRUE(std::any_of(first.begin(), first.end(), [](std::uint8_t value) { return value != 0; }));
}

TEST(KaiW4A32PackTest, RejectsInvalidDimensionsAndTile) {
  EXPECT_THROW(mllm::cpu::common::kaiW4A32PackedSize(17, 63, kTileName), std::invalid_argument);
  EXPECT_THROW(mllm::cpu::common::kaiW4A32PackedSize(17, 64, "unsupported"), std::invalid_argument);
}

}  // namespace
