// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "mllm/mllm.hpp"

namespace {

class CPUContiguousOpTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

TEST_F(CPUContiguousOpTest, CopiesContiguousAndStridedViews) {
  using namespace mllm;  // NOLINT

  auto input = Tensor::empty({2, 3, 4}, kFloat32, kCPU).alloc();
  std::iota(input.ptr<float>(), input.ptr<float>() + input.numel(), 0.0F);

  const auto row_slice = input[{kAll, kAll, {1, 3}}].contiguous();
  ASSERT_EQ(row_slice.shape(), (Tensor::shape_t{2, 3, 2}));
  const std::vector<float> expected_slice = {
      1.0F, 2.0F, 5.0F, 6.0F, 9.0F, 10.0F, 13.0F, 14.0F, 17.0F, 18.0F, 21.0F, 22.0F,
  };
  EXPECT_EQ(std::vector<float>(row_slice.ptr<float>(), row_slice.ptr<float>() + row_slice.numel()), expected_slice);

  const auto contiguous_copy = input.contiguous();
  EXPECT_EQ(std::vector<float>(contiguous_copy.ptr<float>(), contiguous_copy.ptr<float>() + contiguous_copy.numel()),
            std::vector<float>(input.ptr<float>(), input.ptr<float>() + input.numel()));

  const auto strided_slice = input[{kAll, kAll, {0, 4, 2}}].contiguous();
  ASSERT_EQ(strided_slice.shape(), (Tensor::shape_t{2, 3, 2}));
  const std::vector<float> expected_strided_slice = {
      0.0F, 2.0F, 4.0F, 6.0F, 8.0F, 10.0F, 12.0F, 14.0F, 16.0F, 18.0F, 20.0F, 22.0F,
  };
  EXPECT_EQ(std::vector<float>(strided_slice.ptr<float>(), strided_slice.ptr<float>() + strided_slice.numel()),
            expected_strided_slice);
}

TEST_F(CPUContiguousOpTest, CopiesRankZeroScalar) {
  using namespace mllm;  // NOLINT

  auto scalar = Tensor::empty({}, kFloat32, kCPU).alloc();
  scalar.ptr<float>()[0] = 3.25F;

  const auto copy = scalar.contiguous();
  EXPECT_TRUE(copy.shape().empty());
  ASSERT_EQ(copy.numel(), 1);
  EXPECT_FLOAT_EQ(copy.item<float>(), 3.25F);
}

}  // namespace
