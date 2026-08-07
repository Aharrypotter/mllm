// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>

#include "benchmark_harness.hpp"

namespace {

nlohmann::json validSnapshot() {
  return {
      {"platform", "android"},
      {"affinity_cpus", {4, 5, 6, 7}},
      {"ceiling_vector", {3200000, 3200000, 4320000, 4320000}},
      {"cpus",
       {
           {{"cpu", 4},
            {"online", 1},
            {"cpuinfo_max_freq", 3200000},
            {"scaling_max_freq", 3200000},
            {"scaling_cur_freq", 2500000},
            {"scaling_governor", "schedutil"}},
           {{"cpu", 5},
            {"online", 1},
            {"cpuinfo_max_freq", 3200000},
            {"scaling_max_freq", 3200000},
            {"scaling_cur_freq", 2500000},
            {"scaling_governor", "schedutil"}},
           {{"cpu", 6},
            {"online", 1},
            {"cpuinfo_max_freq", 4320000},
            {"scaling_max_freq", 4320000},
            {"scaling_cur_freq", 3000000},
            {"scaling_governor", "schedutil"}},
           {{"cpu", 7},
            {"online", 1},
            {"cpuinfo_max_freq", 4320000},
            {"scaling_max_freq", 4320000},
            {"scaling_cur_freq", 3000000},
            {"scaling_governor", "schedutil"}},
       }},
  };
}

TEST(Qwen35BenchmarkHarnessTest, AcceptsCompleteStableTelemetry) {
  const auto before = validSnapshot();
  auto after = before;
  after["cpus"][0]["scaling_cur_freq"] = 1800000;

  EXPECT_TRUE(mllm::examples::qwen3_5::benchmark::validateRequiredTelemetry(before).empty());
  EXPECT_TRUE(mllm::examples::qwen3_5::benchmark::validateStableTelemetry(before, after).empty());
}

TEST(Qwen35BenchmarkHarnessTest, RejectsMissingCeiling) {
  auto snapshot = validSnapshot();
  snapshot["cpus"][2]["scaling_max_freq"] = nullptr;
  snapshot["ceiling_vector"][2] = nullptr;

  const auto errors = mllm::examples::qwen3_5::benchmark::validateRequiredTelemetry(snapshot);
  EXPECT_NE(std::find(errors.begin(), errors.end(), "cpu6_missing_scaling_max_freq"), errors.end());
}

TEST(Qwen35BenchmarkHarnessTest, RejectsAffinityCeilingOnlineAndGovernorDrift) {
  const auto before = validSnapshot();
  auto after = before;
  after["affinity_cpus"] = {3, 4, 5, 6};
  after["ceiling_vector"][3] = 1689600;
  after["cpus"][1]["online"] = 0;
  after["cpus"][2]["scaling_governor"] = "powersave";

  const auto errors = mllm::examples::qwen3_5::benchmark::validateStableTelemetry(before, after);
  EXPECT_EQ(errors, (std::vector<std::string>{"affinity_changed", "ceiling_vector_changed", "online_vector_changed",
                                              "governor_vector_changed"}));
}

TEST(Qwen35BenchmarkHarnessTest, MacCaptureFailsClosedWhenTelemetryIsRequired) {
#if defined(__APPLE__)
  const auto errors =
      mllm::examples::qwen3_5::benchmark::validateRequiredTelemetry(mllm::examples::qwen3_5::benchmark::captureTelemetry());
  EXPECT_NE(std::find(errors.begin(), errors.end(), "unsupported_telemetry_platform"), errors.end());
#else
  GTEST_SKIP() << "macOS-only telemetry rejection check";
#endif
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
