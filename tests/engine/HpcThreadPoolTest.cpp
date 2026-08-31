#include <array>
#include <atomic>

#include <gtest/gtest.h>

#include "mllm/engine/HpcThreadPool.hpp"

namespace mllm {
namespace {

TEST(HpcThreadPoolTest, RunsEveryIterationForNonDivisibleStep) {
  HpcThreadPool pool(4);
  pool.activate();
  const int task_slot = pool.acquireTaskSlot();
  ASSERT_GE(task_slot, 0);

  std::array<std::atomic<int>, 32> visits;
  for (auto& visit : visits) { visit.store(0); }
  std::atomic<int> invalid_indices{0};

  HpcThreadPoolTask task;
  task.start = 0;
  task.end = 29;
  task.step = 4;
  task.func = [&](int index) {
    if (index < 0 || index >= static_cast<int>(visits.size())) {
      invalid_indices.fetch_add(1);
      return;
    }
    visits[index].fetch_add(1);
  };

  pool.push(std::move(task), task_slot);

  EXPECT_EQ(invalid_indices.load(), 0);
  for (int index = 0; index < static_cast<int>(visits.size()); ++index) {
    const bool expected = index <= 28 && index % 4 == 0;
    EXPECT_EQ(visits[index].load(), expected ? 1 : 0) << "index=" << index;
  }

  pool.releaseTaskSlot(task_slot);
  pool.deactivate();
}

}  // namespace
}  // namespace mllm
