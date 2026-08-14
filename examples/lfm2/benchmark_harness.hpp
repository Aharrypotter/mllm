// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(__ANDROID__) || defined(__linux__)
#include <sched.h>
#endif

namespace mllm::examples::lfm2::benchmark {

inline std::optional<std::string> readTextFile(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) return std::nullopt;
  std::ostringstream contents;
  contents << stream.rdbuf();
  auto value = contents.str();
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
  return value;
}

inline std::optional<int64_t> readIntegerFile(const std::filesystem::path& path) {
  const auto text = readTextFile(path);
  if (!text.has_value()) return std::nullopt;
  try {
    size_t consumed = 0;
    const auto value = std::stoll(*text, &consumed);
    if (consumed != text->size()) return std::nullopt;
    return value;
  } catch (...) { return std::nullopt; }
}

inline std::vector<int> currentAffinityCpus() {
  std::vector<int> cpus;
#if defined(__ANDROID__) || defined(__linux__)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
  }
#endif
  return cpus;
}

inline nlohmann::json captureTelemetry() {
  using Json = nlohmann::json;
  Json snapshot = {
#if defined(__ANDROID__)
      {"platform", "android"},
#elif defined(__linux__)
      {"platform", "linux"},
#elif defined(__APPLE__)
      {"platform", "macos"},
#else
      {"platform", "other"},
#endif
      {"captured_epoch_us",
       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
  };

  const auto affinity = currentAffinityCpus();
  snapshot["affinity_cpus"] = affinity;
  snapshot["cpus"] = Json::array();
  snapshot["ceiling_vector"] = Json::array();
  snapshot["thermal_zones"] = Json::array();

  for (const int cpu : affinity) {
    const auto base = std::filesystem::path("/sys/devices/system/cpu") / ("cpu" + std::to_string(cpu));
    const auto cpufreq = base / "cpufreq";
    const auto online = cpu == 0 ? std::optional<int64_t>(1) : readIntegerFile(base / "online");
    const auto cpuinfo_max = readIntegerFile(cpufreq / "cpuinfo_max_freq");
    const auto scaling_max = readIntegerFile(cpufreq / "scaling_max_freq");
    const auto scaling_cur = readIntegerFile(cpufreq / "scaling_cur_freq");
    const auto governor = readTextFile(cpufreq / "scaling_governor");
    snapshot["cpus"].push_back({
        {"cpu", cpu},
        {"online", online.has_value() ? Json(*online) : Json(nullptr)},
        {"cpuinfo_max_freq", cpuinfo_max.has_value() ? Json(*cpuinfo_max) : Json(nullptr)},
        {"scaling_max_freq", scaling_max.has_value() ? Json(*scaling_max) : Json(nullptr)},
        {"scaling_cur_freq", scaling_cur.has_value() ? Json(*scaling_cur) : Json(nullptr)},
        {"scaling_governor", governor.has_value() ? Json(*governor) : Json(nullptr)},
    });
    snapshot["ceiling_vector"].push_back(scaling_max.has_value() ? Json(*scaling_max) : Json(nullptr));
  }

  const std::filesystem::path thermal_root("/sys/class/thermal");
  std::error_code error;
  if (std::filesystem::exists(thermal_root, error)) {
    std::vector<std::filesystem::path> zones;
    for (const auto& entry : std::filesystem::directory_iterator(thermal_root, error)) {
      if (entry.path().filename().string().starts_with("thermal_zone")) zones.push_back(entry.path());
    }
    std::sort(zones.begin(), zones.end());
    for (const auto& zone : zones) {
      const auto type = readTextFile(zone / "type");
      const auto temp = readIntegerFile(zone / "temp");
      snapshot["thermal_zones"].push_back({
          {"zone", zone.filename().string()},
          {"type", type.has_value() ? Json(*type) : Json(nullptr)},
          {"temp_milli_c", temp.has_value() ? Json(*temp) : Json(nullptr)},
      });
    }
  }
  return snapshot;
}

}  // namespace mllm::examples::lfm2::benchmark
