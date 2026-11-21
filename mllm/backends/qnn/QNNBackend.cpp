#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "QnnLog.h"
#include "HTP/QnnHtpContext.h"

#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/op/QNNCastTypeOp.hpp"
#include "mllm/backends/qnn/op/QNNElewiseOp.hpp"
#include "mllm/backends/qnn/op/QNNEmbeddingOp.hpp"
#include "mllm/backends/qnn/op/QNNGraphOp.hpp"
#include "mllm/backends/qnn/op/QNNLinearOp.hpp"
#include "mllm/backends/qnn/op/QNNParamOp.hpp"
#include "mllm/backends/qnn/op/QNNRMSNormOp.hpp"
#include "mllm/backends/qnn/op/QNNSiLUOp.hpp"
#include "mllm/backends/qnn/op/QNNTransposeOp.hpp"
#include "mllm/backends/qnn/op/QNNViewOp.hpp"
#include "mllm/backends/qnn/op/QNNX2XOp.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

namespace {

bool sequenceTileBufferPoolEnabled() {
  static const bool enabled = [] {
    const auto* value = std::getenv("MLLM_QNN_SEQUENCE_TILE_BUFFER_POOL");
    return value && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

constexpr int32_t kSequenceBatchSize = 2;

bool sequenceBatchRequested(std::string_view graph) {
  static const char* selection = [] {
    const auto* graphs = std::getenv("MLLM_QNN_SEQUENCE_BATCH2_GRAPHS");
    return graphs ? graphs : std::getenv("MLLM_QNN_SEQUENCE_BATCH2_SHADOW_GRAPHS");
  }();
  if (!selection) { return false; }
  for (std::string_view names(selection); !names.empty();) {
    const auto end = names.find(',');
    if (names.substr(0, end) == graph) { return true; }
    if (end == std::string_view::npos) { break; }
    names.remove_prefix(end + 1);
  }
  return false;
}

bool supportsSequenceBatch(const QNNSequenceTilePlan& plan) {
  return plan.tile_count == kSequenceBatchSize && plan.axis > 0 && plan.compiled_shape[0] == 1;
}

template <bool Pack>
void copyQnnSequenceBatch(Tensor& destination, const Tensor& source, const QNNSequenceTilePlan& plan) {
  size_t groups = 1;
  for (size_t axis = 1; axis < plan.axis; ++axis) { groups *= plan.compiled_shape[axis]; }
  auto* destination_bytes = destination.ptr<uint8_t>();
  const auto* source_bytes = source.ptr<uint8_t>();
  if (groups == 1) {
    std::memcpy(destination_bytes, source_bytes, source.bytes());
    return;
  }

  // Logical Q/K/V keep both sequence tiles within each head. QNN's batch
  // layout keeps all heads of one tile together.
  const size_t span_bytes = source.bytes() / (kSequenceBatchSize * groups);
  for (size_t tile = 0; tile < kSequenceBatchSize; ++tile) {
    for (size_t group = 0; group < groups; ++group) {
      const size_t logical_offset = (group * kSequenceBatchSize + tile) * span_bytes;
      const size_t batch_offset = (tile * groups + group) * span_bytes;
      std::memcpy(destination_bytes + (Pack ? batch_offset : logical_offset),
                  source_bytes + (Pack ? logical_offset : batch_offset), span_bytes);
    }
  }
}

}  // namespace

QNNBackend::QNNBackend() : Backend(kQNN, createQNNAllocator()) {
  // register ops
  regOpFactory<QNNAddOpFactory, QNNMulOpFactory, QNNGraphBeginOpFactory, QNNGraphEndOpFactory, QNNLinearOpFactory,
               QNNViewOpFactory, QNNRMSNormOpFactory, QNNTransposeOpFactory, QNNX2XOpFactory, QNNCastTypeOpFactory,
               QNNParamOpFactory, QNNSiLUOpFactory, QNNEmbeddingOpFactory>();

  QnnLog_Level_t qnnLogLevel = QNN_LOG_LEVEL_ERROR;  // default QNN log level
  profilingLevel_ = ProfilingLevel::OFF;
  debug_ = false;  // when set true, NATIVE tensor will be regared as APP_READ tensor

  // Load QNN libraries and hold handles for lifecycle management
  auto [qnnSuccess, qnnHandle] = loadQNNSymbol();
  if (!qnnSuccess) { MLLM_ERROR_EXIT(ExitCode::kQnnError, "Failed to load QNN symbols"); }
  qnnHtpLibHandle_ = qnnHandle;
  MLLM_INFO("QNN symbols loaded successfully");

  auto [sysSuccess, sysHandle] = loadQNNSystemSymbol();
  if (!sysSuccess) { MLLM_ERROR_EXIT(ExitCode::kQnnError, "Failed to load QNN System symbols"); }
  qnnSystemLibHandle_ = sysHandle;
  MLLM_INFO("QNN System symbols loaded successfully");

  runtime_ = QNNRuntime::create(profilingLevel_, qnnLogLevel);
  if (!runtime_) {
    MLLM_ERROR_EXIT(ExitCode::kQnnError, "Failed to create QNN Runtime");
  } else {
    MLLM_INFO("QNN Runtime created successfully");
  }

  // check QNN capability, detect QNN features for future use
  char* backendBuildId{nullptr};
  if (QNN_SUCCESS != runtime_->qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
    MLLM_ERROR("Unable to get build Id from the backend.");
  }
  MLLM_INFO("QNN Backend Build Id: {}", backendBuildId == nullptr ? "" : backendBuildId);
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports tensor sparsity");
  }
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports dynamic dimensions");
  }
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports early termination");
  }

  // set performance parameters for better performance on HTP
  perf_ = QNNPerf::create(&runtime_->qnnInterface);
  perf_->setPowerConfigBurst();
  perf_->setRpcLatencyAndPolling();
  MLLM_INFO("QNN Perf created successfully");
}

QNNBackend::~QNNBackend() {
  // Cleanup order is critical - we hold all QNN library handles to control unload order:
  // 1. Allocator shutdown (memDeRegister + rpcmem_free) - needs QNN alive
  // 2. Clear models - tensor destructors try to free but allocator is shut down
  // 3. Perf cleanup - needs QNN HTP infrastructure alive
  // 4. Runtime cleanup - frees QNN backend/device handles
  // 5. Allocator reset - dlcloses libcdsprpc.so (held by allocator)
  // 6. Close QNN libraries - libQnnSystem.so first, then libQnnHtp.so

  // 1. Properly shutdown allocator while QNN is still alive
  //    This calls memDeRegister and rpcmem_free safely
  if (allocator_) {
    auto* qnnAllocator = dynamic_cast<QNNAllocator*>(allocator_.get());
    if (qnnAllocator) { qnnAllocator->shutdown(); }
  }

  // 2. Clear models - tensor destructors will call free() but they're now no-ops
  qnnModels_.clear();
  qnnModelIndexMap_.clear();

  // 3. Cleanup perf while QNN HTP infrastructure is still alive
  if (perf_) { perf_->shutdown(); }
  perf_.reset();

  // 4. Cleanup runtime - frees QNN backend/device handles
  runtime_->qnnInterface.contextFree(context_, nullptr);
  context_ = nullptr;
  runtime_.reset();

  // 5. Reset allocator - will dlclose libcdsprpc.so since shutdown() was already called
  allocator_.reset();

  // 6. Close QNN libraries in reverse order of dependency
  if (qnnSystemLibHandle_) {
    dlclose(qnnSystemLibHandle_);
    qnnSystemLibHandle_ = nullptr;
  }
  if (qnnHtpLibHandle_) {
    dlclose(qnnHtpLibHandle_);
    qnnHtpLibHandle_ = nullptr;
  }
}

QNNPerf::QNNPerf(const QNN_INTERFACE_VER_TYPE* qnnInterface) {
  assert(qnnInterface != nullptr);
  qnnInterface_ = qnnInterface;

  QnnDevice_Infrastructure_t deviceInfra = nullptr;
  CALL_QNN(qnnInterface_->deviceGetInfrastructure(&deviceInfra));
  QnnHtpDevice_Infrastructure_t* htpInfra = static_cast<QnnHtpDevice_Infrastructure_t*>(deviceInfra);
  perfInfra_ = htpInfra->perfInfra;

  uint32_t deviceId = 0;
  uint32_t coreId = 0;
  CALL_QNN(perfInfra_.createPowerConfigId(deviceId, coreId, &powerConfigId_));

  powerConfigBurst_ = {
      .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
      .dcvsV3Config =
          {
              .contextId = powerConfigId_,  // use the power config id created
              .setDcvsEnable = 1,
              .dcvsEnable = 0,  // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
              .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
              .setSleepLatency = 1,  // True to consider Latency parameter otherwise False
              .sleepLatency = 40,    // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
              .setSleepDisable = 0,  // True to consider sleep disable/enable parameter otherwise False
              .sleepDisable = 0,     // True to disable sleep, False to re-enable sleep
              .setBusParams = 1,     // True to consider Bus parameter otherwise False
              .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .setCoreParams = 1,  // True to consider Core parameter otherwise False
              .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          },
  };

  powerConfigBalanced_ = {
      .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
      .dcvsV3Config =
          {
              .contextId = powerConfigId_,  // use the power config id created
              .setDcvsEnable = 1,
              .dcvsEnable = 1,  // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
              .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
              .setSleepLatency = 1,  // True to consider Latency parameter otherwise False
              .sleepLatency = 1000,  // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
              .setSleepDisable = 1,  // True to consider sleep disable/enable parameter otherwise False
              .sleepDisable = 0,     // True to disable sleep, False to re-enable sleep
              .setBusParams = 1,     // True to consider Bus parameter otherwise False
              .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
              .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
              .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
              .setCoreParams = 1,  // True to consider Core parameter otherwise False
              .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
              .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
              .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
          },
  };
}

void QNNPerf::shutdown() {
  if (isShutdown_) return;
  isShutdown_ = true;
  CALL_QNN(perfInfra_.destroyPowerConfigId(powerConfigId_));
}

QNNPerf::~QNNPerf() {
  // If shutdown() was already called, skip cleanup
  // This prevents crashes during program exit when QNN HTP infrastructure might be destroyed
  if (!isShutdown_) { shutdown(); }
}

void QNNPerf::setRpcLatencyAndPolling() {
  // set RPC Control Latency
  QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;  // refer QnnHtpPerfInfrastructure.h
  ::memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
  rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  rpcControlLatency.rpcControlLatencyConfig = 100;  // use rpc control latency recommended 100 us, refer hexagon sdk
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs1[] = {&rpcControlLatency, nullptr};

  CALL_QNN(perfInfra_.setPowerConfig(powerConfigId_, powerConfigs1));  // set RPC latency config on power config ID created

  // set RPC Polling
  QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;  // refer QnnHtpPerfInfrastructure.h
  ::memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
  rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  rpcPollingTime.rpcPollingTimeConfig = 9999;  // use rpc polling time recommended 0-10000 us
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs2[] = {&rpcPollingTime, nullptr};

  CALL_QNN(perfInfra_.setPowerConfig(powerConfigId_, powerConfigs2));  // set RPC polling config on power config ID created
}

void QNNPerf::setPowerConfigBurst() {
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&powerConfigBurst_, nullptr};
  CALL_QNN(perfInfra_.setPowerConfig(powerConfigId_, powerConfigs));
}

void QNNPerf::setPowerConfigBalanced() {
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&powerConfigBalanced_, nullptr};
  CALL_QNN(perfInfra_.setPowerConfig(powerConfigId_, powerConfigs));
}

QNNRuntime::~QNNRuntime() {
  // Free Profile
  if (profileHandle != nullptr) { CALL_QNN(qnnInterface.profileFree(profileHandle)); }

  // Free Device
  CALL_QNN(qnnInterface.deviceFree(deviceHandle));

  // Free Backend
  CALL_QNN(qnnInterface.backendFree(backendHandle));

  // Free Log
  CALL_QNN(qnnInterface.logFree(logHandle));
}

QNNRuntime* QNNRuntime::initRuntime(ProfilingLevel profilingLevel, QnnLog_Level_t qnnLogLevel) {
  // Create Interface
  QNN_INTERFACE_VER_TYPE qnnInterface{};
  {
    QnnInterface_t** interfaceProviders = nullptr;
    uint32_t numProviders = 0;
    if (QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
      MLLM_ERROR("Failed to get QNN interface providers.");
      return nullptr;
    }
    if (interfaceProviders == nullptr) {
      MLLM_ERROR("Failed to get interface providers: null interface providers received.");
      return nullptr;
    }
    if (numProviders == 0) {
      MLLM_ERROR("Failed to get interface providers: 0 interface providers.");
      return nullptr;
    }
    bool foundValidInterface = false;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
      if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major
          /*&& QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor*/) {
        foundValidInterface = true;
        qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
        break;
      }
    }
    if (!foundValidInterface) {
      MLLM_ERROR("Failed to find a valid QNN interface provider.");
      return nullptr;
    }
  }

  // Create Log
  Qnn_LogHandle_t logHandle = nullptr;
  {
    QnnLog_Callback_t logCallback = __mllmQnnLoggerCallback;
    if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(logCallback, qnnLogLevel, &logHandle)) != QNN_SUCCESS)
        || (logHandle == nullptr)) {
      MLLM_ERROR("Failed to initialize logging in the backend.");
      return nullptr;
    } else {
      MLLM_INFO("Logging initialized successfully");
    }
  }

  // Create Backend
  Qnn_BackendHandle_t backendHandle = nullptr;
  {
    const QnnBackend_Config_t** backendConfig = nullptr;
    if ((QNN_GET_ERROR_CODE(qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle)) != QNN_SUCCESS)
        || (backendHandle == nullptr)) {
      MLLM_ERROR("Failed to create the backend.");
      return nullptr;
    } else {
      MLLM_INFO("Backend created successfully");
    }
  }

  // Create Device
  Qnn_DeviceHandle_t deviceHandle = nullptr;
  {
    // Check whether the device API is supported.
    if (nullptr != qnnInterface.deviceCreate) {
      auto status = qnnInterface.deviceCreate(logHandle, nullptr, &deviceHandle);
      if (QNN_SUCCESS != status) {
        MLLM_ERROR("Failed to create device, error: {}", (int)status);
        return nullptr;
      }
      MLLM_INFO("Device created successfully");
    }
  }

  // Initialize Profiling
  Qnn_ProfileHandle_t profileHandle = nullptr;
  {
    if (ProfilingLevel::OFF != profilingLevel) {
      MLLM_INFO("Profiling turned on; level = {}", (int)profilingLevel);
      if (ProfilingLevel::BASIC == profilingLevel) {
        MLLM_INFO("Basic profiling requested. Creating Qnn Profile object.");
        if (QNN_PROFILE_NO_ERROR != qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_BASIC, &profileHandle)) {
          MLLM_WARN("Unable to create profile handle in the backend.");
          return nullptr;
        }
      } else if (ProfilingLevel::DETAILED == profilingLevel) {
        MLLM_INFO("Detailed profiling requested. Creating Qnn Profile object.");
        if (QNN_PROFILE_NO_ERROR != qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_DETAILED, &profileHandle)) {
          MLLM_ERROR("Unable to create profile handle in the backend.");
          return nullptr;
        }
      }
    }
  }

  // Register Custom OpPackages
  {
    struct OpPackageInfo {
      std::string path;
      std::string interfaceProvider;
      std::string target;
    };

    const auto resolveOpPackage = [](const char* env_name, const char* fallback) {
      const char* value = std::getenv(env_name);
      return value != nullptr && value[0] != '\0' ? std::string(value) : std::string(fallback);
    };

    std::vector<OpPackageInfo> opPackages = {
        {resolveOpPackage("MLLM_QNN_CPU_OP_PACKAGE", "libQnnLLaMAPackage_CPU.so"), "LLaMAPackageInterfaceProvider", "CPU"},
        {resolveOpPackage("MLLM_QNN_HTP_OP_PACKAGE", "libQnnLLaMAPackage_HTP.so"), "LLaMAPackageInterfaceProvider", "HTP"},
    };

    for (const auto& pkg : opPackages) {
      if (!qnnInterface.backendRegisterOpPackage) {
        MLLM_ERROR("backendRegisterOpPackageFnHandle is nullptr.");
        return nullptr;
      }
      if (QNN_BACKEND_NO_ERROR
          != qnnInterface.backendRegisterOpPackage(backendHandle, pkg.path.c_str(), pkg.interfaceProvider.c_str(),
                                                   pkg.target.c_str())) {
        MLLM_ERROR("Could not register Op Package: {} and interface provider: {}", pkg.path.c_str(),
                   pkg.interfaceProvider.c_str());
        return nullptr;
      }
      MLLM_INFO("Registered Op Package: {} and interface provider: {}", pkg.path.c_str(), pkg.interfaceProvider.c_str());
    }
  }

  // Create QNN System Interface
  QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;
  {
    QnnSystemInterface_t** systemInterfaceProviders{nullptr};
    uint32_t numProviders{0};
    if (QNN_SUCCESS
        != QnnSystemInterface_getProviders((const QnnSystemInterface_t***)&systemInterfaceProviders, &numProviders)) {
      MLLM_ERROR("Failed to get system interface providers.");
      return nullptr;
    } else {
      MLLM_INFO("System interface providers found: {}", numProviders);
    }
    if (0 == numProviders) {
      MLLM_ERROR("Failed to get interface providers: 0 interface providers.");
      return nullptr;
    }
    bool foundValidSystemInterface = false;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
      if (QNN_SYSTEM_API_VERSION_MAJOR == systemInterfaceProviders[pIdx]->systemApiVersion.major
          && QNN_SYSTEM_API_VERSION_MINOR <= systemInterfaceProviders[pIdx]->systemApiVersion.minor) {
        qnnSystemInterface = systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
        foundValidSystemInterface = true;
        break;
      } else {
        // Print system interface provider and self version
        MLLM_WARN("System interface provider: {} version: {}", systemInterfaceProviders[pIdx]->systemApiVersion.major,
                  systemInterfaceProviders[pIdx]->systemApiVersion.minor);
        MLLM_WARN("Self version: {} {}", QNN_SYSTEM_API_VERSION_MAJOR, QNN_SYSTEM_API_VERSION_MINOR);
        MLLM_WARN("Unable to find a valid system interface.");
      }
    }
    if (!foundValidSystemInterface) {
      MLLM_ERROR("Unable to find a valid system interface.");
      return nullptr;
    }
  }

  return new QNNRuntime(qnnInterface, qnnSystemInterface, logHandle, backendHandle, deviceHandle, profileHandle);
}

bool QNNRuntime::createContext(Qnn_ContextHandle_t& context, const QnnContext_Config_t** contextConfig) {
  if (QNN_CONTEXT_NO_ERROR != qnnInterface.contextCreate(backendHandle, deviceHandle, contextConfig, &context)) {
    MLLM_ERROR("Could not create context");
    return false;
  }
  return true;
}

bool QNNRuntime::retrieveContext(const std::string& contextBinaryPath, Qnn_ContextHandle_t& context,
                                 std::vector<std::shared_ptr<QNNModel>>& qnnModels, const QnnContext_Config_t** contextConfig) {
  size_t size = 0;
#if defined(__linux__)
  const int contextFd = open(contextBinaryPath.c_str(), O_RDONLY | O_CLOEXEC);
  if (contextFd < 0) {
    MLLM_ERROR("Could not open context binary file {}: {}", contextBinaryPath, std::strerror(errno));
    return false;
  }
  struct stat contextStat {};
  if (fstat(contextFd, &contextStat) != 0 || contextStat.st_size <= 0
      || static_cast<uint64_t>(contextStat.st_size) > std::numeric_limits<size_t>::max()) {
    MLLM_ERROR("QNN context binary has an invalid size: {}", contextBinaryPath);
    close(contextFd);
    return false;
  }
  size = static_cast<size_t>(contextStat.st_size);
  int mmapFlags = MAP_PRIVATE;
#if defined(MAP_POPULATE)
  mmapFlags |= MAP_POPULATE;
#endif
  void* mappedContext = mmap(nullptr, size, PROT_READ, mmapFlags, contextFd, 0);
  const int mmapError = errno;
  close(contextFd);
  if (mappedContext == MAP_FAILED) {
    MLLM_ERROR("Failed to mmap QNN context binary {}: {}", contextBinaryPath, std::strerror(mmapError));
    return false;
  }
  contextBinaryBuffer_ = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(mappedContext), [size](uint8_t* mapping) {
    if (mapping != nullptr) { munmap(mapping, size); }
  });
  MLLM_INFO("Memory-mapped QNN context binary: {} bytes", size);
#else
  std::ifstream file(contextBinaryPath, std::ios::binary | std::ios::ate);
  if (!file.is_open() || !file.good()) {
    MLLM_ERROR("Could not open context binary file: {}", contextBinaryPath);
    return false;
  }
  const std::streamsize streamSize = file.tellg();
  if (streamSize <= 0 || static_cast<uint64_t>(streamSize) > std::numeric_limits<size_t>::max()) {
    MLLM_ERROR("QNN context binary has an invalid size: {}", contextBinaryPath);
    return false;
  }
  size = static_cast<size_t>(streamSize);
  file.seekg(0, std::ios::beg);
  contextBinaryBuffer_ = std::shared_ptr<uint8_t>(new uint8_t[size], std::default_delete<uint8_t[]>());
  file.read(reinterpret_cast<char*>(contextBinaryBuffer_.get()), streamSize);
  if (!file.good() || static_cast<size_t>(file.gcount()) != size) {
    MLLM_ERROR("Failed to read QNN context binary: {}", contextBinaryPath);
    contextBinaryBuffer_.reset();
    return false;
  }
  file.close();
  MLLM_INFO("Read QNN context binary: {} bytes", size);
#endif

  // inspect binary info
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (!qnnSystemInterface.systemContextCreate) {
    MLLM_ERROR("systemContextCreate is nullptr.");
    return false;
  }
  if (QNN_SUCCESS != qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
    MLLM_ERROR("Could not create system handle.");
    return false;
  } else {
    MLLM_INFO("System context created successfully");
  }

  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  Qnn_ContextBinarySize_t binaryInfoSize{0};

  if (QNN_SUCCESS
      != qnnSystemInterface.systemContextGetBinaryInfo(sysCtxHandle, static_cast<void*>(contextBinaryBuffer_.get()), size,
                                                       &binaryInfo, &binaryInfoSize)) {
    MLLM_ERROR("Failed to get context binary info");
    return false;
  } else {
    MLLM_INFO("Context binary info retrieved successfully");
  }

  // Extract graph metadata to create QNNModels instead of GraphInfo_t
  GraphInfo_t** tmpGraphsInfo = nullptr;
  uint32_t graphNum;
  // fill GraphInfo_t based on binary info - temporarily needed for tensor extraction
  if (!copyMetadataToGraphsInfo(binaryInfo, tmpGraphsInfo, graphNum)) {
    MLLM_ERROR("Failed to copy metadata.");
    return false;
  }
  if (QNN_SUCCESS != qnnSystemInterface.systemContextFree(sysCtxHandle)) {
    MLLM_ERROR("Could not free system context.");
    return false;
  } else {
    MLLM_INFO("System context freed successfully");
  }
  sysCtxHandle = nullptr;

  QnnContext_Config_t memoryLimitConfig = QNN_CONTEXT_CONFIG_INIT;
  QnnContext_Config_t persistentBinaryConfig = QNN_CONTEXT_CONFIG_INIT;
  std::vector<const QnnContext_Config_t*> effectiveConfigs;
  if (contextConfig != nullptr) {
    for (size_t index = 0; contextConfig[index] != nullptr; ++index) {
      effectiveConfigs.push_back(contextConfig[index]);
    }
  }

  const char* memoryLimitEnv = std::getenv("MLLM_QNN_CONTEXT_MEMORY_LIMIT_MB");
  bool memoryLimitEnabled = false;
  if (memoryLimitEnv != nullptr && std::strcmp(memoryLimitEnv, "0") != 0) {
    char* end = nullptr;
    const unsigned long long memoryLimit = std::strtoull(memoryLimitEnv, &end, 10);
    if (end == memoryLimitEnv || *end != '\0' || memoryLimit == 0) {
      MLLM_ERROR("Invalid MLLM_QNN_CONTEXT_MEMORY_LIMIT_MB: {}", memoryLimitEnv);
      return false;
    }
    memoryLimitConfig.option = QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT;
    memoryLimitConfig.memoryLimitHint = memoryLimit;
    effectiveConfigs.push_back(&memoryLimitConfig);
    memoryLimitEnabled = true;
    MLLM_INFO("QNN context memory limit hint enabled: {} MB", memoryLimit);
  }

  bool persistentBinaryEnabled = memoryLimitEnabled;
  const char* persistentBinaryEnv = std::getenv("MLLM_QNN_CONTEXT_PERSISTENT_BINARY");
  if (persistentBinaryEnv != nullptr && persistentBinaryEnv[0] != '\0') {
    if (std::strcmp(persistentBinaryEnv, "1") == 0) {
      persistentBinaryEnabled = true;
    } else if (std::strcmp(persistentBinaryEnv, "0") == 0) {
      persistentBinaryEnabled = false;
    } else {
      MLLM_ERROR("Invalid MLLM_QNN_CONTEXT_PERSISTENT_BINARY: {}", persistentBinaryEnv);
      return false;
    }
  }
  if (memoryLimitEnabled && !persistentBinaryEnabled) {
    MLLM_ERROR("QNN context memory limit requires a persistent binary");
    return false;
  }
  if (persistentBinaryEnabled) {
    persistentBinaryConfig.option = QNN_CONTEXT_CONFIG_PERSISTENT_BINARY;
    persistentBinaryConfig.isPersistentBinary = 1;
    effectiveConfigs.push_back(&persistentBinaryConfig);
    if (!memoryLimitEnabled) { MLLM_INFO("QNN persistent context binary enabled without a memory limit hint"); }
  }
  effectiveConfigs.push_back(nullptr);

  const auto createStatus = qnnInterface.contextCreateFromBinary(
      backendHandle, deviceHandle, effectiveConfigs.data(), contextBinaryBuffer_.get(), size, &context, profileHandle);
  if (createStatus != QNN_CONTEXT_NO_ERROR) {
    MLLM_ERROR("QNN contextCreateFromBinary failed: {}", createStatus);
    return false;
  } else {
    MLLM_INFO("Context created from binary successfully");
  }

  // Create QNNModels for each graph and initialize from context
  qnnModels.clear();
  qnnModels.reserve(graphNum);

  for (uint32_t i = 0; i < graphNum; ++i) {
    GraphInfo_t* graphInfo = tmpGraphsInfo[i];

    // Retrieve the graph handle
    Qnn_GraphHandle_t graph = nullptr;
    if (QNN_SUCCESS != qnnInterface.graphRetrieve(context, graphInfo->graphName, &graph)) {
      MLLM_ERROR("Unable to retrieve graph handle for graph: {}", graphInfo->graphName);
      return false;
    }

    // Create QNNModel and initialize from context
    auto qnnModel = std::make_shared<QNNModel>(qnnInterface, backendHandle);
    ModelError_t err =
        qnnModel->initializeFromContext(context, graphInfo->graphName, graph, graphInfo->inputTensors,
                                        graphInfo->numInputTensors, graphInfo->outputTensors, graphInfo->numOutputTensors);

    if (err != MODEL_NO_ERROR) {
      MLLM_ERROR("Failed to initialize QNNModel from context for graph: {} with error: {}", graphInfo->graphName,
                 static_cast<int>(err));
      return false;
    }

    qnnModels.push_back(qnnModel);
    MLLM_INFO("Successfully created QNNModel for graph: {}", graphInfo->graphName);
  }

  if (!persistentBinaryEnabled) {
    contextBinaryBuffer_.reset();
    MLLM_INFO("Released non-persistent QNN context binary buffer");
  } else {
#if defined(__linux__)
    const char* madviseEnv = std::getenv("MLLM_QNN_CONTEXT_MADVISE_DONTNEED");
    const bool releasePages = madviseEnv != nullptr && std::strcmp(madviseEnv, "0") != 0;
    if (releasePages) {
      if (madvise(contextBinaryBuffer_.get(), size, MADV_DONTNEED) == 0) {
        MLLM_INFO("Released resident QNN context pages; persistent mmap retained");
      } else {
        MLLM_ERROR("madvise(MADV_DONTNEED) failed for QNN context: {}", std::strerror(errno));
      }
    } else {
      MLLM_INFO("Persistent QNN context uses reclaimable file-backed mmap");
    }
#endif
  }

  // Clean up temporary GraphInfo_t structures
  for (uint32_t i = 0; i < graphNum; ++i) {
    if (tmpGraphsInfo[i]) {
      if (tmpGraphsInfo[i]->graphName) { free(tmpGraphsInfo[i]->graphName); }
      freeQnnTensors(tmpGraphsInfo[i]->inputTensors, tmpGraphsInfo[i]->numInputTensors);
      freeQnnTensors(tmpGraphsInfo[i]->outputTensors, tmpGraphsInfo[i]->numOutputTensors);
    }
  }
  if (graphNum > 0 && tmpGraphsInfo[0]) { free(tmpGraphsInfo[0]); }
  if (tmpGraphsInfo) { free(tmpGraphsInfo); }

  MLLM_INFO("QNN context retrieved from qnn_context.bin with {} QNNModels(QnnGraphs)", graphNum);
  return true;
}

bool QNNBackend::createContext(bool prepareOnly) {
  if (prepareOnly) {
    QnnHtpContext_CustomConfig_t htpConfig = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
    htpConfig.option = QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY;
    htpConfig.isPrepareOnly = true;

    QnnContext_Config_t contextConfig = QNN_CONTEXT_CONFIG_INIT;
    contextConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    contextConfig.customConfig = &htpConfig;
    const QnnContext_Config_t* contextConfigs[] = {&contextConfig, nullptr};
    if (!runtime_->createContext(context_, contextConfigs)) { return false; }
    MLLM_INFO("QNN context prepare-only mode enabled");
  } else if (!runtime_->createContext(context_, nullptr)) {
    return false;
  }
  // init QNN Allocator
  static_pointer_cast<QNNAllocator>(allocator_)->setQNNPointer(runtime_->qnnInterface, context_);
  return true;
}

bool QNNBackend::loadContext(const std::string& contextPath) {
  if (!runtime_->retrieveContext(contextPath, context_, qnnModels_, nullptr)) { return false; }
  // fill qnnModelIndexMap_ info according to qnnModels_
  for (size_t i = 0; i < qnnModels_.size(); i++) {
    auto graphName = qnnModels_[i]->getQnnGraphName();
    qnnModelIndexMap_.insert(std::make_pair(graphName, i));
  }
  // init QNN Allocator
  static_pointer_cast<QNNAllocator>(allocator_)->setQNNPointer(runtime_->qnnInterface, context_);
  return true;
}

bool QNNBackend::saveContext(const std::string& contextPath) {
  uint64_t binarySize = 0;
  uint64_t writtenSize = 0;

  const auto sizeStatus = runtime_->qnnInterface.contextGetBinarySize(context_, &binarySize);
  if (sizeStatus != QNN_CONTEXT_NO_ERROR || binarySize == 0) {
    MLLM_ERROR("Failed to query QNN context binary size: status {}, size {}", sizeStatus, binarySize);
    return false;
  }

  std::unique_ptr<uint8_t[]> binaryBuffer(new uint8_t[binarySize]);

  const auto binaryStatus =
      runtime_->qnnInterface.contextGetBinary(context_, reinterpret_cast<void*>(binaryBuffer.get()), binarySize, &writtenSize);
  if (binaryStatus != QNN_CONTEXT_NO_ERROR || writtenSize == 0 || writtenSize > binarySize) {
    MLLM_ERROR("Failed to serialize QNN context: status {}, capacity {}, written {}", binaryStatus, binarySize, writtenSize);
    return false;
  }
  std::ofstream file(contextPath, std::ios::binary);
  if (!file.is_open()) {
    MLLM_ERROR("Failed to open QNN context output: {}", contextPath);
    return false;
  }
  file.write(reinterpret_cast<char*>(binaryBuffer.get()), writtenSize);
  if (!file.good()) {
    MLLM_ERROR("Failed to write QNN context output: {}", contextPath);
    return false;
  }
  file.close();

  MLLM_INFO("QNN context saved to {} written {} bytes.", contextPath, writtenSize);
  return true;
}

std::shared_ptr<QNNModel> QNNBackend::createQnnGraph(const std::string& graphName) {
  // If the graph already exists, return the existing model
  if (qnnModelIndexMap_.find(graphName) != qnnModelIndexMap_.end()) {
    currentQnnModelIndex_ = qnnModelIndexMap_[graphName];
    return qnnModels_[currentQnnModelIndex_];
  }

  // Create a new QNNModel
  currentQnnModelIndex_ = static_cast<int>(qnnModels_.size());
  qnnModelIndexMap_.insert(std::make_pair(graphName, currentQnnModelIndex_));

  auto qnnModel = std::make_shared<QNNModel>(runtime_->qnnInterface, runtime_->backendHandle);
  qnnModels_.push_back(qnnModel);

  // Initialize QNN graph info with basic configs
  const QnnGraph_Config_t* graphConfigList[] = {nullptr};

  ModelError_t err = MODEL_NO_ERROR;
  if ((err = qnnModel->initialize(context_, graphName.c_str(), debug_, 1, graphConfigList)) != MODEL_NO_ERROR) {
    MLLM_ERROR("QNN graph initialization failed for graph: {} with error code: {}", graphName, static_cast<int>(err));
    qnnModels_.pop_back();
    qnnModelIndexMap_.erase(graphName);
    return nullptr;
  }

  return qnnModel;
}

void QNNBackend::graphAddNode(const std::string& graphName, const std::string& nodeName, const std::string& nodeType,
                              const std::vector<std::string>& inputTensorNames,
                              const std::vector<std::string>& outputTensorNames,
                              const std::vector<std::shared_ptr<QNNParamTensorWrapper>>& tensorParams,
                              const std::vector<std::shared_ptr<QNNParamScalarWrapper>>& scalarParams,
                              const std::string& packageName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding node", graphName);
    return;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) { return; }

  // Add node to the model
  ModelError_t err = qnnModel->addNode(QNN_OPCONFIG_VERSION_1, nodeName, packageName, nodeType, tensorParams, scalarParams,
                                       inputTensorNames, outputTensorNames);

  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add node {} of type {} to graph {}: error code {}\n", nodeName, nodeType, graphName,
               static_cast<int>(err));
  }
}

bool QNNBackend::graphFinalize(const std::string& graphName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for finalization", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_INFO("Graph {} is loaded from cache, skipping finalization", graphName);
    return true;
  }

  // Graph finalize
  if (MODEL_NO_ERROR != qnnModel->finalizeGraph(runtime_->profileHandle, nullptr)) {
    MLLM_ERROR("Failed to finalize graph: {}", graphName);
    return false;
  }

  qnnModel->freeCachedTensors();

  // Extract profiling info if enabled
  if (ProfilingLevel::OFF != profilingLevel_) { extractBackendProfilingInfo(runtime_->profileHandle); }

  return true;
}

void QNNBackend::setSequenceTile(int32_t sequence_tile) {
  if (sequence_tile < 0) { throw std::invalid_argument("QNN sequence tile cannot be negative"); }
  if (!sequenceTileGraphs_.empty() && sequence_tile != sequenceTile_) {
    throw std::logic_error("QNN sequence tile cannot change after graph execution starts");
  }
  sequenceTile_ = sequence_tile;
}

void QNNBackend::graphExecute(const std::string& graphName, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for execution", graphName);
    return;
  }
  auto model = qnnModels_[it->second];

  // Validate input size matches expected input count
  if (inputs.size() != model->getGraphInputTensorWrappers().size()) {
    MLLM_ERROR("Input size mismatch: expected {}, got {} for graph '{}'", model->getGraphInputTensorWrappers().size(),
               inputs.size(), graphName);
    return;
  }

  if (outputs.size() != model->getGraphOutputTensorWrappers().size()) {
    MLLM_ERROR("Output size mismatch: expected {}, got {} for graph '{}'", model->getGraphOutputTensorWrappers().size(),
               outputs.size(), graphName);
    return;
  }

  if (sequenceTile_ > 0) {
    // A shared tile remains exclusive through both input staging and output copy.
    static std::mutex sequence_tile_pool_mutex;
    const bool use_buffer_pool = sequenceTileBufferPoolEnabled();
    std::unique_lock pool_lock(sequence_tile_pool_mutex, std::defer_lock);
    if (use_buffer_pool) { pool_lock.lock(); }
    const auto input_wrappers = model->getGraphInputTensorWrappers();
    const auto output_wrappers = model->getGraphOutputTensorWrappers();
    const auto allocate_scratch = [&](SequenceTileGraphState& state, bool batched) {
      const auto allocate = [&](const auto& logical, const auto& wrappers, const auto& plans,
                                std::vector<Tensor>& scratch, bool is_input) {
        const auto find_peer = [&](size_t index, const auto& matches) -> Tensor {
          if (use_buffer_pool) {
            for (const auto& [_, peer] : sequenceTileGraphs_) {
              if (&peer == &state) { continue; }
              const auto& tiles = is_input ? peer.tile_inputs : peer.tile_outputs;
              if (index < tiles.size() && matches(tiles[index])) { return tiles[index]; }
            }
          }
          return {};
        };
        for (size_t index = 0; index < scratch.size(); ++index) {
          const auto* data = scratch[index].ptr<void>();
          if (find_peer(index, [&](const Tensor& tile) { return tile.ptr<void>() == data; }).isNil()) {
            sequenceTileProfile_.scratch_bytes -= scratch[index].bytes();
          }
        }
        scratch.clear();
        scratch.reserve(plans.size());
        for (size_t index = 0; index < plans.size(); ++index) {
          auto shape = plans[index].compiled_shape;
          if (batched) { shape[0] = kSequenceBatchSize; }
          const auto dtype = logical[index].dtype();
          auto tile = find_peer(index, [&](const Tensor& other) { return other.dtype() == dtype && other.shape() == shape; });
          if (tile.isNil()) {
            tile = Tensor::empty(shape, dtype, kQNN).alloc();
            tile.setName("qnn_sequence_tile_" + graphName + (is_input ? "_input" : "_output") + std::to_string(index));
            sequenceTileProfile_.scratch_bytes += tile.bytes();
          }
          if (batched) {
            if (tile.bytes() != logical[index].bytes()) {
              throw std::invalid_argument("QNN batched tile storage differs from the logical tensor size");
            }
          } else {
            wrappers[index]->__rebindDataContainerForExecution(tile);
            wrappers[index]->alloc();
          }
          scratch.push_back(std::move(tile));
        }
      };
      allocate(inputs, input_wrappers, state.input_plans, state.tile_inputs, true);
      allocate(outputs, output_wrappers, state.output_plans, state.tile_outputs, false);
    };

    auto state_it = sequenceTileGraphs_.find(graphName);
    if (state_it == sequenceTileGraphs_.end()) {
      SequenceTileGraphState state;
      auto prepare_boundary = [&](Tensor& logical, const std::shared_ptr<QNNTensorWrapper>& wrapper,
                                  std::vector<QNNSequenceTilePlan>& plans) {
        if (logical.isNil() || logical.device() != kQNN || !logical.isContiguous()
            || logical.impl()->storage()->ptr_ == nullptr) {
          throw std::invalid_argument("QNN sequence tiling requires allocated contiguous QNN graph boundaries");
        }
        if (QNN_TENSOR_GET_DATA_TYPE(wrapper->getNativeTensor()) != mllmDataTypeToQnnDataType(logical.dtype())) {
          throw std::invalid_argument("QNN sequence-tile boundary dtype differs from the compiled graph");
        }
        Tensor::shape_t compiled_shape;
        compiled_shape.reserve(wrapper->getDimension()->size());
        for (const uint32_t dimension : *wrapper->getDimension()) { compiled_shape.push_back(static_cast<int32_t>(dimension)); }
        auto plan = planQnnSequenceTiling(logical.shape(), compiled_shape, sequenceTile_);
        if (state.tile_count == 0) {
          state.tile_count = plan.tile_count;
        } else if (state.tile_count != plan.tile_count) {
          throw std::invalid_argument("QNN graph boundaries have inconsistent sequence-tile counts");
        }

        plans.push_back(std::move(plan));
      };

      for (size_t index = 0; index < inputs.size(); ++index) {
        prepare_boundary(inputs[index], input_wrappers[index], state.input_plans);
      }
      for (size_t index = 0; index < outputs.size(); ++index) {
        if (outputs[index].isNil()) {
          throw std::invalid_argument("QNN sequence tiling received a nil logical output boundary");
        }
        if (outputs[index].impl()->storage()->ptr_ == nullptr) { outputs[index].alloc(); }
        prepare_boundary(outputs[index], output_wrappers[index], state.output_plans);
      }
      const bool batch_requested = sequenceBatchRequested(graphName);
      const auto capability = runtime_->qnnInterface.propertyHasCapability;
      const bool batched = batch_requested && !state.input_plans.empty() && !state.output_plans.empty()
                           && capability
                           && capability(QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE) == QNN_PROPERTY_SUPPORTED
                           && std::all_of(state.input_plans.begin(), state.input_plans.end(), supportsSequenceBatch)
                           && std::all_of(state.output_plans.begin(), state.output_plans.end(), supportsSequenceBatch);
      allocate_scratch(state, batched);
      if (batch_requested) {
        MLLM_INFO("QNN_SHADOW_BATCH2_SETUP graph={} enabled={}", graphName, static_cast<int>(batched));
      }
      MLLM_INFO("QNN_SEQUENCE_TILE_SETUP graph={} tile_sequence={} tiles={} scratch_bytes={}", graphName, sequenceTile_,
                state.tile_count, sequenceTileProfile_.scratch_bytes);
      state_it = sequenceTileGraphs_.emplace(graphName, std::move(state)).first;
    }

    auto& state = state_it->second;
    auto validate_boundary = [&](const Tensor& logical, const QNNSequenceTilePlan& expected) {
      if (logical.isNil() || logical.device() != kQNN || !logical.isContiguous()
          || logical.impl()->storage()->ptr_ == nullptr) {
        throw std::invalid_argument("QNN sequence-tile graph boundary is not allocated contiguous QNN memory");
      }
      const auto actual = planQnnSequenceTiling(logical.shape(), expected.compiled_shape, sequenceTile_);
      if (actual.axis != expected.axis || actual.tile_count != expected.tile_count) {
        throw std::invalid_argument("QNN sequence-tile graph boundary changed after setup");
      }
    };
    for (size_t index = 0; index < inputs.size(); ++index) { validate_boundary(inputs[index], state.input_plans[index]); }
    for (size_t index = 0; index < outputs.size(); ++index) {
      if (outputs[index].impl()->storage()->ptr_ == nullptr) { outputs[index].alloc(); }
      validate_boundary(outputs[index], state.output_plans[index]);
    }

    std::vector<Qnn_Tensor_t> qnn_inputs;
    std::vector<Qnn_Tensor_t> qnn_outputs;
    qnn_inputs.reserve(input_wrappers.size());
    qnn_outputs.reserve(output_wrappers.size());
    using clock = std::chrono::steady_clock;
    const auto elapsed_us = [](clock::time_point start, clock::time_point end) {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    };
    ++sequenceTileProfile_.graph_calls;
    if (!state.tile_inputs.empty() && state.tile_inputs.front().shape()[0] == kSequenceBatchSize
        && state.input_plans.front().compiled_shape[0] == 1) {
      const auto shared_allocator = std::static_pointer_cast<QNNAllocator>(allocator());
      std::vector<std::vector<uint32_t>> dimensions;
      dimensions.reserve(inputs.size() + outputs.size());
      const auto bind = [&](const auto& tiles, const auto& wrappers, std::vector<Qnn_Tensor_t>& descriptors) {
        for (size_t index = 0; index < tiles.size(); ++index) {
          auto& shape = dimensions.emplace_back(*wrappers[index]->getDimension());
          shape[0] = kSequenceBatchSize;
          auto& descriptor = descriptors.emplace_back(*wrappers[index]->getNativeTensor());
          QNN_TENSOR_SET_DIMENSIONS(descriptor, shape.data());
          QNN_TENSOR_SET_MEM_HANDLE(descriptor, nullptr);
          shared_allocator->registerQnnTensorToSharedBuffer(tiles[index].template ptr<void>(), descriptor);
        }
      };
      bind(state.tile_inputs, input_wrappers, qnn_inputs);
      bind(state.tile_outputs, output_wrappers, qnn_outputs);

      auto stage_start = clock::now();
      for (size_t index = 0; index < inputs.size(); ++index) {
        copyQnnSequenceBatch<true>(state.tile_inputs[index], inputs[index], state.input_plans[index]);
      }
      sequenceTileProfile_.input_copy_us += elapsed_us(stage_start, clock::now());
      stage_start = clock::now();
      const auto status = runtime_->qnnInterface.graphExecute(model->getQnnGraph(), qnn_inputs.data(), qnn_inputs.size(),
                                                               qnn_outputs.data(), qnn_outputs.size(),
                                                               runtime_->profileHandle, nullptr);
      const auto execute_us = elapsed_us(stage_start, clock::now());
      sequenceTileProfile_.execute_us += execute_us;
      ++sequenceTileProfile_.tile_calls;
      if (status == QNN_SUCCESS) {
        stage_start = clock::now();
        for (size_t index = 0; index < outputs.size(); ++index) {
          copyQnnSequenceBatch<false>(outputs[index], state.tile_outputs[index], state.output_plans[index]);
        }
        sequenceTileProfile_.output_copy_us += elapsed_us(stage_start, clock::now());
        static const char* trace = std::getenv("MLLM_QNN_SEQUENCE_BATCH2_TRACE");
        if (trace && std::string_view(trace) == "1") {
          MLLM_INFO("QNN_SHADOW_BATCH2_EXEC graph={} tiles={} native_calls=1 exec_us={}", graphName, state.tile_count,
                    execute_us);
        }
        if (ProfilingLevel::OFF != profilingLevel_) { extractBackendProfilingInfo(runtime_->profileHandle); }
        return;
      }
      MLLM_WARN("QNN_SHADOW_BATCH2_FALLBACK graph={} status={}", graphName, status);
      qnn_inputs.clear();
      qnn_outputs.clear();
      allocate_scratch(state, false);
    }

    for (const auto& wrapper : input_wrappers) { qnn_inputs.push_back(*wrapper->getNativeTensor()); }
    for (const auto& wrapper : output_wrappers) { qnn_outputs.push_back(*wrapper->getNativeTensor()); }
    for (int32_t tile_index = 0; tile_index < state.tile_count; ++tile_index) {
      auto stage_start = clock::now();
      for (size_t index = 0; index < inputs.size(); ++index) {
        const auto& plan = state.input_plans[index];
        copyQnnSequenceTile(state.tile_inputs[index], 0, inputs[index], tile_index * sequenceTile_, sequenceTile_, plan.axis);
      }
      sequenceTileProfile_.input_copy_us += elapsed_us(stage_start, clock::now());

      stage_start = clock::now();
      CALL_QNN(runtime_->qnnInterface.graphExecute(model->getQnnGraph(), qnn_inputs.data(), qnn_inputs.size(),
                                                   qnn_outputs.data(), qnn_outputs.size(), runtime_->profileHandle, nullptr));
      sequenceTileProfile_.execute_us += elapsed_us(stage_start, clock::now());

      stage_start = clock::now();
      for (size_t index = 0; index < outputs.size(); ++index) {
        const auto& plan = state.output_plans[index];
        copyQnnSequenceTile(outputs[index], tile_index * sequenceTile_, state.tile_outputs[index], 0, sequenceTile_, plan.axis);
      }
      sequenceTileProfile_.output_copy_us += elapsed_us(stage_start, clock::now());
      ++sequenceTileProfile_.tile_calls;
    }
    if (ProfilingLevel::OFF != profilingLevel_) { extractBackendProfilingInfo(runtime_->profileHandle); }
    return;
  }

  std::vector<Qnn_Tensor_t> qnn_inputs;
  std::vector<Qnn_Tensor_t> qnn_outputs;
  // Prepare QNN inputs
  for (int i = 0; i < model->getGraphInputTensorWrappers().size(); i++) {
    auto wrapper = model->getGraphInputTensorWrappers()[i];
    const auto& runtime_input = inputs[i];

    // Validate input tensors
    if (runtime_input.isNil()) {
      MLLM_ERROR("Input tensor {} is nil for graph '{}'", i, graphName);
      return;
    }

    // Module calls can produce fresh Tensor objects. Bind this graph boundary
    // to the current call rather than retaining the tensor seen while tracing.
    wrapper->__rebindDataContainerForExecution(runtime_input);

    // Allocate and register the wrapper tensor with QNN allocator
    // QNNAllocator will handle registered memory descriptor when needed
    wrapper->alloc();
    qnn_inputs.push_back(*(wrapper->getNativeTensor()));
  }
  // Prepare QNN outputs
  for (int j = 0; j < model->getGraphOutputTensorWrappers().size(); j++) {
    auto wrapper = model->getGraphOutputTensorWrappers()[j];
    const auto& runtime_output = outputs[j];

    // Validate output tensors
    if (runtime_output.isNil()) {
      MLLM_ERROR("Output tensor {} is nil for graph '{}'", j, graphName);
      return;
    }

    wrapper->__rebindDataContainerForExecution(runtime_output);

    // alloc and register qnn tensor
    wrapper->alloc();  // QNNAllocator will handle registered memory descriptor
    qnn_outputs.push_back(*(wrapper->getNativeTensor()));
  }

  CALL_QNN(runtime_->qnnInterface.graphExecute(model->getQnnGraph(), qnn_inputs.data(), qnn_inputs.size(), qnn_outputs.data(),
                                               qnn_outputs.size(), runtime_->profileHandle, nullptr));

  if (ProfilingLevel::OFF != profilingLevel_) { extractBackendProfilingInfo(runtime_->profileHandle); }
}

bool QNNBackend::addTensor(const std::string& graphName, const std::string& tensorName, Qnn_TensorType_t type,
                           const Tensor& tensor, Qnn_QuantizeParams_t quantize) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding tensor", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_ERROR("Cannot add tensor {} to finalized graph {}", tensorName, graphName);
    return false;
  }

  ModelError_t err = qnnModel->addTensor(tensorName, type, tensor, quantize);
  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add tensor {} to graph {}: error code {}", tensorName, graphName, static_cast<int>(err));
    return false;
  }

  return true;
}

bool QNNBackend::addStaticTensor(const std::string& graphName, const std::string& tensorName, const Tensor& tensor,
                                 Qnn_QuantizeParams_t quantize, Qnn_DataType_t dataTypeOverride) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding static tensor", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_ERROR("Cannot add static tensor {} to finalized graph {}", tensorName, graphName);
    return false;
  }

  ModelError_t err = qnnModel->addStaticTensor(tensorName, tensor, quantize, dataTypeOverride);
  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add static tensor {} to graph {}: error code {}", tensorName, graphName, static_cast<int>(err));
    return false;
  }

  return true;
}

std::shared_ptr<QNNTensorWrapper> QNNBackend::getTensorWrapper(const std::string& graphName, const std::string& tensorName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for getting tensor wrapper", graphName);
    return nullptr;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  return qnnModel->getTensorWrapper(tensorName);
}

void QNNBackend::extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle) {
  // Extract profiling information from QNN backend
  // This is a placeholder implementation
  if (profileHandle == nullptr) { return; }

  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR != runtime_->qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
    MLLM_WARN("Failed to get profile events");
    return;
  }

  MLLM_INFO("Extracted {} profiling events", numEvents);
}

}  // namespace mllm::qnn
