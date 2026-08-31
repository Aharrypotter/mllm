add_executable(Mllm-Test-Engine-PerfMem PerfMemTest.cpp)
target_link_libraries(Mllm-Test-Engine-PerfMem PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Engine-PerfMem PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Engine-AsyncModule AsyncModuleTest.cpp)
target_link_libraries(Mllm-Test-Engine-AsyncModule PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Engine-AsyncModule PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Engine-RadixTree RadixTreeTest.cpp)
target_link_libraries(Mllm-Test-Engine-RadixTree PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Engine-RadixTree PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Engine-PrefixCache PrefixCacheTest.cpp)
target_link_libraries(Mllm-Test-Engine-PrefixCache PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Engine-PrefixCache PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Engine-Service ServiceTest.cpp)
target_link_libraries(Mllm-Test-Engine-Service PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Engine-Service PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Engine-RandomStates RandomStatesTest.cpp)
target_link_libraries(Mllm-Test-Engine-RandomStates PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Engine-RandomStates PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Engine-HpcThreadPool HpcThreadPoolTest.cpp)
target_link_libraries(Mllm-Test-Engine-HpcThreadPool PRIVATE gtest_main MllmRT)
target_include_directories(Mllm-Test-Engine-HpcThreadPool PRIVATE ${MLLM_INCLUDE_DIR})
add_test(NAME HpcThreadPoolNonDivisibleStep COMMAND Mllm-Test-Engine-HpcThreadPool)
