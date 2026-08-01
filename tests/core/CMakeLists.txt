add_executable(Mllm-Test-Core-LoadParams LoadParamsTest.cpp)
target_link_libraries(Mllm-Test-Core-LoadParams PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Core-LoadParams PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Core-ARGeneration ARGenerationTest.cpp)
target_link_libraries(Mllm-Test-Core-ARGeneration PRIVATE gtest MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Core-ARGeneration PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Core-Qwen35BenchmarkHarness Qwen35BenchmarkHarnessTest.cpp)
target_link_libraries(Mllm-Test-Core-Qwen35BenchmarkHarness PRIVATE gtest MllmRT)
target_include_directories(Mllm-Test-Core-Qwen35BenchmarkHarness PRIVATE ${MLLM_INCLUDE_DIR}
                                                                           ${PROJECT_SOURCE_DIR}/examples/qwen3_5)

include(GoogleTest)
