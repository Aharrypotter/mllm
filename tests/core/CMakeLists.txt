add_executable(Mllm-Test-Core-LoadParams LoadParamsTest.cpp)
target_link_libraries(Mllm-Test-Core-LoadParams PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Core-LoadParams PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Core-ARGeneration ARGenerationTest.cpp)
target_link_libraries(Mllm-Test-Core-ARGeneration PRIVATE gtest MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Core-ARGeneration PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
