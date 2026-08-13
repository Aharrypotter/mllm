add_executable(mllm-minicpm5-runner main.cpp)
target_link_libraries(mllm-minicpm5-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-minicpm5-runner PRIVATE ${MLLM_INCLUDE_DIR})
