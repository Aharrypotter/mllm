add_executable(mllm-lfm2-runner main.cpp)
target_link_libraries(mllm-lfm2-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-lfm2-runner PRIVATE ${MLLM_INCLUDE_DIR})
