if(MLLM_BUILD_ARM_BACKEND)
  add_executable(Mllm-Benchmark-ARM-HPC-Sgemm arm_mllm_blas_sgemm.cpp)
  target_link_libraries(Mllm-Benchmark-ARM-HPC-Sgemm PRIVATE benchmark::benchmark MllmRT MllmCPUBackend)

  add_executable(Mllm-Benchmark-Lfm2-Parallel-Linear lfm2_parallel_linear.cpp)
  target_link_libraries(Mllm-Benchmark-Lfm2-Parallel-Linear PRIVATE MllmRT MllmCPUBackend)

  add_executable(Mllm-Benchmark-Lfm2-Parallel-Linear-Shared-Mx lfm2_parallel_linear_shared_mx.cpp)
  target_link_libraries(Mllm-Benchmark-Lfm2-Parallel-Linear-Shared-Mx PRIVATE MllmRT MllmCPUBackend)
endif()
