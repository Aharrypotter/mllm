add_executable(Mllm-Test-Nn-FooNetTest FooNetTest.cpp)
target_link_libraries(Mllm-Test-Nn-FooNetTest PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-FooNetTest PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-StaticCacheTest StaticCacheTest.cpp)
target_link_libraries(Mllm-Test-Nn-StaticCacheTest PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-StaticCacheTest PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-KVHeadStaticCache KVHeadStaticCacheTest.cpp)
target_link_libraries(Mllm-Test-Nn-KVHeadStaticCache PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-KVHeadStaticCache PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-GroupedQueryAttention GroupedQueryAttentionTest.cpp)
target_link_libraries(Mllm-Test-Nn-GroupedQueryAttention PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-GroupedQueryAttention PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
