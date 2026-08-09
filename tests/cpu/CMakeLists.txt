add_executable(Mllm-Test-CPUKernel KernelTest.cpp)
target_link_libraries(Mllm-Test-CPUKernel PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-CPUKernel PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Qwen35-GDN Qwen35GDNTest.cpp)
target_link_libraries(Mllm-Test-Qwen35-GDN PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Qwen35-GDN PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Qwen35-GDN-Conv Qwen35GDNConvTest.cpp)
target_link_libraries(Mllm-Test-Qwen35-GDN-Conv PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Qwen35-GDN-Conv PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-KaiW4A32Pack KaiW4A32PackTest.cpp)
target_link_libraries(Mllm-Test-KaiW4A32Pack PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-KaiW4A32Pack PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Qwen35-Tokenizer Qwen35TokenizerTest.cpp)
target_link_libraries(Mllm-Test-Qwen35-Tokenizer PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Qwen35-Tokenizer PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Qwen35-Config Qwen35ConfigTest.cpp)
target_link_libraries(Mllm-Test-Qwen35-Config PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Qwen35-Config PRIVATE ${MLLM_INCLUDE_DIR})
target_compile_definitions(Mllm-Test-Qwen35-Config
                           PRIVATE QWEN35_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../examples/qwen3_5")

add_executable(Mllm-Test-MiniCPM5-Config MiniCPM5ConfigTest.cpp)
target_link_libraries(Mllm-Test-MiniCPM5-Config PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-MiniCPM5-Config PRIVATE ${MLLM_INCLUDE_DIR})
target_compile_definitions(Mllm-Test-MiniCPM5-Config
                           PRIVATE MINICPM5_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../examples/minicpm5")

add_executable(Mllm-Test-MiniCPM5-Tokenizer MiniCPM5TokenizerTest.cpp)
target_link_libraries(Mllm-Test-MiniCPM5-Tokenizer PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-MiniCPM5-Tokenizer PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-MiniCPM5-Model MiniCPM5ModelTest.cpp)
target_link_libraries(Mllm-Test-MiniCPM5-Model PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-MiniCPM5-Model PRIVATE ${MLLM_INCLUDE_DIR})
target_compile_definitions(Mllm-Test-MiniCPM5-Model
                           PRIVATE MINICPM5_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../examples/minicpm5")

add_executable(Mllm-Test-CPUContiguousOp ContiguousOpTest.cpp)
target_link_libraries(Mllm-Test-CPUContiguousOp PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-CPUContiguousOp PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
