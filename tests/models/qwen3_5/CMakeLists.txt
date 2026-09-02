add_executable(Mllm-Test-Qwen35-Config Qwen35ConfigTest.cpp)
target_link_libraries(Mllm-Test-Qwen35-Config PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Qwen35-Config PRIVATE ${MLLM_INCLUDE_DIR})
target_compile_definitions(Mllm-Test-Qwen35-Config
                           PRIVATE QWEN35_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/qwen3_5")

add_executable(Mllm-Test-Qwen35-Tokenizer Qwen35TokenizerTest.cpp)
target_link_libraries(Mllm-Test-Qwen35-Tokenizer PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Qwen35-Tokenizer PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Qwen35-Multimodal Qwen35MultimodalTest.cpp)
target_link_libraries(Mllm-Test-Qwen35-Multimodal PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Qwen35-Multimodal PRIVATE ${MLLM_INCLUDE_DIR})

add_test(NAME Qwen35ConfigFocused COMMAND Mllm-Test-Qwen35-Config)
add_test(NAME Qwen35TokenizerFocused COMMAND Mllm-Test-Qwen35-Tokenizer)
add_test(NAME Qwen35MultimodalFocused COMMAND Mllm-Test-Qwen35-Multimodal)
set_tests_properties(Qwen35ConfigFocused Qwen35TokenizerFocused Qwen35MultimodalFocused PROPERTIES LABELS qwen35)
