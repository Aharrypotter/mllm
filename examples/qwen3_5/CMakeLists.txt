add_executable(mllm-qwen3-5-runner main.cpp)
target_link_libraries(mllm-qwen3-5-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-5-runner PRIVATE ${MLLM_INCLUDE_DIR})

set(MLLM_QWEN35_VIDEO_DECODER_BACKEND "none" CACHE STRING
    "Qwen3.5 example video decoder backend: none or portable")
set_property(CACHE MLLM_QWEN35_VIDEO_DECODER_BACKEND PROPERTY STRINGS none portable)
set(MLLM_QWEN35_MINIMP4_INCLUDE_DIR "" CACHE PATH "Directory containing minimp4.h")
set(MLLM_QWEN35_OPENH264_SOURCE_DIR "" CACHE PATH "Pinned OpenH264 source tree used for API headers")
set(MLLM_QWEN35_OPENH264_LIBRARIES "" CACHE STRING "OpenH264 libraries used by the portable video decoder backend")
if("${MLLM_QWEN35_VIDEO_DECODER_BACKEND}" STREQUAL "portable")
  if(NOT EXISTS "${MLLM_QWEN35_MINIMP4_INCLUDE_DIR}/minimp4.h")
    message(FATAL_ERROR "MLLM_QWEN35_MINIMP4_INCLUDE_DIR must contain minimp4.h")
  endif()
  if(NOT EXISTS "${MLLM_QWEN35_OPENH264_SOURCE_DIR}/codec/api/wels/codec_api.h")
    message(FATAL_ERROR "MLLM_QWEN35_OPENH264_SOURCE_DIR must contain pinned OpenH264 API headers")
  endif()
  if(NOT MLLM_QWEN35_OPENH264_LIBRARIES)
    message(FATAL_ERROR "MLLM_QWEN35_OPENH264_LIBRARIES must name the matching decoder libraries")
  endif()
  target_sources(mllm-qwen3-5-runner PRIVATE video_decoder.cpp)
  target_include_directories(mllm-qwen3-5-runner PRIVATE ${MLLM_QWEN35_MINIMP4_INCLUDE_DIR}
                                                         ${MLLM_QWEN35_OPENH264_SOURCE_DIR}/codec/api)
  target_link_libraries(mllm-qwen3-5-runner PRIVATE ${MLLM_QWEN35_OPENH264_LIBRARIES})
  target_compile_definitions(mllm-qwen3-5-runner PRIVATE MLLM_QWEN35_VIDEO_DECODER_BACKEND_PORTABLE=1)

  add_executable(mllm-qwen3-5-portable-video-decoder-smoke video_decoder_smoke.cpp video_decoder.cpp)
  target_include_directories(mllm-qwen3-5-portable-video-decoder-smoke PRIVATE ${MLLM_INCLUDE_DIR}
                                                                                ${MLLM_QWEN35_MINIMP4_INCLUDE_DIR}
                                                                                ${MLLM_QWEN35_OPENH264_SOURCE_DIR}/codec/api)
  target_link_libraries(mllm-qwen3-5-portable-video-decoder-smoke PRIVATE MllmRT MllmCPUBackend
                                                                               ${MLLM_QWEN35_OPENH264_LIBRARIES})
  if(MLLM_ENABLE_TEST)
    add_test(
      NAME Qwen35PortableVideoDecoderSmoke
      COMMAND
        mllm-qwen3-5-portable-video-decoder-smoke
        ${CMAKE_SOURCE_DIR}/bench_assets/qwen35_moving_square_192x128_4fps_4s.mp4
        ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moving_square_selected.rgb24
        ${CMAKE_CURRENT_BINARY_DIR}/qwen35_moving_square_patches.f32
        --expect-moving-square)
    set_tests_properties(Qwen35PortableVideoDecoderSmoke PROPERTIES LABELS "qwen35;video;portable-decoder")
  endif()
elseif(NOT "${MLLM_QWEN35_VIDEO_DECODER_BACKEND}" STREQUAL "none")
  message(FATAL_ERROR
          "Unsupported MLLM_QWEN35_VIDEO_DECODER_BACKEND='${MLLM_QWEN35_VIDEO_DECODER_BACKEND}'; expected none or portable")
endif()
