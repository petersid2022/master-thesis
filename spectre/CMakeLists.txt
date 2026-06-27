cmake_minimum_required(VERSION 3.18)
project(spectre LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLAMA_DIR
    "${CMAKE_SOURCE_DIR}/../llama.cpp"
    CACHE PATH "Path to llama.cpp source tree"
)

find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "${LLAMA_DIR}/build:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(LLAMA REQUIRED IMPORTED_TARGET llama)

add_executable(spectre src/main.cpp)

set_target_properties(
    spectre
    PROPERTIES CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    BUILD_RPATH "${LLAMA_DIR}/build/bin"
    INSTALL_RPATH "${LLAMA_DIR}/build/bin"
)

target_compile_options(spectre PRIVATE
    -Wall -Wextra
    -Werror
    -Wpedantic
    -Wconversion
)

target_include_directories(spectre PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_directories(spectre PRIVATE "${LLAMA_DIR}/build/bin")
target_link_libraries(spectre PRIVATE PkgConfig::LLAMA)

target_link_options(spectre PRIVATE
    "-Wl,-rpath-link,${LLAMA_DIR}/build/bin"
)

foreach(_cuda_stub_dir
        /usr/local/cuda/lib64/stubs
        /usr/local/cuda/targets/x86_64-linux/lib/stubs)
    if(EXISTS "${_cuda_stub_dir}/libcuda.so")
        message(STATUS "spectre: linking against CUDA driver stub at ${_cuda_stub_dir}")
        target_link_directories(spectre PRIVATE "${_cuda_stub_dir}")
        target_link_libraries(spectre PRIVATE cuda)
        break()
    endif()
endforeach()

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    target_compile_options(spectre PRIVATE
        -fsanitize=address,leak
        -fstack-protector-strong
        -fno-omit-frame-pointer
    )
    target_link_options(spectre PRIVATE
        -fsanitize=address,leak
    )
else()
    set_target_properties(spectre PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
    target_compile_options(spectre PRIVATE
        $<$<CONFIG:Release>:-ffp-contract=fast>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-O3>
    )
endif()
