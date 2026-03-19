cmake_minimum_required(VERSION 3.18)
project(spectre LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLAMA_DIR
    "${CMAKE_SOURCE_DIR}/../llama.cpp"
    CACHE PATH "Path to llama.cpp source tree")

find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "${LLAMA_DIR}/build:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(LLAMA REQUIRED IMPORTED_TARGET llama)

add_executable(main src/main.cpp)

set_target_properties(
  main
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             BUILD_RPATH "${LLAMA_DIR}/build/bin"
             INSTALL_RPATH "${LLAMA_DIR}/build/bin")

target_compile_options(main PRIVATE -Wall -Wextra -Werror -pedantic)

target_include_directories(main PRIVATE ${LLAMA_DIR}/include
                                        ${LLAMA_DIR}/ggml/include)

target_link_directories(main PRIVATE "${LLAMA_DIR}/build/bin")
target_link_libraries(main PRIVATE PkgConfig::LLAMA)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
  target_compile_options(main PRIVATE -fsanitize=address,leak
                                      -fno-omit-frame-pointer)
  target_link_options(main PRIVATE -fsanitize=address,leak)
endif()
