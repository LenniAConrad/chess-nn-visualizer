cmake_minimum_required(VERSION 3.20)
project(cpp_nn_visualizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CNNV_WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)
else()
    set(CNNV_WARNING_FLAGS)
endif()

include(FetchContent)
include(CheckLanguage)
find_package(Threads REQUIRED)
find_package(Doxygen QUIET)

option(CNNV_ENABLE_CUDA "Build optional CUDA tensor kernels when nvcc is available" ON)
set(CNNV_CUDA_SRC)
if(CNNV_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86" CACHE STRING
                "CUDA architectures to build for")
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        list(APPEND CNNV_CUDA_SRC src/nn/ops/CudaOps.cu)
        message(STATUS "CUDA tensor kernels enabled (${CMAKE_CUDA_COMPILER})")
    else()
        message(STATUS "CUDA requested but no CUDA compiler was found; using CPU tensor kernels")
    endif()
endif()

# Always fetch raylib via FetchContent. find_package was producing a stub
# IMPORTED target even on systems without raylib installed, which collided
# with the FetchContent build of the real target.
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
# CUSTOMIZE_BUILD must be ON for the SUPPORT_FILEFORMAT_* flags below to take
# effect — without it raylib uses its default config.h and SVG is off.
set(CUSTOMIZE_BUILD ON CACHE BOOL "" FORCE)
set(SUPPORT_MODULE_RSHAPES ON CACHE BOOL "" FORCE)
set(SUPPORT_MODULE_RTEXT   ON CACHE BOOL "" FORCE)
set(SUPPORT_MODULE_RTEXTURES ON CACHE BOOL "" FORCE)
set(SUPPORT_MODULE_RAUDIO OFF CACHE BOOL "" FORCE)
# Image format support. PNG is on by default; SVG must be opted in because
# the piece sprites in assets/pieces are SVG (rasterised at load time).
set(SUPPORT_FILEFORMAT_PNG ON CACHE BOOL "" FORCE)
set(SUPPORT_FILEFORMAT_SVG ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(raylib)

# ---------------------------------------------------------------------------
# Core static library: chess + nn + io. No raylib dependency.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CNNV_CORE_SRC
    CONFIGURE_DEPENDS
    src/chess/*.cpp
    src/game/*.cpp
    src/nn/*.cpp
    src/io/*.cpp
)

add_library(cnnv_core STATIC ${CNNV_CORE_SRC} ${CNNV_CUDA_SRC})
target_include_directories(cnnv_core PUBLIC src)
target_compile_options(cnnv_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CNNV_WARNING_FLAGS}>)
if(CNNV_CUDA_SRC)
    target_compile_definitions(cnnv_core PUBLIC CNNV_HAS_CUDA=1)
    set_target_properties(cnnv_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RUNTIME_LIBRARY Shared)
endif()
set_target_properties(cnnv_core PROPERTIES LINKER_LANGUAGE CXX)

# ---------------------------------------------------------------------------
# Visualization static library: raylib-dependent UI.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CNNV_VIZ_SRC
    CONFIGURE_DEPENDS
    src/viz/*.cpp
)

add_library(cnnv_viz STATIC ${CNNV_VIZ_SRC})
target_include_directories(cnnv_viz PUBLIC src)
target_compile_options(cnnv_viz PRIVATE ${CNNV_WARNING_FLAGS})
target_link_libraries(cnnv_viz PUBLIC cnnv_core raylib Threads::Threads)
set_target_properties(cnnv_viz PROPERTIES LINKER_LANGUAGE CXX)

# ---------------------------------------------------------------------------
# Main executable.
# ---------------------------------------------------------------------------
add_executable(cnnv src/main.cpp)
target_compile_options(cnnv PRIVATE ${CNNV_WARNING_FLAGS})
target_link_libraries(cnnv PRIVATE cnnv_viz cnnv_core raylib)

# ---------------------------------------------------------------------------
# Optional API documentation. Available when Doxygen is installed.
# ---------------------------------------------------------------------------
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating Doxygen API documentation"
        VERBATIM)
endif()

# ---------------------------------------------------------------------------
# Tests. Run with `ctest --test-dir build` after building.
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
