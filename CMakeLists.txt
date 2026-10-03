cmake_minimum_required(VERSION 3.16)
project(Mallard VERSION 0.4.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# Version string for the run header, fixed at configure time
set(Mallard_GIT_DESCRIBE "")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE Mallard_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Warnings for Mallard's own targets; third-party headers are included as SYSTEM
option(${PROJECT_NAME}_WARNINGS_AS_ERRORS "Treat warnings in Mallard's own code as errors" OFF)
function(mallard_target_warnings target)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        return()
    endif()
    # Kokkos builds for CUDA with nvcc_wrapper, which CMake identifies as GNU
    if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(nvcc TRUE)
    endif()
    set(warnings
        -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wcast-align -Wunused
        -Woverloaded-virtual -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough)
    # nvcc rewrites functional casts such as T(x) into C-style casts before the host compiler sees them
    if(NOT nvcc)
        list(APPEND warnings -Wold-style-cast)
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND warnings -Wduplicated-cond -Wlogical-op)
    else()
        # GCC's -Wnull-dereference flags code inlined from toml11 and exprtk, which SYSTEM cannot silence
        list(APPEND warnings -Wnull-dereference)
    endif()
    if(${PROJECT_NAME}_WARNINGS_AS_ERRORS)
        list(APPEND warnings -Werror)
        if(nvcc)
            list(APPEND warnings "SHELL:-Werror all-warnings")
        endif()
    endif()
    target_compile_options(${target} PRIVATE ${warnings})
endfunction()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
    set(MALLARD_THIRD_PARTY SYSTEM)
endif()

# Compile-time options
option(${PROJECT_NAME}_USE_DOUBLE "Use double precision" ON)
message(STATUS "Mallard: Double precision: ${${PROJECT_NAME}_USE_DOUBLE}")
option(${PROJECT_NAME}_SPECIES_LAYOUT_LEFT "Species partial densities stored species-major (LayoutLeft)" OFF)
if(${PROJECT_NAME}_SPECIES_LAYOUT_LEFT)
    add_compile_definitions(${PROJECT_NAME}_SPECIES_LAYOUT_LEFT)
endif()
if(${PROJECT_NAME}_USE_DOUBLE)
    add_compile_definitions(${PROJECT_NAME}_USE_DOUBLE)
endif()

set(${PROJECT_NAME}_DIM 2 CACHE STRING "Spatial dimension (2 or 3)")
set_property(CACHE ${PROJECT_NAME}_DIM PROPERTY STRINGS 2 3)
if(NOT ${PROJECT_NAME}_DIM MATCHES "^[23]$")
    message(FATAL_ERROR "Mallard_DIM must be 2 or 3")
endif()
message(STATUS "Mallard: Dimension: ${${PROJECT_NAME}_DIM}")
add_compile_definitions(${PROJECT_NAME}_DIM=${${PROJECT_NAME}_DIM})

option(${PROJECT_NAME}_ENABLE_MPI "Distributed memory parallelism with MPI" OFF)
if(${PROJECT_NAME}_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    message(STATUS "Mallard: MPI ${MPI_CXX_VERSION}")
    option(${PROJECT_NAME}_GPU_AWARE_MPI "Pass device buffers to MPI directly (CUDA-aware MPI)" OFF)
    option(${PROJECT_NAME}_ENABLE_KAMINPAR "Graph partitioning with dKaMinPar (needs oneTBB)" OFF)
    if(${PROJECT_NAME}_ENABLE_KAMINPAR)
        include(FetchContent)
        FetchContent_Declare(KaMinPar
            GIT_REPOSITORY https://github.com/KaHIP/KaMinPar.git
            GIT_TAG v3.7.3
            EXCLUDE_FROM_ALL
            ${MALLARD_THIRD_PARTY})
        set(KAMINPAR_BUILD_DISTRIBUTED ON CACHE BOOL "" FORCE)
        set(KAMINPAR_BUILD_APPS OFF CACHE BOOL "" FORCE)
        set(KAMINPAR_BUILD_IO OFF CACHE BOOL "" FORCE)
        set(KAMINPAR_BUILD_WITH_SPARSEHASH OFF CACHE BOOL "" FORCE)
        set(KAMINPAR_BUILD_WITH_CCACHE OFF CACHE BOOL "" FORCE)
        set(KAMINPAR_BUILD_WITH_MTUNE_NATIVE OFF CACHE BOOL "" FORCE)
        set(KAMINPAR_64BIT_NODE_IDS ON CACHE BOOL "" FORCE)
        set(KAMINPAR_64BIT_EDGE_IDS ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(KaMinPar)
    endif()
endif()

# HDF5 mesh files (parallel HDF5 when MPI is on and the library supports it)
option(${PROJECT_NAME}_ENABLE_HDF5 "HDF5 mesh files and the mallard-mesh-convert tool" OFF)
option(USE_SYSTEM_HDF5 "Use system-installed HDF5" ON)
if(${PROJECT_NAME}_ENABLE_HDF5)
    if(USE_SYSTEM_HDF5)
        if(${PROJECT_NAME}_ENABLE_MPI)
            set(HDF5_PREFER_PARALLEL TRUE)
        endif()
        find_package(HDF5 REQUIRED COMPONENTS C)
        message(STATUS "Mallard: HDF5 ${HDF5_VERSION} (parallel: ${HDF5_IS_PARALLEL})")
    else()
        message(STATUS "Using external HDF5 from submodule")
        set(HDF5_ENABLE_PARALLEL ${${PROJECT_NAME}_ENABLE_MPI} CACHE BOOL "" FORCE)
        add_subdirectory(src/external/hdf5 ${CMAKE_BINARY_DIR}/external/hdf5 ${MALLARD_THIRD_PARTY})
        set(HDF5_C_LIBRARIES hdf5-static)
    endif()
endif()

# Add Kokkos dependency
option(USE_SYSTEM_KOKKOS "Use system-installed Kokkos" ON)
if(USE_SYSTEM_KOKKOS)
    find_package(Kokkos 5 REQUIRED)
else()
    add_subdirectory(src/external/kokkos ${CMAKE_BINARY_DIR}/external/kokkos ${MALLARD_THIRD_PARTY})
endif()

# Cantera YAML mechanism files
include(FetchContent)
FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.9.0
    EXCLUDE_FROM_ALL
    ${MALLARD_THIRD_PARTY})
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)
# yaml-cpp has no device code, and nvcc cannot compile its bundled dragonbox:
# Kokkos' nvcc_wrapper passes it to the host compiler
if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    target_compile_options(yaml-cpp PRIVATE --host-only)
endif()

# Include other external libraries
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/src/external/toml11/include")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/src/external/exprtk")

# Add your main project source
add_subdirectory(src)
add_subdirectory(test)

# Documentation setup
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    find_package(Doxygen)
    if(Doxygen_FOUND)
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/docs/Doxyfile
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs
            COMMENT "Generating documentation with Doxygen"
            VERBATIM
        )
        add_dependencies(Mallard docs)
    else()
        message(WARNING "Doxygen not found. Documentation won't be generated.")
    endif()
endif()

# Installation setup
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX /path/to/install/dir CACHE PATH "Default installation path" FORCE)
endif()

install(TARGETS Mallard
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

# Enable testing
enable_testing()
add_test(NAME MallardTest COMMAND MallardTest)