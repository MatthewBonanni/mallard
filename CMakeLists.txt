cmake_minimum_required(VERSION 3.16)
project(Mallard VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Build options
add_compile_options(-Wall -Wextra -Wpedantic)

# Compile-time options
option(${PROJECT_NAME}_USE_DOUBLES "Use double precision" ON)
message(STATUS "Mallard: Double precision: ${${PROJECT_NAME}_USE_DOUBLES}")
if(${PROJECT_NAME}_USE_DOUBLES)
    add_compile_definitions(${PROJECT_NAME}_USE_DOUBLES)
endif()

# Add Kokkos and KokkosKernels subdirectories
option(USE_SYSTEM_KOKKOS "Use system-installed Kokkos" ON)

if(USE_SYSTEM_KOKKOS)
    find_package(Kokkos REQUIRED)
    find_package(KokkosKernels REQUIRED)
else()
    add_subdirectory(src/external/kokkos ${CMAKE_BINARY_DIR}/external/kokkos)
    add_subdirectory(src/external/kokkos-kernels ${CMAKE_BINARY_DIR}/external/kokkos-kernels)
    include_directories(${CMAKE_BINARY_DIR}/external/kokkos)
    include_directories(${CMAKE_BINARY_DIR}/external/kokkos-kernels)
endif()

# Include other external libraries
include_directories("${CMAKE_SOURCE_DIR}/src/external/toml11/include")
include_directories("${CMAKE_SOURCE_DIR}/src/external/exprtk")

# Add your main project source
add_subdirectory(src)
add_subdirectory(test)

# Documentation setup
option(BUILD_DOCS "Build documentation" ON)
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
