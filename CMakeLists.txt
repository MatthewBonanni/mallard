cmake_minimum_required(VERSION 3.16)
project(Mallard VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(src)
add_subdirectory(test)

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

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX /path/to/install/dir CACHE PATH "Default installation path" FORCE)
endif()

install(TARGETS Mallard
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

enable_testing()
add_test(NAME MallardTest COMMAND MallardTest)