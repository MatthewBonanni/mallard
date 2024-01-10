cmake_minimum_required(VERSION 3.12)
project(Mallard VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(INSTALL_DOC "Set to OFF to skip build/install Documentation" ON)

add_subdirectory(src)
add_subdirectory(test)

install(TARGETS Mallard
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)