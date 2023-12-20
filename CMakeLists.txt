cmake_minimum_required(VERSION 3.12)
project(Mallard)

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(INSTALL_DOC "Set to OFF to skip build/install Documentation" ON)

add_subdirectory(src)
add_subdirectory(test)

install(TARGETS Mallard
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)