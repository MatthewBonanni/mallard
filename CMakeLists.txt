cmake_minimum_required(VERSION 3.12)
project(FVCode)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(src)
add_subdirectory(test)

install(TARGETS FVCode
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)