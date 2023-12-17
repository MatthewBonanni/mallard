cmake_minimum_required(VERSION 3.12)

add_library(MathFunctions
    math/add.cpp
)

add_executable(FVCode
    main.cpp
)

target_include_directories(FVCode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(FVCode MathFunctions)

set_target_properties(FVCode PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

target_compile_options(FVCode PRIVATE -Wall -Wextra -pedantic)