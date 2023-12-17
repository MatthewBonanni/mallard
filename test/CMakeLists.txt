cmake_minimum_required(VERSION 3.12)

add_executable(FVCodeTests
    test_main.cpp
    add_test.cpp
)

target_include_directories(FVCodeTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(FVCodeTests PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

find_package(GTest REQUIRED)
target_link_libraries(FVCodeTests PRIVATE GTest::GTest GTest::Main)

add_dependencies(FVCodeTests FVCode)
target_link_libraries(FVCodeTests PRIVATE FVCodeLibrary)

target_compile_options(FVCodeTests PRIVATE -Wall -Wextra -pedantic)