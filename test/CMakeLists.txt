cmake_minimum_required(VERSION 3.12)

add_executable(MallardTests
    test_main.cpp
    common_math_test.cpp
    time_integrator_test.cpp
    physics_test.cpp
)

target_include_directories(MallardTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MallardTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set_target_properties(MallardTests PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

find_package(GTest REQUIRED)
target_link_libraries(MallardTests PRIVATE GTest::GTest GTest::Main)

add_dependencies(MallardTests Mallard)
target_link_libraries(MallardTests PUBLIC tomlplusplus::tomlplusplus)
target_link_libraries(MallardTests PRIVATE CommonLib)
target_link_libraries(MallardTests PRIVATE SolverLib)
target_link_libraries(MallardTests PRIVATE NumericsLib)

target_compile_options(MallardTests PRIVATE -Wall -Wextra -pedantic)