# CMakeLists.txt for MallardTest
add_executable(MallardTest
    test_main.cpp
    common_math_test.cpp
    time_integrator_test.cpp
    physics_test.cpp
)

target_include_directories(MallardTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MallardTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set_target_properties(MallardTest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

find_package(GTest REQUIRED)
target_link_libraries(MallardTest PRIVATE GTest::GTest GTest::Main)

add_dependencies(MallardTest Mallard)
target_link_libraries(MallardTest PRIVATE tomlplusplus::tomlplusplus)
target_link_libraries(MallardTest PRIVATE CommonLib)
target_link_libraries(MallardTest PRIVATE SolverLib)
target_link_libraries(MallardTest PRIVATE NumericsLib)

target_compile_options(MallardTest PRIVATE -Wall -Wextra -pedantic)