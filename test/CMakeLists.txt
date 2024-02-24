# CMakeLists.txt for MallardTest
add_executable(MallardTest
    test_main.cpp
    common_math_test.cpp
    riemann_solver_test.cpp
    time_integrator_test.cpp
    physics_test.cpp
)

target_include_directories(MallardTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

target_link_libraries(MallardTest PRIVATE gtest_main)

add_dependencies(MallardTest Mallard)
target_link_libraries(MallardTest PRIVATE Kokkos::kokkos)
target_link_libraries(MallardTest PRIVATE CommonLib)
target_link_libraries(MallardTest PRIVATE SolverLib)
target_link_libraries(MallardTest PRIVATE NumericsLib)
target_link_libraries(MallardTest PRIVATE PhysicsLib)

target_compile_options(MallardTest PRIVATE -Wall -Wextra -pedantic)
