# CMakeLists.txt for MallardTest
add_executable(MallardTest
    test_main.cpp
    common_math_test.cpp
    mesh_test.cpp
    physics_test.cpp
    riemann_solver_test.cpp
    time_integrator_test.cpp
    reconstruction_test.cpp
    solver_test.cpp
    teno_test.cpp
    viscous_test.cpp
    restart_test.cpp
    boundary_test.cpp
    source_test.cpp
    io_test.cpp
    mesh_file_test.cpp
    comm_test.cpp
)

target_include_directories(MallardTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.18.0
)
FetchContent_MakeAvailable(googletest)

target_link_libraries(MallardTest PRIVATE gtest)

target_link_libraries(MallardTest PRIVATE MallardCore)
