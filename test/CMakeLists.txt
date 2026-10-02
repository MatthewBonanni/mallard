# CMakeLists.txt for MallardTest
# Tests shared by both dimensions, then those specific to one
set(MALLARD_TEST_SOURCES
    test_main.cpp
    common_math_test.cpp
    time_integrator_test.cpp
    comm_test.cpp
    mesh_block_test.cpp
    distributed_mesh_test.cpp
)
if(Mallard_DIM EQUAL 2)
    list(APPEND MALLARD_TEST_SOURCES
    mesh_test.cpp
    physics_test.cpp
    riemann_solver_test.cpp
    reconstruction_test.cpp
    solver_test.cpp
    teno_test.cpp
    viscous_test.cpp
    restart_test.cpp
    boundary_test.cpp
    source_test.cpp
    io_test.cpp
    mesh_file_test.cpp
    mpi_test.cpp
    periodic_test.cpp
    )
else()
    list(APPEND MALLARD_TEST_SOURCES
    physics3d_test.cpp
    mesh3d_test.cpp
    teno3d_test.cpp
    riemann3d_test.cpp
    boundary3d_test.cpp
    gradient3d_test.cpp
    solver3d_test.cpp
    io3d_test.cpp
    mpi3d_test.cpp
    periodic3d_test.cpp
    )
endif()
add_executable(MallardTest ${MALLARD_TEST_SOURCES})

target_include_directories(MallardTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.18.0
  ${MALLARD_THIRD_PARTY}
)
FetchContent_MakeAvailable(googletest)

target_link_libraries(MallardTest PRIVATE gtest)

target_link_libraries(MallardTest PRIVATE MallardCore)
mallard_target_warnings(MallardTest)
