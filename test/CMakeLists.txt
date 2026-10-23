# CMakeLists.txt for MallardTest
# Tests shared by both dimensions, then those specific to one
set(MALLARD_TEST_SOURCES
    test_main.cpp
    scratch.cpp
    common_math_test.cpp
    compiled_expression_test.cpp
    region_test.cpp
    case_template_test.cpp
    time_integrator_test.cpp
    comm_test.cpp
    chemistry_thermo_test.cpp
    chemistry_kinetics_test.cpp
    chemistry_reactor_test.cpp
    chemistry_benchmark_test.cpp
    chemistry_transport_test.cpp
    mixture_test.cpp
    radiation_test.cpp
    mixture_transport_test.cpp
    mesh_block_test.cpp
    mesh_import_test.cpp
    distributed_mesh_test.cpp
    statistics_test.cpp
    probes_test.cpp
    remap_test.cpp
    les_test.cpp
    numerics_preset_test.cpp
    tfles_test.cpp
    check_test.cpp
    synthetic_inflow_test.cpp
    input_upgrade_test.cpp
    format_compat_test.cpp
    wall_model_test.cpp
    units_test.cpp
)
if(Mallard_DIM EQUAL 2)
    list(APPEND MALLARD_TEST_SOURCES
    mesh_test.cpp
    curved_test.cpp
    physics_test.cpp
    riemann_solver_test.cpp
    reconstruction_test.cpp
    solver_test.cpp
    teno_test.cpp
    viscous_test.cpp
    restart_test.cpp
    run_control_test.cpp
    step_control_test.cpp
    boundary_test.cpp
    nscbc_test.cpp
    source_test.cpp
    axisymmetric_test.cpp
    io_test.cpp
    mesh_file_test.cpp
    mpi_test.cpp
    periodic_test.cpp
    preset_validation_test.cpp
    )
else()
    list(APPEND MALLARD_TEST_SOURCES
    physics3d_test.cpp
    mesh3d_test.cpp
    curved3d_test.cpp
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

# exprtk, the reference of the compiled expressions, rounds every operation on its own there
if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    set_source_files_properties(compiled_expression_test.cpp PROPERTIES COMPILE_OPTIONS "--fmad=false;-ffp-contract=off")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(compiled_expression_test.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_include_directories(MallardTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(MallardTest PRIVATE MALLARD_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
                                               MALLARD_EXECUTABLE="$<TARGET_FILE:Mallard>")
add_dependencies(MallardTest Mallard)
# The run-control tests start runs on several ranks with the MPI launcher
if(Mallard_ENABLE_MPI AND MPIEXEC_EXECUTABLE)
    target_compile_definitions(MallardTest PRIVATE MALLARD_MPIEXEC="${MPIEXEC_EXECUTABLE}")
    execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version OUTPUT_VARIABLE mpiexec_version ERROR_QUIET)
    if(mpiexec_version MATCHES "Open MPI|OpenRTE|PRRTE")
        target_compile_definitions(MallardTest PRIVATE MALLARD_MPIEXEC_OPEN_MPI)
    endif()
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.18.0
  ${MALLARD_THIRD_PARTY}
)
# Test-only: `cmake --install` leaves GoogleTest out
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_link_libraries(MallardTest PRIVATE gtest)

target_link_libraries(MallardTest PRIVATE MallardCore)
# Malformed CGNS files written by the tests
if(Mallard_ENABLE_CGNS)
    target_link_libraries(MallardTest PRIVATE ${MALLARD_CGNS_LIBRARY})
endif()
mallard_target_warnings(MallardTest)
