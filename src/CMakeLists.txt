# CMakeLists.txt for Mallard
add_library(MallardCore STATIC
    common/common_math.cpp
    common/common_io.cpp
    common/build_info.cpp
    common/growing_buffer.cpp
    common/log.cpp
    mesh/mesh.cpp
    mesh/mesh_io.cpp
    mesh/periodic.cpp
    mesh/mesh_3d.cpp
    mesh/mesh_block.cpp
    mesh/zone.cpp
    boundary/boundary.cpp
    boundary/synthetic_inflow.cpp
    chemistry/mechanism.cpp
    chemistry/transport.cpp
    chemistry/units.cpp
    numerics/face_reconstruction.cpp
    numerics/quadrature.cpp
    numerics/scalar_reconstruction.cpp
    numerics/teno.cpp
    numerics/teno_setup.cpp
    numerics/time_integrator.cpp
    physics/mixture.cpp
    physics/les.cpp
    physics/pasr.cpp
    physics/tfles.cpp
    physics/physics.cpp
    solver/cell_chemistry.cpp
    solver/chemistry_balance.cpp
    solver/chemistry_benchmark.cpp
    solver/solver.cpp
    solver/solver_chemistry.cpp
    solver/solver_graph.cpp
    solver/solver_initialize.cpp
    solver/solver_mixture.cpp
    solver/solver_les.cpp
    solver/solver_rhs.cpp
    solver/solver_sponge.cpp
    solver/solver_inflow.cpp
    solver/statistics.cpp
    io/data_writer.cpp
    io/data_writer_hdf5.cpp
    parallel/comm.cpp
    parallel/device_comm.cpp
    parallel/distribution.cpp
    parallel/distributed_mesh.cpp
    parallel/global_order.cpp
    parallel/halo_exchange.cpp
    parallel/partition.cpp
)

target_include_directories(MallardCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/boundary
    ${CMAKE_CURRENT_SOURCE_DIR}/chemistry
    ${CMAKE_CURRENT_SOURCE_DIR}/numerics
    ${CMAKE_CURRENT_SOURCE_DIR}/physics
    ${CMAKE_CURRENT_SOURCE_DIR}/solver
    ${CMAKE_CURRENT_SOURCE_DIR}/io
    ${CMAKE_CURRENT_SOURCE_DIR}/parallel
)
target_link_libraries(MallardCore PUBLIC Kokkos::kokkos)
# The TENO setup computes every table with the same operations on the host
# and on the device, and contracting multiplies and adds into FMAs would
# change them differently on each: no contraction in that file
if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    set_source_files_properties(numerics/teno_setup.cpp PROPERTIES COMPILE_OPTIONS "--fmad=false;-ffp-contract=off")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(numerics/teno_setup.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_link_libraries(MallardCore PRIVATE yaml-cpp::yaml-cpp)
mallard_target_warnings(MallardCore)

configure_file(common/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h @ONLY)
target_include_directories(MallardCore PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
if(Mallard_ENABLE_MPI)
    target_link_libraries(MallardCore PUBLIC MPI::MPI_CXX)
    target_compile_definitions(MallardCore PUBLIC Mallard_HAS_MPI)
    if(Mallard_ENABLE_KAMINPAR)
        target_link_libraries(MallardCore PUBLIC KaMinPar::dKaMinPar)
        target_compile_definitions(MallardCore PUBLIC Mallard_HAS_KAMINPAR)
    endif()
    if(Mallard_GPU_AWARE_MPI)
        target_compile_definitions(MallardCore PUBLIC Mallard_GPU_AWARE_MPI)
    endif()
    if(Mallard_ENABLE_NCCL)
        target_include_directories(MallardCore SYSTEM PUBLIC ${NCCL_INCLUDE_DIR})
        target_link_libraries(MallardCore PUBLIC ${NCCL_LIBRARY})
        target_compile_definitions(MallardCore PUBLIC Mallard_HAS_NCCL)
    endif()
endif()

if(Mallard_ENABLE_HDF5)
    target_include_directories(MallardCore SYSTEM PUBLIC ${HDF5_C_INCLUDE_DIRS})
    target_compile_definitions(MallardCore PUBLIC Mallard_HAS_HDF5 ${HDF5_C_DEFINITIONS})
    target_link_libraries(MallardCore PUBLIC ${HDF5_C_LIBRARIES})
endif()

add_executable(Mallard main.cpp)
target_link_libraries(Mallard PRIVATE MallardCore)
mallard_target_warnings(Mallard)

add_executable(MallardReactor reactor_main.cpp)
target_link_libraries(MallardReactor PRIVATE MallardCore)
mallard_target_warnings(MallardReactor)

if(Mallard_ENABLE_HDF5)
    add_executable(mallard-mesh-convert mesh_convert.cpp)
    target_link_libraries(mallard-mesh-convert PRIVATE MallardCore)
    mallard_target_warnings(mallard-mesh-convert)
endif()
