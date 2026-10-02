# CMakeLists.txt for Mallard
add_library(MallardCore STATIC
    common/common_math.cpp
    common/common_io.cpp
    mesh/mesh.cpp
    mesh/mesh_io.cpp
    mesh/mesh_3d.cpp
    mesh/mesh_block.cpp
    mesh/zone.cpp
    boundary/boundary.cpp
    numerics/face_reconstruction.cpp
    numerics/quadrature.cpp
    numerics/teno.cpp
    numerics/time_integrator.cpp
    physics/physics.cpp
    solver/solver.cpp
    solver/solver_initialize.cpp
    solver/solver_rhs.cpp
    io/data_writer.cpp
    parallel/comm.cpp
    parallel/distribution.cpp
    parallel/distributed_mesh.cpp
    parallel/halo_exchange.cpp
    parallel/partition.cpp
)

target_include_directories(MallardCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/boundary
    ${CMAKE_CURRENT_SOURCE_DIR}/numerics
    ${CMAKE_CURRENT_SOURCE_DIR}/physics
    ${CMAKE_CURRENT_SOURCE_DIR}/solver
    ${CMAKE_CURRENT_SOURCE_DIR}/io
    ${CMAKE_CURRENT_SOURCE_DIR}/parallel
)
target_link_libraries(MallardCore PUBLIC Kokkos::kokkos)
mallard_target_warnings(MallardCore)
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
endif()

if(Mallard_ENABLE_HDF5)
    target_include_directories(MallardCore SYSTEM PUBLIC ${HDF5_C_INCLUDE_DIRS})
    target_compile_definitions(MallardCore PUBLIC Mallard_HAS_HDF5 ${HDF5_C_DEFINITIONS})
    target_link_libraries(MallardCore PUBLIC ${HDF5_C_LIBRARIES})
endif()

add_executable(Mallard main.cpp)
target_link_libraries(Mallard PRIVATE MallardCore)
mallard_target_warnings(Mallard)

if(Mallard_ENABLE_HDF5)
    add_executable(mallard-mesh-convert mesh_convert.cpp)
    target_link_libraries(mallard-mesh-convert PRIVATE MallardCore)
    mallard_target_warnings(mallard-mesh-convert)
endif()
