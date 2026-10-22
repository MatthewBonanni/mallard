# CMakeLists.txt for Mallard
add_library(MallardCore STATIC
    common/common_math.cpp
    common/common_io.cpp
    common/input.cpp
    common/units.cpp
    common/memory_tracker.cpp
    common/build_info.cpp
    common/case_template.cpp
    common/growing_buffer.cpp
    common/log.cpp
    mesh/curved.cpp
    mesh/mesh.cpp
    mesh/mesh_io.cpp
    mesh/read_cgns.cpp
    mesh/read_fluent.cpp
    mesh/read_openfoam.cpp
    mesh/cells_from_faces.cpp
    mesh/periodic.cpp
    mesh/mesh_3d.cpp
    mesh/mesh_block.cpp
    mesh/mesh_quality.cpp
    mesh/zone.cpp
    mesh/box_tree.cpp
    boundary/boundary.cpp
    boundary/synthetic_inflow.cpp
    boundary/wall_exchange.cpp
    chemistry/mechanism.cpp
    chemistry/transport.cpp
    chemistry/units.cpp
    numerics/face_reconstruction.cpp
    numerics/overlap.cpp
    numerics/quadrature.cpp
    numerics/scalar_reconstruction.cpp
    numerics/teno.cpp
    numerics/teno_setup.cpp
    numerics/time_integrator.cpp
    numerics/step_control.cpp
    physics/mixture.cpp
    physics/les.cpp
    physics/pasr.cpp
    physics/tfles.cpp
    physics/physics.cpp
    solver/cell_chemistry.cpp
    solver/chemistry_balance.cpp
    solver/chemistry_benchmark.cpp
    solver/numerics_preset.cpp
    solver/solver.cpp
    solver/solver_check.cpp
    solver/solver_chemistry.cpp
    solver/solver_graph.cpp
    solver/solver_step_control.cpp
    solver/solver_initialize.cpp
    solver/solver_mixture.cpp
    solver/solver_les.cpp
    solver/solver_wall_model.cpp
    solver/solver_rhs.cpp
    solver/solver_radiation.cpp
    solver/solver_sponge.cpp
    solver/solver_inflow.cpp
    solver/remap.cpp
    solver/statistics.cpp
    io/compression.cpp
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
target_link_libraries(MallardCore PUBLIC Kokkos::kokkos Threads::Threads)

# Input (common/input.h) records the keys setup reads; a toml::value copy would not
file(GLOB_RECURSE mallard_sources CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.h ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/test/*.h ${CMAKE_SOURCE_DIR}/test/*.cpp)
list(FILTER mallard_sources EXCLUDE REGEX "/src/external/|/src/common/input\\.")
foreach(source ${mallard_sources})
    file(STRINGS ${source} toml_values REGEX "toml::(value|table)([^_a-z]|$)")
    if(toml_values)
        file(READ ${source} content)
        string(REGEX MATCH "toml::(value|table)[^_a-z]" type "${content}")
        string(FIND "${content}" "${type}" offset)
        string(REGEX REPLACE ".$" "" type "${type}")
        string(SUBSTRING "${content}" 0 ${offset} before)
        string(REGEX MATCHALL "\n" newlines "${before}")
        list(LENGTH newlines line)
        math(EXPR line "${line} + 1")
        file(RELATIVE_PATH path ${CMAKE_SOURCE_DIR} ${source})
        message(FATAL_ERROR "${path}:${line} uses ${type}; Mallard reads input only as Input (src/common/input.h) so "
                            "that unused keys are detected: replace toml::value with Input and toml::table with "
                            "Input::table_type.")
    endif()
endforeach()
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

# Mallard new finds the templates in the source tree when run from the build tree
set_source_files_properties(common/case_template.cpp PROPERTIES
    COMPILE_DEFINITIONS "MALLARD_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

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

if(Mallard_ENABLE_ZLIB)
    target_link_libraries(MallardCore PUBLIC ZLIB::ZLIB)
    target_compile_definitions(MallardCore PUBLIC Mallard_HAS_ZLIB)
endif()

if(Mallard_ENABLE_HDF5)
    target_include_directories(MallardCore SYSTEM PUBLIC ${HDF5_C_INCLUDE_DIRS})
    target_compile_definitions(MallardCore PUBLIC Mallard_HAS_HDF5 ${HDF5_C_DEFINITIONS})
    target_link_libraries(MallardCore PUBLIC ${HDF5_C_LIBRARIES})
endif()

if(Mallard_ENABLE_CGNS)
    target_compile_definitions(MallardCore PUBLIC Mallard_HAS_CGNS)
    target_link_libraries(MallardCore PRIVATE ${MALLARD_CGNS_LIBRARY})
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
