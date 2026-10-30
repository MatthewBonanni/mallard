# CMakeLists.txt for Mallard
add_library(MallardCore STATIC
    common/common_math.cpp
    common/compiled_expression.cpp
    common/region.cpp
    common/tabulated.cpp
    common/common_io.cpp
    common/input.cpp
    common/units.cpp
    common/memory_tracker.cpp
    common/build_info.cpp
    common/case_template.cpp
    common/growing_buffer.cpp
    common/json.cpp
    common/log.cpp
    common/provenance.cpp
    common/run_control.cpp
    common/run_log.cpp
    common/run_status.cpp
    common/input_upgrade.cpp
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
    solver/probe_geometry.cpp
    solver/probes.cpp
    solver/solver.cpp
    solver/solver_check.cpp
    solver/solver_chemistry.cpp
    solver/solver_control.cpp
    solver/solver_graph.cpp
    solver/solver_step_control.cpp
    solver/initial_state.cpp
    solver/solver_initialize.cpp
    solver/solver_mixture.cpp
    solver/solver_les.cpp
    solver/solver_wall_model.cpp
    solver/solver_rhs.cpp
    solver/solver_radiation.cpp
    solver/solver_sponge.cpp
    solver/solver_inflow.cpp
    solver/source_terms.cpp
    solver/remap.cpp
    solver/spectra.cpp
    solver/statistics.cpp
    io/compression.cpp
    io/field_units.cpp
    io/file_format.cpp
    io/data_writer.cpp
    io/data_writer_hdf5.cpp
    io/data_writer_vtkhdf.cpp
    parallel/binding.cpp
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

# Input (common/input.h) records the keys setup reads, and its accessors how it reads
# them (type, default); a toml::value copy, toml::find or toml::find_or would not
file(GLOB_RECURSE mallard_sources CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.h ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/test/*.h ${CMAKE_SOURCE_DIR}/test/*.cpp)
list(FILTER mallard_sources EXCLUDE REGEX "/src/external/|/src/common/input\\.")
foreach(source ${mallard_sources})
    file(STRINGS ${source} toml_values REGEX "toml::(value|table)([^_a-z]|$)|toml::find(_or)?<")
    if(toml_values)
        file(READ ${source} content)
        string(REGEX MATCH "toml::(value|table)[^_a-z]|toml::find(_or)?<" type "${content}")
        string(FIND "${content}" "${type}" offset)
        string(REGEX REPLACE ".$" "" type "${type}")
        string(SUBSTRING "${content}" 0 ${offset} before)
        string(REGEX MATCHALL "\n" newlines "${before}")
        list(LENGTH newlines line)
        math(EXPR line "${line} + 1")
        file(RELATIVE_PATH path ${CMAKE_SOURCE_DIR} ${source})
        message(FATAL_ERROR "${path}:${line} uses ${type}; Mallard reads input only as Input (src/common/input.h) so "
                            "that unused keys are detected: replace toml::value with Input, toml::table with "
                            "Input::table_type, find_value<T> with find_value<T> and find_value_or<T> with "
                            "find_value_or<T>.")
    endif()
endforeach()
# The TENO setup computes every table with the same operations on the host
# and on the device, and contracting multiplies and adds into FMAs would
# change them differently on each: no contraction in that file
if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    set_source_files_properties(numerics/teno_setup.cpp PROPERTIES COMPILE_OPTIONS "--fmad=false;-ffp-contract=off")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # -ffp-contract=off overrides the -fp-model=precise of every file, as intended
    set_source_files_properties(numerics/teno_setup.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-Wno-overriding-option")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(numerics/teno_setup.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_link_libraries(MallardCore PRIVATE yaml-cpp::yaml-cpp)
mallard_target_warnings(MallardCore)

# mallard new and the test suite find their data in the source tree when run from the build tree
set_source_files_properties(common/build_info.cpp PROPERTIES
    COMPILE_DEFINITIONS "MALLARD_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)
set(Mallard_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_type_upper}}")
string(STRIP "${Mallard_CXX_FLAGS}" Mallard_CXX_FLAGS)
string(REPLACE "\\" "\\\\" Mallard_CXX_FLAGS "${Mallard_CXX_FLAGS}")
string(REPLACE "\"" "\\\"" Mallard_CXX_FLAGS "${Mallard_CXX_FLAGS}")
configure_file(common/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h @ONLY)
# The git state for provenance, refreshed at every build rather than at configure time
set(git_version_command ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/generated/git_version.h -P ${CMAKE_SOURCE_DIR}/cmake/git_version.cmake)
execute_process(COMMAND ${git_version_command})
add_custom_target(mallard_git_version COMMAND ${git_version_command}
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/generated/git_version.h)
add_dependencies(MallardCore mallard_git_version)
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

# The commands of the mallard executable (cli.h), which the test suite also runs
add_library(MallardCommands STATIC cli.cpp commands.cpp front_end.cpp run_main.cpp reactor_main.cpp mesh_convert.cpp)
target_link_libraries(MallardCommands PUBLIC MallardCore)
mallard_target_warnings(MallardCommands)

add_executable(mallard main.cpp)
target_link_libraries(mallard PRIVATE MallardCommands)
set_target_properties(mallard PROPERTIES OUTPUT_NAME mallard${Mallard_DIM}d)
mallard_target_warnings(mallard)
# Run as mallard, either solver runs the one of the case's dimension (front_end.h)
add_custom_command(TARGET mallard POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink mallard${Mallard_DIM}d $<TARGET_FILE_DIR:mallard>/mallard)
