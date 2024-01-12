# CMakeLists.txt for BoundaryLib
set(LIB_NAME BoundaryLib)
set(LIB_SRCS
    boundary.cpp
    boundary_symmetry.cpp
    boundary_wall_adiabatic.cpp
    boundary_upt.cpp
    boundary_p_out.cpp
)

# Add the library
add_library(${LIB_NAME} STATIC ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(${LIB_NAME} PRIVATE tomlplusplus::tomlplusplus)
target_link_libraries(${LIB_NAME} PRIVATE CommonLib)
target_link_libraries(${LIB_NAME} PRIVATE SolverLib)
target_link_libraries(${LIB_NAME} PRIVATE MeshLib)
target_link_libraries(${LIB_NAME} PRIVATE PhysicsLib)