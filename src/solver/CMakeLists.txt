# CMakeLists.txt for SolverLib
set(LIB_NAME SolverLib)
set(LIB_SRCS
    solver.cpp
    solver_initialize.cpp
    solver_rhs.cpp
)

# Add the library
add_library(${LIB_NAME} STATIC ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(${LIB_NAME} PRIVATE tomlplusplus::tomlplusplus)
target_link_libraries(${LIB_NAME} PRIVATE CommonLib)
target_link_libraries(${LIB_NAME} PRIVATE MeshLib)
target_link_libraries(${LIB_NAME} PRIVATE BoundaryLib)
target_link_libraries(${LIB_NAME} PRIVATE NumericsLib)
target_link_libraries(${LIB_NAME} PRIVATE PhysicsLib)
target_link_libraries(${LIB_NAME} PRIVATE IOLib)