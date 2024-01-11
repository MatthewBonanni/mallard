# CMakeLists.txt for MeshLib
set(LIB_NAME MeshLib)
set(LIB_SRCS
    mesh.cpp
    zone.cpp
)

# Add the library
add_library(${LIB_NAME} ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(${LIB_NAME} PRIVATE tomlplusplus::tomlplusplus)
target_link_libraries(${LIB_NAME} PRIVATE CommonLib)
target_link_libraries(${LIB_NAME} PRIVATE BoundaryLib)
target_link_libraries(${LIB_NAME} PRIVATE PhysicsLib)