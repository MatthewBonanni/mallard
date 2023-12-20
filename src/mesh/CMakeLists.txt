# CMakeLists.txt for MeshLib
set(LIB_NAME MeshLib)
set(LIB_SRCS
    mesh.cpp
)

# Add the library
add_library(${LIB_NAME} ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)

# Link libraries
target_link_libraries(${LIB_NAME} PUBLIC CommonLib)