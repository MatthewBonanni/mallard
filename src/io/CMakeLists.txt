# CMakeLists.txt for IOLib
set(LIB_NAME IOLib)
set(LIB_SRCS
    data_writer.cpp
)

# Add the library
add_library(${LIB_NAME} STATIC ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(${LIB_NAME} PRIVATE Kokkos::kokkos)
target_link_libraries(${LIB_NAME} PRIVATE CommonLib)
target_link_libraries(${LIB_NAME} PRIVATE SolverLib)
target_link_libraries(${LIB_NAME} PRIVATE MeshLib)