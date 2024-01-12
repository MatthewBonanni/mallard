# CMakeLists.txt for NumericsLib
set(LIB_NAME NumericsLib)
set(LIB_SRCS
    time_integrator.cpp
    face_reconstruction.cpp
)

# Add the library
add_library(${LIB_NAME} STATIC ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(${LIB_NAME} PRIVATE CommonLib)
target_link_libraries(${LIB_NAME} PRIVATE MeshLib)