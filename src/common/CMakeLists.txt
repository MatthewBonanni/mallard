# CMakeLists.txt for MeshLib
set(LIB_NAME CommonLib)
set(LIB_SRCS
    common_math.cpp
    common_io.cpp
)

# Add the library
add_library(${LIB_NAME} ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries