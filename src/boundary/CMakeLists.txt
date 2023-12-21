# CMakeLists.txt for BoundaryLib
set(LIB_NAME BoundaryLib)
set(LIB_SRCS
    boundary.cpp
)

# Add the library
add_library(${LIB_NAME} ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)

# Link libraries