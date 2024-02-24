# CMakeLists.txt for PhysicsLib
set(LIB_NAME PhysicsLib)
set(LIB_SRCS
    physics.cpp
)

# Add the library
add_library(${LIB_NAME} STATIC ${LIB_SRCS})

# Include directories
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(${LIB_NAME} PRIVATE CommonLib)