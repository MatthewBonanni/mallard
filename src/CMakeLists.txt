cmake_minimum_required(VERSION 3.12)

add_executable(Mallard
    main.cpp
)

set_target_properties(Mallard PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

add_subdirectory(common)
add_subdirectory(mesh)
add_subdirectory(boundary)
add_subdirectory(numerics)
add_subdirectory(solver)

target_include_directories(Mallard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_include_directories(Mallard PUBLIC ${tomlplusplus_SOURCE_DIR}/include)

target_link_libraries(Mallard tomlplusplus::tomlplusplus)
target_link_libraries(Mallard CommonLib)
target_link_libraries(Mallard MeshLib)
target_link_libraries(Mallard BoundaryLib)
target_link_libraries(Mallard NumericsLib)
target_link_libraries(Mallard SolverLib)

target_compile_options(Mallard PRIVATE -Wall -Wextra -pedantic)