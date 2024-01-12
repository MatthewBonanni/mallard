# CMakeLists.txt for Mallard
add_executable(Mallard
    main.cpp
)

option(${PROJECT_NAME}_CHECK_NAN "Check for NaNs in the solution after each step" OFF)

if(${PROJECT_NAME}_CHECK_NAN)
    add_compile_definitions(${PROJECT_NAME}_CHECK_NAN)
endif()

set(Kokkos_ROOT "${CMAKE_SOURCE_DIR}/build/external/kokkos/install" CACHE PATH "Kokkos install directory")

find_package(Kokkos REQUIRED)

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
add_subdirectory(physics)
add_subdirectory(io)

target_include_directories(Mallard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(Mallard Kokkos::kokkos)
target_link_libraries(Mallard tomlplusplus::tomlplusplus)
target_link_libraries(Mallard CommonLib)
target_link_libraries(Mallard MeshLib)
target_link_libraries(Mallard BoundaryLib)
target_link_libraries(Mallard NumericsLib)
target_link_libraries(Mallard SolverLib)
target_link_libraries(Mallard PhysicsLib)
target_link_libraries(Mallard IOLib)

target_compile_options(Mallard PRIVATE -Wall -Wextra -pedantic)