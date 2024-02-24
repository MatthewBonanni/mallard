# CMakeLists.txt for Mallard
add_executable(Mallard
    main.cpp
)

add_subdirectory(common)
add_subdirectory(mesh)
add_subdirectory(boundary)
add_subdirectory(numerics)
add_subdirectory(solver)
add_subdirectory(physics)
add_subdirectory(io)

target_link_libraries(Mallard Kokkos::kokkos)
target_link_libraries(Mallard Kokkos::kokkoskernels)
target_link_libraries(Mallard CommonLib)
target_link_libraries(Mallard MeshLib)
target_link_libraries(Mallard BoundaryLib)
target_link_libraries(Mallard NumericsLib)
target_link_libraries(Mallard SolverLib)
target_link_libraries(Mallard PhysicsLib)
target_link_libraries(Mallard IOLib)