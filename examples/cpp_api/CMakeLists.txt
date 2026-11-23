# A program that runs a Mallard case through the C++ library API, built against an installed Mallard:
#   cmake -S examples/cpp_api -B build-example -DCMAKE_PREFIX_PATH=<Mallard's install prefix>
cmake_minimum_required(VERSION 3.21)
project(run_case LANGUAGES CXX)

find_package(Mallard REQUIRED COMPONENTS 2d)

add_executable(run_case run_case.cpp)
target_link_libraries(run_case PRIVATE Mallard::mallard2d)
