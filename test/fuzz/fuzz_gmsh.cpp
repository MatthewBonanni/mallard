// Gmsh .msh files (ASCII and binary, versions 2 and 4)
#include "fuzz_common.h"

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    fuzz::init_kokkos();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    // A file starting with '(' is read as Fluent (fuzz_fluent)
    if (size > 0 && data[0] == '(') return -1;
    const auto path = fuzz::scratch("mesh.msh");
    fuzz::write_file(path, data, size);
    fuzz::read_mesh(path);
    return 0;
}
