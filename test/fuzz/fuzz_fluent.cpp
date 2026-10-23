// Fluent mesh files (ASCII and binary sections)
#include "fuzz_common.h"

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    fuzz::init_kokkos();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    const auto path = fuzz::scratch("mesh.cas");
    fuzz::write_file(path, data, size);
    fuzz::read_mesh(path);
    return 0;
}
