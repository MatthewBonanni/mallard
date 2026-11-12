// OpenFOAM polyMesh directories. The input is the files points, faces, owner,
// neighbour and boundary in that order, separated by lines "//@@"
#include <algorithm>
#include <array>
#include <string_view>

#include "fuzz_common.h"

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    fuzz::init_kokkos();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    static constexpr std::array<const char *, 5> files = {"points", "faces", "owner", "neighbour", "boundary"};
    static constexpr std::string_view separator = "\n//@@\n";
    const std::string_view text(reinterpret_cast<const char *>(data), size);
    const auto case_dir = fuzz::scratch("case");
    std::filesystem::remove_all(case_dir);
    size_t begin = 0;
    for (const char * name : files) {
        if (begin > text.size()) break;
        const size_t end = std::min(text.find(separator, begin), text.size());
        fuzz::write_file(case_dir / "constant" / "polyMesh" / name, data + begin, end - begin);
        begin = end + separator.size();
    }
    fuzz::read_mesh(case_dir);
    return 0;
}
