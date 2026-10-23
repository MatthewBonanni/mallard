// Mallard HDF5 mesh files. Raw bytes would mostly exercise libhdf5's own
// parser, so the input fills a well-formed file's arrays instead: offsets,
// node and zone ids, coordinates and zone names, consistent or not.
#include <fuzzer/FuzzedDataProvider.h>

#include "fuzz_common.h"
#include "mesh_block.h"

namespace {

std::vector<uint64_t> offsets(FuzzedDataProvider & in, size_t n, size_t n_values) {
    std::vector<uint64_t> o(n + 1, 0);
    const bool valid = in.ConsumeBool();
    for (size_t i = 1; i <= n; i++) {
        o[i] = valid ? std::min<uint64_t>(o[i - 1] + in.ConsumeIntegralInRange<uint64_t>(0, 9), n_values)
                     : in.ConsumeIntegral<uint64_t>();
    }
    if (valid && n > 0) o[n] = n_values;
    return o;
}

std::vector<uint64_t> ids(FuzzedDataProvider & in, size_t n, uint64_t n_valid) {
    std::vector<uint64_t> v(n);
    for (auto & id : v) id = in.ConsumeBool() ? in.ConsumeIntegralInRange<uint64_t>(0, n_valid) : in.ConsumeIntegral<uint64_t>();
    return v;
}

} // namespace

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    fuzz::init_kokkos();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    FuzzedDataProvider in(data, size);
    MeshBlock block;
    block.node_coords.resize(in.ConsumeIntegralInRange<size_t>(0, 64));
    for (auto & x : block.node_coords) {
        for (double & c : x) c = in.ConsumeFloatingPoint<double>();
    }
    const size_t n_cells = in.ConsumeIntegralInRange<size_t>(0, 32);
    block.cell_nodes = ids(in, in.ConsumeIntegralInRange<size_t>(0, 256), block.n_nodes());
    block.cell_offsets = offsets(in, n_cells, block.cell_nodes.size());
    const size_t n_faces = in.ConsumeIntegralInRange<size_t>(0, 32);
    block.face_nodes = ids(in, in.ConsumeIntegralInRange<size_t>(0, 128), block.n_nodes());
    block.face_offsets = offsets(in, n_faces, block.face_nodes.size());
    block.zone_names.resize(in.ConsumeIntegralInRange<size_t>(0, 4));
    for (auto & name : block.zone_names) name = in.ConsumeRandomLengthString(16);
    block.face_zone.resize(n_faces);
    for (auto & z : block.face_zone) z = in.ConsumeIntegralInRange<uint32_t>(0, 4);
    if (in.ConsumeBool()) {
        block.face_high_order_nodes.resize(in.ConsumeIntegralInRange<size_t>(0, 32));
        for (auto & x : block.face_high_order_nodes) {
            for (double & c : x) c = in.ConsumeFloatingPoint<double>();
        }
        block.face_high_order_offsets = offsets(in, n_faces, block.face_high_order_nodes.size());
    }
    const auto path = fuzz::scratch("mesh.h5");
    try {
        write_mesh_h5(path.string(), block);
    } catch (const std::exception &) {
        return -1;
    }
    fuzz::read_mesh(path);
    return 0;
}
