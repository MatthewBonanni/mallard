/**
 * @file mesh_convert.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief mallard-mesh-convert: writes a mesh as a Mallard HDF5 mesh file.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include <toml.hpp>

#include "comm.h"
#include "mesh_block.h"

int main(int argc, char * argv[]) {
    comm::Session session(argc, argv);
    if (argc != 3) {
        if (comm::is_root()) {
            std::cerr << "Usage: " << argv[0] << " INPUT OUTPUT.h5\n"
                      << "  INPUT is a Gmsh file (.msh), or a Mallard input file (.toml) whose [mesh]\n"
                      << "  table describes the mesh. Several ranks write in parallel (needs parallel\n"
                      << "  HDF5); each reads the whole Gmsh file but generates only its block.\n";
        }
        return 1;
    }
    try {
        const std::string in = argv[1], out = argv[2];
        const MeshBlock block = std::filesystem::path(in).extension() == ".toml" ? read_mesh_block(toml::parse(in))
                                                                                 : read_gmsh_block(in);
        write_mesh_h5(out, block);
        const uint64_t n_cells = comm::allreduce(block.n_cells(), comm::Op::SUM);
        const uint64_t n_nodes = comm::allreduce(block.n_nodes(), comm::Op::SUM);
        if (comm::is_root()) std::cout << "Wrote " << out << ": " << n_cells << " cells, " << n_nodes << " nodes\n";
    } catch (const std::exception & e) {
        std::cerr << "mallard-mesh-convert: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
