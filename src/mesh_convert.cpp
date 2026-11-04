/**
 * @file mesh_convert.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief `mallard mesh convert`: writes a mesh as a Mallard HDF5 mesh file.
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

#include "input.h"

#include "comm.h"
#include "commands.h"
#include "mesh_block.h"

int mesh_convert_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    if (!have_hdf5()) {
        if (comm::is_root()) {
            std::cerr << "mallard mesh convert: this build has no HDF5 support "
                         "(configure with -DMallard_ENABLE_HDF5=ON).\n";
        }
        return 1;
    }
    try {
        const std::string & in = args.positional[0], & out = args.positional[1];
        const MeshBlock block = std::filesystem::path(in).extension() == ".toml" ? read_mesh_block(parse_input(in))
                                                                                 : read_file_block(in);
        write_mesh_h5(out, block);
        const uint64_t n_cells = comm::allreduce(block.n_cells(), comm::Op::SUM);
        const uint64_t n_nodes = comm::allreduce(block.n_nodes(), comm::Op::SUM);
        if (comm::is_root()) std::cout << "Wrote " << out << ": " << n_cells << " cells, " << n_nodes << " nodes\n";
    } catch (const std::exception & e) {
        std::cerr << "mallard mesh convert: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
