/**
 * @file front_end.cpp
 * @brief `mallard`: runs mallard2d or mallard3d, whichever a case needs.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "front_end.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "build_info.h"
#include "case_template.h"
#include "cli.h"
#include "input.h"
#include "mesh/mesh_block.h"
#include "mesh/mesh_file.h"

namespace fs = std::filesystem;

namespace front_end {

namespace {

Dimension unknown(const std::string & why) { return {0, why}; }
Dimension unreadable(const std::string & why) { return {0, why, true}; }

Dimension openfoam_dimension(const fs::path & path) {
    for (const fs::path & dir : {path, path / "polyMesh", path / "constant" / "polyMesh"}) {
        std::ifstream in(dir / "boundary");
        if (!in.good()) continue;
        std::stringstream text;
        text << in.rdbuf();
        // A 2D mesh is one cell thick between empty patches (see read_openfoam)
        std::istringstream tokens(text.str());
        std::string previous, token;
        while (tokens >> token) {
            if (previous == "type" && (token == "empty;" || token == "empty")) return {2, ""};
            previous = token;
        }
        return {3, ""};
    }
    return unknown("no OpenFOAM polyMesh boundary file in " + path.string() + ", its polyMesh/ or its constant/polyMesh/");
}

/** @brief The dimension section, (2 2) or (2 3), of a Fluent file's text header. */
Dimension fluent_dimension(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    std::string head(1 << 20, '\0');
    in.read(head.data(), static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<size_t>(in.gcount()));
    for (size_t pos = head.find("(2"); pos != std::string::npos; pos = head.find("(2", pos + 2)) {
        size_t start = pos;
        while (start > 0 && (head[start - 1] == ' ' || head[start - 1] == '\t' || head[start - 1] == '\r')) start--;
        if (start > 0 && head[start - 1] != '\n') continue;
        size_t i = pos + 2;
        if (i >= head.size() || (head[i] != ' ' && head[i] != '\t')) continue;
        while (i < head.size() && (head[i] == ' ' || head[i] == '\t')) i++;
        if (i + 1 >= head.size() || (head[i] != '2' && head[i] != '3')) continue;
        const int dim = head[i] - '0';
        i++;
        while (i < head.size() && (head[i] == ' ' || head[i] == '\t')) i++;
        if (i < head.size() && head[i] == ')') return {dim, ""};
    }
    return unknown("Fluent file " + path + " has no dimension section");
}

/** @brief Gmsh element types of dimension 3 (see gmsh_element_type in mesh_io.cpp). */
bool gmsh_volume_element(int type) {
    static const std::set<int> volume = {4, 5, 6, 7, 11, 12, 13, 14, 17, 18, 19, 29, 90, 92};
    return volume.count(type) > 0;
}

/**
 * @brief A Gmsh file's dimension: 4.x from the volume count of $Entities,
 *        2.2 from its elements (3D at the first volume element).
 */
Dimension gmsh_dimension(const std::string & path) {
    std::ifstream in(path);
    std::string line;
    double version = 0.0;
    auto next = [&]() {
        if (!std::getline(in, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    };
    while (next()) {
        if (line == "$MeshFormat" && next()) {
            std::istringstream(line) >> version;
        } else if (line == "$Entities" && next()) {
            size_t points = 0, curves = 0, surfaces = 0, volumes = 0;
            if (std::istringstream(line) >> points >> curves >> surfaces >> volumes) return {volumes > 0 ? 3 : 2, ""};
        } else if (line == "$Elements" && next()) {
            std::istringstream header(line);
            size_t n = 0;
            header >> n;
            if (version >= 4.0) {
                for (size_t b = 0; b < n && next(); b++) {
                    int entity_dim = 0, tag = 0, type = 0;
                    size_t in_block = 0;
                    std::istringstream(line) >> entity_dim >> tag >> type >> in_block;
                    if (entity_dim == 3) return {3, ""};
                    for (size_t e = 0; e < in_block && next(); e++) {}
                }
            } else {
                for (size_t e = 0; e < n && next(); e++) {
                    size_t tag = 0;
                    int type = 0;
                    std::istringstream(line) >> tag >> type;
                    if (gmsh_volume_element(type)) return {3, ""};
                }
            }
            return n > 0 ? Dimension{2, ""} : unknown("Gmsh file " + path + " has no elements");
        }
    }
    return unknown("Gmsh file " + path + " has no $Elements section");
}

/** @brief Mallard's own dimension-specific solver of dimension dim, in directory bin. */
fs::path solver(const fs::path & bin, int dim) { return bin / ("mallard" + std::to_string(dim) + "d"); }

/** @brief The test suite of dimension dim of a build (bin/../test) or an install (bin) tree, else empty. */
fs::path test_suite(const fs::path & bin, int dim) {
    const std::string name = "mallard-test" + std::to_string(dim) + "d";
    for (const fs::path & path : {bin / name, bin.parent_path() / "test" / name}) {
        if (fs::exists(path)) return path;
    }
    return {};
}

std::vector<char *> c_args(std::string & program, std::vector<std::string> & args) {
    std::vector<char *> out = {program.data()};
    for (auto & a : args) out.push_back(a.data());
    out.push_back(nullptr);
    return out;
}

int exec(const fs::path & program, std::vector<std::string> & args) {
    std::string path = program.string();
    std::vector<char *> argv = c_args(path, args);
    execv(path.c_str(), argv.data());
    std::cerr << "mallard: cannot run " << path << ": " << std::strerror(errno) << "\n";
    return 1;
}

/** @brief `mallard test`: the test suites installed, or that of --dim; several one after another. */
int run_tests(const fs::path & bin, int dim, std::vector<std::string> & args) {
    std::vector<fs::path> suites;
    for (const int d : {2, 3}) {
        if (dim != 0 && d != dim) continue;
        const fs::path suite = test_suite(bin, d);
        if (!suite.empty()) suites.push_back(suite);
    }
    if (suites.empty()) {
        std::cerr << "mallard test: no "
                  << (dim ? "mallard-test" + std::to_string(dim) + "d" : "mallard-test2d or mallard-test3d") << " in "
                  << bin.string() << " or " << (bin.parent_path() / "test").string() << ".\n"
                  << "Build it with -DMallard_ENABLE_TESTS=ON" << (dim ? " -DMallard_DIM=" + std::to_string(dim) : "")
                  << " and install it with -DMallard_INSTALL_TESTS=ON (both the default).\n";
        return 1;
    }
    if (suites.size() == 1) return exec(suites[0], args);
    int status = 0;
    for (const fs::path & suite : suites) {
        std::cout << "mallard test: " << suite.string() << std::endl;
        std::string path = suite.string();
        std::vector<char *> argv = c_args(path, args);
        const pid_t pid = fork();
        if (pid == 0) {
            execv(path.c_str(), argv.data());
            std::cerr << "mallard test: cannot run " << path << ": " << std::strerror(errno) << "\n";
            _exit(127);
        }
        int wstatus = 0;
        if (pid < 0 || waitpid(pid, &wstatus, 0) < 0 || !WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) status = 1;
    }
    return status;
}

/** @brief What a command runs on: a dimension, and the case it is of; dimension_free for commands without one. */
struct Need {
    Dimension dimension;
    std::string what;
    bool dimension_free = false;
};

Need command_need(const cli::Command * command, const std::vector<std::string> & args) {
    const Need free = {{}, "", true};
    cli::Args parsed;
    std::string error;
    if (!command || !cli::parse(*command, args, parsed, error) || parsed.help) return free;
    const std::string first = parsed.positional.empty() ? "" : parsed.positional[0];
    switch (command->dim) {
        case cli::Case::input: {
            const std::string file = parsed.has("-i") ? parsed.value_or("-i", "") : first;
            if (file.empty()) return free;
            return {input_dimension(file), file};
        }
        case cli::Case::template_name: {
            if (first.empty() || parsed.has("--list")) return free;
            try {
                const int dim = case_template::template_dimension(first);
                return dim == 0 ? free : Need{{dim, ""}, "template " + first};
            } catch (const std::exception & e) {
                return {unknown(e.what()), "template " + first};
            }
        }
        case cli::Case::mesh:
            if (first.empty()) return free;
            return {fs::path(first).extension() == ".toml" ? input_dimension(first) : mesh_file_dimension(first), first};
        default:
            return free;
    }
}

} // namespace

Dimension mesh_file_dimension(const std::string & path) {
    try {
        if (fs::is_directory(path)) return openfoam_dimension(path);
        if (!fs::exists(path)) return unreadable("Could not open mesh file: " + path + ".");
        if (is_hdf5_mesh(path)) return {mesh_h5_dimension(path), ""};
        const std::string ext = fs::path(path).extension().string();
        if (ext == ".cgns") return {cgns_dimension(path), ""};
        if (ext == ".cas") return fluent_dimension(path);
        std::ifstream in(path);
        char first = 0;
        in >> first;
        return first == '(' ? fluent_dimension(path) : gmsh_dimension(path);
    } catch (const std::exception & e) {
        return unreadable(e.what());
    }
}

Dimension input_dimension(const std::string & input_file) {
    try {
        const Input input = parse_input(input_file);
        const std::string type = find_value_or<std::string>(input, "mesh", "type", "file");
        if (type == "file") {
            const fs::path mesh = find_value_or<std::string>(input, "mesh", "filename", "mesh.msh");
            if (!fs::exists(mesh)) {
                std::error_code ec;
                const fs::path cwd = fs::current_path(ec);
                return unreadable("mesh file '" + mesh.string() + "' not found" +
                                  (mesh.is_relative() && !ec ? " relative to the working directory (" + cwd.string() + ")"
                                                             : std::string()));
            }
            return mesh_file_dimension(mesh.string());
        }
        if (type == "cartesian_tri" || type == "wedge") return {2, ""};
        if (type == "cartesian_tet" || type == "cartesian_prism" || type == "cartesian_pyramid" ||
            type == "cartesian_mixed") {
            return {3, ""};
        }
        const bool mesh = input.contains("mesh") && input.at("mesh").is_table();
        if (mesh && (input.at("mesh").contains("Nz") || input.at("mesh").contains("Lz"))) return {3, ""};
        for (const auto & dir : find_value_or<std::vector<std::string>>(input, "mesh", "periodic", {})) {
            if (dir == "z") return {3, ""};
        }
        return {2, ""};
    } catch (const std::exception & e) {
        return unreadable(e.what());
    }
}

bool invoked(const char * argv0) { return fs::path(argv0).filename() == "mallard"; }

int test_main(const cli::Args & args) {
    const std::string value = args.value_or("--dim", "0");
    if (value != "0" && value != "2" && value != "3") {
        return cli::usage_error(*cli::command("test"), "--dim takes 2 or 3, not " + value);
    }
    const int dim = value[0] - '0';
    const fs::path bin = executable_path().parent_path();
    std::vector<std::string> passthrough = args.passthrough;
    return run_tests(bin, dim, passthrough);
}

int main(int argc, char ** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    std::vector<std::string> dim_option;
    int dim = 0;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] != "--dim") continue;
        const std::string value = i + 1 < args.size() ? args[i + 1] : "";
        if (value != "2" && value != "3") {
            std::cerr << "mallard: --dim takes 2 or 3 (see mallard --help)\n";
            return 2;
        }
        dim = value[0] - '0';
        dim_option = {args[i], value};
        args.erase(args.begin() + std::ptrdiff_t(i), args.begin() + std::ptrdiff_t(i) + 2);
        break;
    }
    size_t words = 0;
    const cli::Command * command = cli::find(args, words);
    if (command && command->dim == cli::Case::front_end) {
        args.insert(args.begin() + std::ptrdiff_t(words), dim_option.begin(), dim_option.end());
        std::string program = argv[0];
        std::vector<char *> command_argv = c_args(program, args);
        return cli::main(static_cast<int>(command_argv.size()) - 1, command_argv.data());
    }
    const fs::path bin = executable_path().parent_path();

    std::vector<int> installed;
    for (const int d : {2, 3}) {
        if (fs::exists(solver(bin, d))) installed.push_back(d);
    }
    if (installed.empty()) {
        std::cerr << "mallard: neither mallard2d nor mallard3d is in " << bin.string() << ".\n";
        return 1;
    }
    const bool stated = dim != 0;
    const Need need = stated ? Need{{dim, ""}, "", false}
                             : command_need(command, {args.begin() + std::ptrdiff_t(words), args.end()});
    dim = need.dimension.dim;
    if (dim == 0) {
        if (!need.dimension_free && !need.dimension.unreadable && installed.size() > 1) {
            std::cerr << "mallard: cannot tell whether " << need.what << " is 2D or 3D: " << need.dimension.why
                      << "\nState it with --dim 2 or --dim 3, e.g. mallard --dim 3 -i input.toml.\n";
            return 1;
        }
        dim = installed[0];
    }
    if (!fs::exists(solver(bin, dim))) {
        std::cerr << "mallard: " << (stated ? "--dim " + std::to_string(dim) : need.what + " is " + std::to_string(dim) + "D and")
                  << " needs mallard" << dim << "d, which is not in " << bin.string() << ".\nBuild it with -DMallard_DIM="
                  << dim << " and install it to the same prefix as mallard" << installed[0] << "d (cmake --install).\n";
        return 1;
    }
    return exec(solver(bin, dim), args);
}

} // namespace front_end
