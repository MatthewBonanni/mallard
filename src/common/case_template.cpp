/**
 * @file case_template.cpp
 * @brief `mallard new`: ready-to-run case directories from the examples and tutorials.
 *
 * The templates are listed in templates/manifest.toml, which points at the
 * files of an example or tutorial; a case is assembled from those files when
 * it is created, so the templates cannot drift from the examples.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "case_template.h"
#include "input.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <toml.hpp>

#include "build_info.h"

namespace fs = std::filesystem;

namespace case_template {

namespace {

struct Template {
    std::string name;
    std::string group;
    std::string summary;
    std::string cost;
    std::string source;
    std::vector<std::string> files;
    int dim = 2;
    int threads = 4;
    std::vector<std::string> mesh;
    std::string mesh_creates;
    std::vector<std::string> init;
    std::string init_creates;
    std::vector<std::string> post;
    std::string results;
    std::vector<std::string> quick;
    std::vector<std::string> quick_mesh;
    std::vector<std::string> quick_init;
    std::vector<std::string> quick_post;
    std::vector<std::string> packages;
};

struct Group {
    std::string name;
    std::string title;
};

struct Manifest {
    fs::path root;
    std::string input_docs;
    std::string source_url;
    std::vector<std::pair<std::string, std::string>> tables;
    std::vector<Group> groups;
    std::vector<Template> templates;
};

const char * const manifest_path = "templates/manifest.toml";

std::string trim(const std::string & s) {
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r") - b + 1);
}

std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
}

std::string join_lines(const std::vector<std::string> & lines) {
    std::string out;
    for (const auto & l : lines) out += l + "\n";
    return out;
}

std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

void write_file(const fs::path & path, const std::string & text, bool executable = false) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << text;
    out.close();
    if (executable) {
        fs::permissions(path, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                        fs::perm_options::add);
    }
}

/**
 * @brief The table a TOML header line opens ("[a.b]" or "[[a.b]]" -> "a.b"), or "".
 */
std::string header_table(const std::string & line) {
    const std::string t = trim(line);
    if (t.empty() || t[0] != '[') return "";
    const bool array = t.rfind("[[", 0) == 0;
    const size_t open = array ? 2 : 1;
    const size_t close = t.find(array ? "]]" : "]", open);
    if (close == std::string::npos) return "";
    std::string name;
    for (const char c : t.substr(open, close - open)) {
        if (c != ' ' && c != '\t' && c != '"') name += c;
    }
    return name;
}

/**
 * @brief Number of lines a `key = value` starting at lines[i] spans (multi-line strings and arrays).
 */
size_t value_extent(const std::vector<std::string> & lines, size_t i) {
    const std::string value = trim(lines[i].substr(lines[i].find('=') + 1));
    for (const char * quotes : {"\"\"\"", "'''"}) {
        if (value.rfind(quotes, 0) == 0) {
            if (value.find(quotes, 3) != std::string::npos) return 1;
            for (size_t j = i + 1; j < lines.size(); ++j) {
                if (lines[j].find(quotes) != std::string::npos) return j - i + 1;
            }
            return lines.size() - i;
        }
    }
    int depth = 0;
    bool in_string = false;
    for (size_t j = i; j < lines.size(); ++j) {
        const std::string & l = j == i ? value : lines[j];
        for (size_t k = 0; k < l.size(); ++k) {
            const char c = l[k];
            if (in_string) {
                if (c == '\\') ++k;
                else if (c == '"') in_string = false;
            } else if (c == '"') {
                in_string = true;
            } else if (c == '#') {
                break;
            } else if (c == '[' || c == '{') {
                ++depth;
            } else if (c == ']' || c == '}') {
                --depth;
            }
        }
        if (depth <= 0) return j - i + 1;
    }
    return lines.size() - i;
}

bool defines_key(const std::string & line, const std::string & key) {
    const std::string t = trim(line);
    if (t.rfind(key, 0) != 0) return false;
    return trim(t.substr(key.size())).rfind('=', 0) == 0;
}

std::string replace_all(std::string text, const std::string & from, const std::string & to) {
    for (size_t pos = text.find(from); pos != std::string::npos; pos = text.find(from, pos + to.size())) {
        text.replace(pos, from.size(), to);
    }
    return text;
}

std::vector<std::string> string_list(const Input & v, const std::string & key) {
    if (!v.contains(key)) return {};
    return find_value<std::vector<std::string>>(v, key);
}

/**
 * @brief Where the examples, tools and manifest are (data_dirs()).
 */
fs::path data_dir() {
    const std::vector<fs::path> candidates = data_dirs();
    for (const auto & c : candidates) {
        if (fs::exists(c / manifest_path)) return c;
    }
    std::string message = "cannot find the case templates (" + std::string(manifest_path) + "); looked in";
    for (const auto & c : candidates) message += "\n  " + c.string();
    throw std::runtime_error(message + "\nSet MALLARD_DATA_DIR to Mallard's source or share/mallard directory.");
}

Manifest load_manifest() {
    Manifest m;
    m.root = fs::absolute(data_dir());
    const Input v = parse_input((m.root / manifest_path).string());
    m.input_docs = find_value<std::string>(v, "docs", "input");
    m.source_url = find_value<std::string>(v, "docs", "source");
    for (const auto & t : v.at("tables").as_array()) {
        m.tables.emplace_back(find_value<std::string>(t, "table"), find_value<std::string>(t, "about"));
    }
    for (const auto & g : v.at("groups").as_array()) {
        m.groups.push_back({find_value<std::string>(g, "name"), find_value<std::string>(g, "title")});
    }
    for (const auto & e : v.at("templates").as_array()) {
        Template t;
        t.name = find_value<std::string>(e, "name");
        t.group = find_value<std::string>(e, "group");
        t.summary = find_value<std::string>(e, "summary");
        t.cost = find_value<std::string>(e, "cost");
        t.source = find_value<std::string>(e, "source");
        t.files = e.contains("files") ? string_list(e, "files") : std::vector<std::string>{"input.toml"};
        t.dim = find_value_or<int>(e, "dim", 2);
        t.threads = find_value_or<int>(e, "threads", 4);
        t.mesh = string_list(e, "mesh");
        t.mesh_creates = find_value_or<std::string>(e, "mesh_creates", "");
        t.init = string_list(e, "init");
        t.init_creates = find_value_or<std::string>(e, "init_creates", "");
        t.post = string_list(e, "post");
        t.results = find_value_or<std::string>(e, "results", "");
        t.quick = string_list(e, "quick");
        t.quick_mesh = string_list(e, "quick_mesh");
        t.quick_init = string_list(e, "quick_init");
        t.quick_post = string_list(e, "quick_post");
        t.packages = string_list(e, "packages");
        m.templates.push_back(std::move(t));
    }
    return m;
}

size_t edit_distance(const std::string & a, const std::string & b) {
    std::vector<size_t> row(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) row[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        size_t diag = row[0];
        row[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t up = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, diag + (a[i - 1] == b[j - 1] ? 0 : 1)});
            diag = up;
        }
    }
    return row[b.size()];
}

void print_usage(std::ostream & out) {
    out << "Usage: mallard new <template> [directory] [--quick] [--force]\n"
           "       mallard new --list\n"
           "\n"
           "Creates a ready-to-run case from one of Mallard's examples or tutorials:\n"
           "input.toml, run.sh, the mesh or initial-state script it needs, post.sh and\n"
           "a README. The directory defaults to the template's name.\n"
           "\n"
           "  --list, -l   the templates, with what each shows and what it costs\n"
           "  --quick      a coarse, short version that runs in seconds to a minute:\n"
           "               for trying the steps, not for the physical result\n"
           "  --force      write into a directory that is not empty\n";
}

void print_list(const Manifest & m) {
    size_t width = 0;
    for (const auto & t : m.templates) width = std::max(width, t.name.size());
    std::cout << "Case templates: mallard new <template> [directory]\n";
    for (const auto & g : m.groups) {
        std::cout << "\n" << g.title << "\n";
        for (const auto & t : m.templates) {
            if (t.group != g.name) continue;
            std::cout << "  " << t.name << std::string(width + 2 - t.name.size(), ' ')
                      << (t.dim == 3 ? "3D  " : "2D  ") << t.summary << "\n";
        }
    }
    std::cout << "\nThis Mallard is a " << Mallard_DIM << "D build; "
              << (Mallard_DIM == 2 ? "3D templates need one built with -DMallard_DIM=3.\n"
                                   : "2D templates need one built with -DMallard_DIM=2.\n");
}

/**
 * @brief Inserts a one-line description above the first header of each documented table.
 */
std::string annotate(const std::string & text, const Manifest & m) {
    std::set<std::string> done;
    std::vector<std::string> out;
    for (const auto & line : split_lines(text)) {
        const std::string table = header_table(line);
        if (!table.empty() && !done.count(table)) {
            done.insert(table);
            for (const auto & [name, about] : m.tables) {
                if (name != table) continue;
                const std::string anchor = name.substr(0, name.find('.'));
                out.push_back("# [" + name + "]: " + about);
                out.push_back("#   " + m.input_docs + "#" + anchor);
            }
        }
        out.push_back(line);
    }
    return join_lines(out);
}

/**
 * @brief The scripts of tools/ (as paths in it, e.g. post/plot_vtu.py) a list of commands runs.
 */
std::set<std::string> tool_scripts(const std::vector<std::string> & commands) {
    std::set<std::string> scripts;
    const std::regex re(R"(tools/((?:[a-z]+/)?[A-Za-z0-9_]+\.py))");
    for (const auto & c : commands) {
        for (std::sregex_iterator it(c.begin(), c.end(), re), end; it != end; ++it) scripts.insert((*it)[1]);
    }
    return scripts;
}

/**
 * @brief Adds the tools the scripts import (the mallard package, and scripts
 * next to them), and returns the third-party Python packages they need. The
 * packages of a module imported only inside a function (such as mallard.h5,
 * for HDF5 snapshots) are optional.
 */
std::set<std::string> close_imports(const fs::path & tools, const std::vector<fs::path> & extra_scripts,
                                    std::set<std::string> & scripts) {
    static const std::vector<std::pair<std::string, std::string>> known = {
        {"numpy", "numpy"}, {"scipy", "scipy"}, {"matplotlib", "matplotlib"}, {"gmsh", "gmsh"},
        {"cantera", "cantera"}, {"h5py", "h5py"}, {"pyvista", "pyvista"}, {"imageio", "imageio"},
        {"imageio_ffmpeg", "imageio-ffmpeg"}};
    const std::regex re(R"(^(\s*)(?:import|from)\s+([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?(?:\s+import\s+([A-Za-z0-9_]+))?)");
    std::set<std::string> packages;
    std::vector<std::pair<fs::path, bool>> pending;
    for (const auto & s : extra_scripts) pending.emplace_back(s, true);
    for (const auto & s : scripts) pending.emplace_back(tools / s, true);
    std::set<fs::path> required;
    std::set<fs::path> seen;
    while (!pending.empty()) {
        const auto [script, needed] = pending.back();
        pending.pop_back();
        if (needed ? !required.insert(script).second : !seen.insert(script).second) continue;
        seen.insert(script);
        for (const auto & line : split_lines(read_file(script))) {
            std::smatch match;
            if (!std::regex_search(line, match, re)) continue;
            const std::string module = match[2];
            const bool top_level = needed && match[1].length() == 0;
            const fs::path sibling = script.parent_path() / (module + ".py");
            if (module == "mallard") {
                // The whole package (its modules import each other); only those imported here need their packages
                for (const auto & entry : fs::directory_iterator(tools / "mallard")) {
                    if (entry.path().extension() == ".py") scripts.insert("mallard/" + entry.path().filename().string());
                }
                for (const std::string & sub : {match[3].str(), match[4].str()}) {
                    if (sub.empty() || !fs::exists(tools / "mallard" / (sub + ".py"))) continue;
                    pending.emplace_back(tools / "mallard" / (sub + ".py"), top_level);
                }
            } else if (script.parent_path().parent_path() == tools && fs::exists(sibling)) {
                scripts.insert(fs::relative(sibling, tools).generic_string());
                pending.emplace_back(sibling, top_level);
            } else if (needed) {
                for (const auto & [mod, pkg] : known) {
                    if (mod == module) packages.insert(pkg);
                }
            }
        }
    }
    return packages;
}

std::string join(const std::set<std::string> & items, const std::string & sep) {
    std::string out;
    for (const auto & s : items) out += (out.empty() ? "" : sep) + s;
    return out;
}

/**
 * @brief A command of the manifest as a shell line: Python scripts run with $PYTHON.
 */
std::string shell_command(const std::string & command) {
    const std::string first = command.substr(0, command.find(' '));
    const auto is_script = [](const std::string & w) {
        return (w.size() > 3 && w.compare(w.size() - 3, 3, ".py") == 0) ||
               (w.size() > 4 && w.compare(w.size() - 4, 4, ".py\"") == 0);
    };
    if (is_script(first)) return "\"$PYTHON\" " + command;
    return command;
}

std::string python_check(const std::string & script, const std::set<std::string> & packages) {
    if (packages.empty()) return "";
    std::string modules;
    for (const auto & p : packages) modules += (modules.empty() ? "" : ", ") + replace_all(p, "-", "_");
    return "PYTHON=\"${PYTHON:-python3}\"\n"
           "\"$PYTHON\" -c \"import " + modules + "\" 2>/dev/null || {\n"
           "    echo \"" + script + ": needs Python with " + join(packages, " ") + ":\" >&2\n"
           "    echo \"    pip install " + join(packages, " ") + "\" >&2\n"
           "    echo \"or point PYTHON to a Python that has them: PYTHON=/path/to/python ./" + script + "\" >&2\n"
           "    exit 1\n"
           "}\n";
}

std::string step_script(const std::string & name, const std::string & about,
                        const std::vector<std::string> & commands, const std::set<std::string> & packages,
                        const std::string & preamble = "") {
    std::string s = "#!/bin/sh\n# " + about + "\nset -e\ncd \"$(dirname \"$0\")\"\n";
    s += python_check(name, packages);
    if (packages.empty()) s += "PYTHON=\"${PYTHON:-python3}\"\n";
    s += preamble;
    for (const auto & c : commands) {
        if (c.rfind("gmsh ", 0) == 0 && s.find("command -v gmsh") == std::string::npos) {
            s += "command -v gmsh > /dev/null || {\n"
                 "    echo \"" + name + ": needs the gmsh command: pip install gmsh, or https://gmsh.info\" >&2\n"
                 "    exit 1\n"
                 "}\n";
        }
        s += shell_command(c) + "\n";
    }
    return s;
}

std::string run_script(const Template & t, const fs::path & mallard, bool mesh, bool init) {
    const std::string threads = std::to_string(t.threads);
    std::string s =
        "#!/bin/sh\n"
        "# Runs the case.\n"
        "#   ./run.sh           " + threads + " CPU threads (on a GPU build: the GPU)\n"
        "#   ./run.sh -t 8      8 CPU threads\n"
        "#   ./run.sh -n 4      4 MPI ranks (MPI builds; on GPU builds, one GPU per rank)\n"
        "# Another Mallard build (this case needs a " + std::to_string(t.dim) + "D one):\n"
        "#   MALLARD=/path/to/build/src/mallard ./run.sh\n"
        "set -e\n"
        "cd \"$(dirname \"$0\")\"\n"
        "MALLARD=\"${MALLARD:-" + mallard.string() + "}\"\n"
        "threads=\n"
        "ranks=\n"
        "while [ $# -gt 0 ]; do\n"
        "    case \"$1\" in\n"
        "        -t) threads=\"$2\"; shift 2 ;;\n"
        "        -n) ranks=\"$2\"; shift 2 ;;\n"
        "        -h|--help) sed -n '2,7p' \"$0\" | cut -c3-; exit 0 ;;\n"
        "        *) echo \"run.sh: unknown option $1 (./run.sh -h lists them)\" >&2; exit 2 ;;\n"
        "    esac\n"
        "done\n";
    if (mesh) s += "[ -f " + t.mesh_creates + " ] || ./mesh.sh\n";
    if (init) s += "[ -f " + t.init_creates + " ] || ./init.sh\n";
    s += "if [ -n \"$ranks\" ]; then\n"
         "    exec mpirun -n \"$ranks\" \"$MALLARD\" -i input.toml --kokkos-num-threads=\"${threads:-1}\" "
         "--kokkos-map-device-id-by=mpi_rank\n"
         "fi\n"
         "exec \"$MALLARD\" -i input.toml --kokkos-num-threads=\"${threads:-" + threads + "}\"\n";
    return s;
}

std::string readme(const Template & t, const Manifest & m, bool mesh, bool init, bool quick,
                   const std::set<std::string> & packages) {
    std::string r = "# " + t.name + "\n\n" + t.summary + ".\n\n";
    r += "Created by `mallard new " + t.name + (quick ? " --quick" : "") + "` (Mallard " + mallard_version() +
         ") from [`" + t.source + "`](" + m.source_url + t.source + "), " + std::to_string(t.dim) +
         "D build. Size and cost: " + t.cost + ".\n";
    if (quick) {
        r += "\nThis is the quick version: a coarse mesh and a short run, to try the steps. "
             "`mallard new " + t.name + "` gives the full case.\n";
    }
    r += "\n## Run it\n\n```sh\n";
    if (mesh) r += "./mesh.sh     # the mesh, " + t.mesh_creates + " (run.sh calls it if the file is missing)\n";
    if (init) r += "./init.sh     # the initial state, " + t.init_creates + " (run.sh calls it if missing)\n";
    r += "./run.sh      # " + std::to_string(t.threads) + " CPU threads; -t N for N threads, -n N for N MPI ranks or GPUs\n";
    if (!t.post.empty()) r += "./post.sh     # " + t.results + "\n";
    r += "```\n\n";
    r += "`MALLARD=/path/to/mallard ./run.sh` runs another build of Mallard";
    if (!packages.empty()) {
        r += "; the Python scripts need " + join(packages, ", ") + " (`pip install " + join(packages, " ") +
             "`), and `PYTHON=/path/to/python` picks the interpreter";
    }
    r += ".\n\n## Files\n\n";
    r += "- `input.toml`: the case. The comment above each table says what it sets and links to its keys in "
         "the [input reference](" + m.input_docs + ").\n";
    if (mesh) r += "- `mesh.sh`: generates the mesh.\n";
    if (init) r += "- `init.sh`: generates the initial state.\n";
    r += "- `run.sh`: runs Mallard.\n";
    if (!t.post.empty()) r += "- `post.sh`: post-processing.\n";
    r += "- `tools/`: the Python scripts these call, from Mallard's `tools/`.\n";
    r += "\nSnapshots (`.vtu`, `.pvd`) open in ParaView. To change the case, edit `input.toml` and run again.\n";
    return r;
}

std::string input_header(const Template & t, const Manifest & m, bool quick) {
    std::string h =
        "# ----------------------------------------------------------------------------\n"
        "# " + t.name + ": " + t.summary + ".\n"
        "# Created by `mallard new " + t.name + (quick ? " --quick" : "") + "` from " + t.source + " (" +
        std::to_string(t.dim) + "D build).\n";
    if (quick) {
        h += "# Quick version, a coarse mesh and a short run to try the steps:\n";
        for (const auto & q : t.quick) h += "#   " + q + "\n";
    }
    h += "# Run with ./run.sh; every key is in the input reference,\n"
         "#   " + m.input_docs + "\n"
         "# ----------------------------------------------------------------------------\n\n";
    return h;
}

int create(const Manifest & m, const Template & t, const fs::path & dir, bool quick, bool force) {
    if (fs::exists(dir) && !fs::is_directory(dir)) throw std::runtime_error(dir.string() + " is a file");
    if (fs::exists(dir) && !fs::is_empty(dir) && !force) {
        throw std::runtime_error(dir.string() + "/ exists and is not empty (--force writes into it anyway)");
    }
    const fs::path source = m.root / t.source;
    const fs::path tools = m.root / "tools";

    // {data} in a command is the directory of the templates (tutorial scripts run from there)
    const auto expand = [&](const std::vector<std::string> & commands) {
        std::vector<std::string> out;
        for (const auto & c : commands) out.push_back(replace_all(c, "{data}", "\"" + m.root.string() + "\""));
        return out;
    };
    const auto mesh = expand(quick && !t.quick_mesh.empty() ? t.quick_mesh : t.mesh);
    const auto init = expand(quick && !t.quick_init.empty() ? t.quick_init : t.init);
    const auto post = expand(quick && !t.quick_post.empty() ? t.quick_post : t.post);

    std::string input = relocate(read_file(source / t.files.front()));
    if (quick) {
        for (const auto & q : t.quick) {
            const auto eq = q.find('=');
            if (eq == std::string::npos) throw std::runtime_error("manifest: quick entry without '=': " + q);
            input = set_key(input, trim(q.substr(0, eq)), trim(q.substr(eq + 1)));
        }
    }

    fs::create_directories(dir);
    write_file(dir / "input.toml", input_header(t, m, quick) + annotate(input, m));
    for (size_t i = 1; i < t.files.size(); ++i) {
        const fs::path target = dir / t.files[i];
        fs::create_directories(target.parent_path());
        if (target.extension() == ".py") {
            // The example's own scripts find tools/ next to them in the case
            write_file(target, relocate(read_file(source / t.files[i])));
        } else {
            fs::copy_file(source / t.files[i], target, fs::copy_options::overwrite_existing);
        }
    }
    // The example's scripts a list of commands runs
    const auto local_scripts = [&](const std::vector<std::string> & commands) {
        std::vector<fs::path> scripts;
        for (const auto & f : t.files) {
            if (fs::path(f).extension() != ".py") continue;
            for (const auto & c : commands) {
                if (c.rfind(f + " ", 0) == 0 || c == f) {
                    scripts.push_back(source / f);
                    break;
                }
            }
        }
        return scripts;
    };

    std::set<std::string> mechanisms;
    const std::regex mech_re(R"(mechanisms/([A-Za-z0-9_.-]+\.yaml))");
    for (std::sregex_iterator it(input.begin(), input.end(), mech_re), end; it != end; ++it) {
        mechanisms.insert((*it)[1]);
    }
    if (!mechanisms.empty()) fs::create_directories(dir / "mechanisms");
    for (const auto & mech : mechanisms) {
        fs::copy_file(m.root / "mechanisms" / mech, dir / "mechanisms" / mech, fs::copy_options::overwrite_existing);
    }

    std::set<std::string> mesh_tools = tool_scripts(mesh);
    std::set<std::string> init_tools = tool_scripts(init);
    std::set<std::string> post_tools = tool_scripts(post);
    auto mesh_packages = close_imports(tools, local_scripts(mesh), mesh_tools);
    auto init_packages = close_imports(tools, local_scripts(init), init_tools);
    auto post_packages = close_imports(tools, local_scripts(post), post_tools);
    for (const auto & p : t.packages) {
        if (!post.empty()) post_packages.insert(p);
        if (!init.empty()) init_packages.insert(p);
    }
    std::set<std::string> all_tools(mesh_tools);
    all_tools.insert(init_tools.begin(), init_tools.end());
    all_tools.insert(post_tools.begin(), post_tools.end());
    if (!all_tools.empty()) fs::create_directories(dir / "tools");
    for (const auto & s : all_tools) {
        fs::create_directories((dir / "tools" / s).parent_path());
        fs::copy_file(tools / s, dir / "tools" / s, fs::copy_options::overwrite_existing);
    }
    std::set<std::string> packages(mesh_packages);
    packages.insert(init_packages.begin(), init_packages.end());
    packages.insert(post_packages.begin(), post_packages.end());

    std::vector<std::pair<std::string, std::string>> written = {
        {"input.toml", "the case; a comment above each table links to its keys"}};
    for (size_t i = 1; i < t.files.size(); ++i) {
        if (fs::path(t.files[i]).extension() == ".py") written.emplace_back(t.files[i], "the example's script");
    }
    if (!mesh.empty()) {
        write_file(dir / "mesh.sh", step_script("mesh.sh", "Generates the mesh, " + t.mesh_creates + ".", mesh,
                                                mesh_packages), true);
        written.emplace_back("mesh.sh", "generates the mesh, " + t.mesh_creates);
    }
    if (!init.empty()) {
        write_file(dir / "init.sh", step_script("init.sh", "Generates the initial state, " + t.init_creates + ".",
                                                init, init_packages), true);
        written.emplace_back("init.sh", "generates the initial state, " + t.init_creates);
    }
    write_file(dir / "run.sh", run_script(t, executable_path(), !mesh.empty(), !init.empty()), true);
    written.emplace_back("run.sh", "runs it: ./run.sh [-t threads] [-n MPI ranks]");
    if (!post.empty()) {
        const std::string last =
            "# The newest snapshot PREFIX_N: .vtu, or .pvtu of an MPI run (not its pieces)\n"
            "last() {\n"
            "    newest=$(ls -t \"$1\"_[0-9]*.vtu \"$1\"_[0-9]*.pvtu 2>/dev/null | grep -v '_p[0-9]*\\.vtu$' | head -n 1)\n"
            "    [ -n \"$newest\" ] || { echo \"post.sh: no snapshots $1_*: run ./run.sh first\" >&2; exit 1; }\n"
            "    ls \"$1\"_[0-9]*.\"${newest##*.}\" | grep -v '_p[0-9]*\\.vtu$' | tail -n 1\n"
            "}\n";
        write_file(dir / "post.sh", step_script("post.sh", "Post-processing: " + t.results + ".", post,
                                                post_packages, last), true);
        written.emplace_back("post.sh", t.results);
    }
    write_file(dir / "README.md", readme(t, m, !mesh.empty(), !init.empty(), quick, packages));
    written.emplace_back("README.md", "what the case shows, how to run it");
    if (!all_tools.empty()) written.emplace_back("tools/", join(all_tools, " "));
    if (!mechanisms.empty()) written.emplace_back("mechanisms/", join(mechanisms, " "));

    std::cout << "Created " << dir.string() << "/ from the " << t.name << " template (" << t.source
              << (quick ? ", quick version" : "") << "):\n";
    for (const auto & [file, about] : written) {
        std::cout << "  " << file << std::string(file.size() < 12 ? 12 - file.size() : 1, ' ') << about << "\n";
    }
    if (t.dim != Mallard_DIM) {
        std::cout << "\nNote: this case needs a " << t.dim << "D build of Mallard (cmake -DMallard_DIM=" << t.dim
                  << "); this one is " << Mallard_DIM << "D. Run it with\n"
                  << "  MALLARD=/path/to/build" << t.dim << "d/src/mallard ./run.sh\n";
    }
    std::cout << "\nNext:\n  cd " << dir.string() << "\n";
    if (!mesh.empty()) std::cout << "  ./mesh.sh\n";
    if (!init.empty()) std::cout << "  ./init.sh\n";
    std::cout << "  ./run.sh      # " << (quick ? "quick version" : t.cost) << "\n";
    if (!post.empty()) std::cout << "  ./post.sh\n";
    if (!packages.empty()) std::cout << "(the Python scripts need: pip install " << join(packages, " ") << ")\n";
    return 0;
}

} // namespace

std::string set_key(const std::string & text, const std::string & path, const std::string & value) {
    const auto dot = path.rfind('.');
    if (dot == std::string::npos) throw std::runtime_error("set_key: " + path + " names no table");
    const std::string table = path.substr(0, dot);
    const std::string key = path.substr(dot + 1);
    std::vector<std::string> lines = split_lines(text);
    // The sections of the table, as [header, end) line ranges, and whether any defines the key
    std::vector<std::pair<size_t, size_t>> sections;
    bool defined = false;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (header_table(lines[i]) != table) continue;
        size_t end = i + 1;
        while (end < lines.size() && header_table(lines[end]).empty()) ++end;
        sections.emplace_back(i, end);
        for (size_t j = i + 1; j < end; ++j) defined = defined || defines_key(lines[j], key);
    }
    const bool found_table = !sections.empty();
    // Back to front, so that edits keep the earlier ranges valid
    for (auto s = sections.rbegin(); s != sections.rend(); ++s) {
        const auto [i, end] = *s;
        size_t j = i + 1;
        while (j < end && !defines_key(lines[j], key)) ++j;
        if (j < end) {
            const size_t n = value_extent(lines, j);
            const std::string indent = lines[j].substr(0, lines[j].find_first_not_of(" \t"));
            lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(j),
                        lines.begin() + static_cast<std::ptrdiff_t>(j + n));
            lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(j), indent + key + " = " + value);
        } else if (!defined) {
            lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(i + 1), key + " = " + value);
        }
    }
    if (!found_table) throw std::runtime_error("set_key: no [" + table + "] table for " + path);
    return join_lines(lines);
}

std::string relocate(const std::string & text) {
    std::string out = text;
    for (const char * build : {"../../build3d/src/mallard", "../../build-3d/src/mallard", "../../build/src/mallard"}) {
        out = replace_all(out, build, "mallard");
    }
    out = replace_all(out, "../../tools", "tools");
    out = replace_all(out, "../../mechanisms/", "mechanisms/");
    return out;
}

int new_main(int argc, char ** argv) {
    std::vector<std::string> positional;
    bool list = false;
    bool quick = false;
    bool force = false;
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list" || arg == "-l") {
            list = true;
        } else if (arg == "--quick") {
            quick = true;
        } else if (arg == "--force") {
            force = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(std::cout);
            return 0;
        } else if (arg.rfind('-', 0) == 0) {
            std::cerr << "mallard new: unknown option " << arg << "\n\n";
            print_usage(std::cerr);
            return 2;
        } else {
            positional.push_back(arg);
        }
    }
    try {
        if (list || positional.empty()) {
            const Manifest m = load_manifest();
            print_list(m);
            if (!list) std::cout << "\nCreate one with: mallard new <template> [directory]\n";
            return list ? 0 : 2;
        }
        if (positional.size() > 2) {
            std::cerr << "mallard new: too many arguments\n\n";
            print_usage(std::cerr);
            return 2;
        }
        const Manifest m = load_manifest();
        const std::string & name = positional[0];
        for (const auto & t : m.templates) {
            if (t.name == name) return create(m, t, positional.size() > 1 ? positional[1] : name, quick, force);
        }
        std::cerr << "mallard new: no template named " << name << ".";
        const auto best = std::min_element(m.templates.begin(), m.templates.end(), [&](const auto & a, const auto & b) {
            return edit_distance(a.name, name) < edit_distance(b.name, name);
        });
        if (best != m.templates.end() && edit_distance(best->name, name) <= 3) {
            std::cerr << " Did you mean " << best->name << "?";
        }
        std::cerr << "\n`mallard new --list` shows them all.\n";
        return 2;
    } catch (const std::exception & e) {
        std::cerr << "mallard new: " << e.what() << "\n";
        return 1;
    }
}

} // namespace case_template
