/**
 * @file vtkhdf_series.cpp
 * @brief VTKHDF series of write-once snapshot files and their index of virtual datasets.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "vtkhdf_series.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <mutex>
#include <stdexcept>

#include "common_io.h"
#include "hdf5_util.h"
#include "snapshot_block.h"

namespace vtkhdf_series {

#ifdef Mallard_HAS_HDF5

using h5::check;
using h5::Handle;

namespace {

const std::array<const char *, 4> POLY_KINDS = {"Vertices", "Lines", "Polygons", "Strips"};

bool is_value(const std::string & path) {
    const std::string name = path.substr(path.rfind('/') + 1);
    return name.starts_with("NumberOf") || path.starts_with("FieldData/");
}

bool is_mesh(const std::string & path) { return !path.starts_with("CellData/") && !path.starts_with("PointData/"); }

std::pair<uint64_t, uint64_t> dims_of(hid_t dset) {
    Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    hsize_t dims[2] = {0, 0};
    const int rank = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    return {dims[0], rank == 2 ? dims[1] : 0};
}

/** @brief Datasets under group, by path from top, Steps left out. */
void list_datasets(hid_t group, const std::string & prefix, std::vector<std::string> & out) {
    H5G_info_t info;
    check(H5Gget_info(group, &info), "H5Gget_info");
    for (hsize_t i = 0; i < info.nlinks; i++) {
        const ssize_t size = H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        std::string name(static_cast<size_t>(std::max<ssize_t>(size, 0)), '\0');
        H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), name.size() + 1, H5P_DEFAULT);
        if (prefix.empty() && name == "Steps") continue;
        Handle object(H5Oopen(group, name.c_str(), H5P_DEFAULT), H5Oclose, "opening " + prefix + name);
        if (H5Iget_type(object) == H5I_GROUP) {
            list_datasets(object, prefix + name + "/", out);
        } else if (H5Iget_type(object) == H5I_DATASET) {
            out.push_back(prefix + name);
        }
    }
}

/** @brief Where the array at path (under top, in the file named file) is: here, or the file an external link names. */
Source locate(hid_t top, const std::string & file, const std::string & path) {
    Source s{file, "/VTKHDF/" + path, 0, 0, 0, 0};
    H5L_info_t info;
    check(H5Lget_info(top, path.c_str(), &info, H5P_DEFAULT), "H5Lget_info " + path);
    if (info.type == H5L_TYPE_EXTERNAL) {
        std::vector<char> value(info.u.val_size);
        check(H5Lget_val(top, path.c_str(), value.data(), value.size(), H5P_DEFAULT), "H5Lget_val " + path);
        const char * target_file = nullptr;
        const char * target_path = nullptr;
        unsigned flags = 0;
        check(H5Lunpack_elink_val(value.data(), value.size(), &flags, &target_file, &target_path), "H5Lunpack_elink_val");
        s.file = target_file;
        s.path = target_path;
    }
    Handle dset(H5Dopen2(top, path.c_str(), H5P_DEFAULT), H5Dclose, "opening " + path);
    std::tie(s.extent, s.cols) = dims_of(dset);
    s.rows = s.extent;
    return s;
}

double read_value(hid_t top, const std::string & path, uint64_t row) {
    return h5::read_rows<double>(top, path.c_str(), row, 1, 1, H5P_DEFAULT).at(0);
}

std::vector<int64_t> read_column(hid_t group, const std::string & name, uint64_t n, uint32_t cols = 1) {
    return h5::read_rows<int64_t>(group, name.c_str(), 0, n, cols, H5P_DEFAULT);
}

/** @brief The names of an object's attributes. */
std::vector<std::string> attribute_names(hid_t object) {
    std::vector<std::string> names;
    hsize_t at = 0;
    const auto add = [](hid_t, const char * name, const H5A_info_t *, void * out) -> herr_t {
        static_cast<std::vector<std::string> *>(out)->emplace_back(name);
        return 0;
    };
    check(H5Aiterate2(object, H5_INDEX_NAME, H5_ITER_INC, &at, add, &names), "H5Aiterate2");
    return names;
}

/** @brief Copies every attribute of one object to another. */
void copy_attributes(hid_t from, hid_t to) {
    for (const std::string & name : attribute_names(from)) {
        Handle attr(H5Aopen(from, name.c_str(), H5P_DEFAULT), H5Aclose, "opening attribute " + name);
        Handle type(H5Aget_type(attr), H5Tclose, "H5Aget_type");
        Handle memory(H5Tget_native_type(type, H5T_DIR_DEFAULT), H5Tclose, "H5Tget_native_type");
        Handle space(H5Aget_space(attr), H5Sclose, "H5Aget_space");
        const hssize_t n = std::max<hssize_t>(H5Sget_simple_extent_npoints(space), 1);
        std::vector<char> buffer(static_cast<size_t>(n) * H5Tget_size(memory));
        check(H5Aread(attr, memory, buffer.data()), "reading attribute " + name);
        if (H5Aexists(to, name.c_str()) > 0) check(H5Adelete(to, name.c_str()), "H5Adelete");
        Handle copy(H5Acreate2(to, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, "creating " + name);
        check(H5Awrite(copy, memory, buffer.data()), "writing attribute " + name);
        if (H5Tdetect_class(memory, H5T_VLEN) > 0 || H5Tis_variable_str(memory) > 0) {
#if H5_VERSION_GE(1, 12, 0)
            check(H5Treclaim(memory, space, H5P_DEFAULT, buffer.data()), "H5Treclaim");
#else
            check(H5Dvlen_reclaim(memory, space, H5P_DEFAULT, buffer.data()), "H5Dvlen_reclaim");
#endif
        }
    }
}

/** @brief The largest attribute of an object, in bytes as stored. */
size_t largest_attribute(hid_t object) {
    size_t largest = 0;
    for (const std::string & name : attribute_names(object)) {
        Handle attr(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose, "opening attribute " + name);
        largest = std::max<size_t>(largest, H5Aget_storage_size(attr));
    }
    return largest;
}

/** @brief The group at a path under top, created with its parents where missing. */
Handle group_of(hid_t top, const std::string & path) {
    std::string at;
    for (size_t slash = path.find('/'); slash != std::string::npos; slash = path.find('/', slash + 1)) {
        at = path.substr(0, slash);
        if (H5Lexists(top, at.c_str(), H5P_DEFAULT) <= 0) {
            Handle g(H5Gcreate2(top, at.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating " + at);
        }
    }
    return at.empty() ? Handle(H5Gopen2(top, ".", H5P_DEFAULT), H5Gclose, "opening VTKHDF")
                      : Handle(H5Gopen2(top, at.c_str(), H5P_DEFAULT), H5Gclose, "opening " + at);
}

std::string leaf(const std::string & path) { return path.substr(path.rfind('/') + 1); }

/** @brief A dataset of n values (rows of cols) as the source dataset's type, from doubles. */
void write_values(hid_t group, const std::string & name, hid_t type, const std::vector<double> & values, int cols = 1) {
    const int rank = cols > 1 ? 2 : 1;
    const hsize_t dims[2] = {values.size() / static_cast<size_t>(cols), hsize_t(cols)};
    Handle space(H5Screate_simple(rank, dims, nullptr), H5Sclose, "H5Screate_simple");
    Handle dset(H5Dcreate2(group, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose,
                "creating " + name);
    if (!values.empty()) check(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()), name);
}

/** @brief One row per value, as fixed-length strings. */
void write_strings(hid_t parent, const char * name, const std::vector<std::string> & values) {
    size_t width = 1;
    for (const std::string & v : values) width = std::max(width, v.size());
    width = snapshot_block::dataset_width(width);
    std::vector<char> buffer(std::max<size_t>(values.size(), 1) * width, '\0');
    for (size_t i = 0; i < values.size(); i++) std::copy(values[i].begin(), values[i].end(), &buffer[i * width]);
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
    check(H5Tset_size(type, width), "H5Tset_size");
    check(H5Tset_strpad(type, H5T_STR_NULLPAD), "H5Tset_strpad");
    const hsize_t n = values.size();
    Handle space(H5Screate_simple(1, &n, nullptr), H5Sclose, "H5Screate_simple");
    Handle dset(H5Dcreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose, name);
    if (n > 0) check(H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()), name);
}

/** @brief The rows of each step of an array in the index, one step after the other unless they are all the same. */
struct Layout {
    bool shared = true;
    std::vector<uint64_t> first;  // per step, its first row in the index
};

/** @brief A virtual dataset of the steps' sources: shared ones once, the others one after the other. */
Layout write_array(hid_t top, const std::string & path, const std::vector<Step> & steps, bool shared,
                   const std::filesystem::path & dir) {
    Layout layout{shared, std::vector<uint64_t>(steps.size(), 0)};
    struct Mapping {
        Source source;
        uint64_t at;
    };
    std::vector<Mapping> mappings;
    uint64_t total = 0;
    for (size_t s = 0; s < steps.size(); s++) {
        const Source & src = steps[s].arrays.at(path);
        if (shared && s > 0) break;
        layout.first[s] = total;
        if (src.rows > 0) {
            Mapping * last = mappings.empty() ? nullptr : &mappings.back();
            if (last && last->source.file == src.file && last->source.path == src.path &&
                last->source.row + last->source.rows == src.row && last->at + last->source.rows == total) {
                last->source.rows += src.rows;
            } else {
                mappings.push_back({src, total});
            }
        }
        total += src.rows;
    }
    const Source & first = steps.front().arrays.at(path);
    Handle source_file(H5Fopen((dir / first.file).string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose,
                       "opening " + first.file);
    Handle source(H5Dopen2(source_file, first.path.c_str(), H5P_DEFAULT), H5Dclose, "opening " + first.path);
    Handle type(H5Dget_type(source), H5Tclose, "H5Dget_type");
    const int rank = first.cols > 0 ? 2 : 1;
    const hsize_t dims[2] = {total, first.cols};
    Handle space(H5Screate_simple(rank, dims, nullptr), H5Sclose, "H5Screate_simple");
    Handle dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "H5Pcreate");
    for (const Mapping & m : mappings) {
        Handle virtual_space(H5Scopy(space), H5Sclose, "H5Scopy");
        h5::select_rows(virtual_space, m.at, m.source.rows);
        const hsize_t source_dims[2] = {m.source.extent, m.source.cols};
        Handle source_space(H5Screate_simple(rank, source_dims, nullptr), H5Sclose, "H5Screate_simple");
        h5::select_rows(source_space, m.source.row, m.source.rows);
        check(H5Pset_virtual(dcpl, virtual_space, m.source.file.c_str(), m.source.path.c_str(), source_space),
              "H5Pset_virtual " + path);
    }
    Handle group = group_of(top, path);
    Handle dset(H5Dcreate2(group, leaf(path).c_str(), type, space, H5P_DEFAULT, mappings.empty() ? H5P_DEFAULT : hid_t(dcpl),
                           H5P_DEFAULT),
                H5Dclose, "creating " + path);
    copy_attributes(source, dset);
    return layout;
}

} // namespace

std::vector<Step> describe(const std::string & file, size_t max_steps) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    const std::string name = std::filesystem::path(file).filename().string();
    Handle h5file(H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + file);
    Handle top(H5Gopen2(h5file, "VTKHDF", H5P_DEFAULT), H5Gclose, file + ": opening VTKHDF");
    const bool poly = h5::read_strings_attribute(top, "Type").at(0) == "PolyData";
    std::vector<std::string> paths;
    list_datasets(top, "", paths);
    const bool blocks = H5Lexists(h5file, snapshot_block::DATASET, H5P_DEFAULT) > 0;
    const std::vector<std::string> block_rows =
        blocks ? h5::read_string_rows(h5file, snapshot_block::DATASET) : std::vector<std::string>{};
    std::vector<Step> out;
    if (H5Lexists(top, "Steps", H5P_DEFAULT) <= 0) {
        Step s;
        s.file = name;
        for (const std::string & path : paths) {
            if (is_value(path)) {
                s.values[path] = read_value(top, path, 0);
            } else {
                s.arrays[path] = locate(top, name, path);
            }
        }
        s.t = s.values.contains("FieldData/TIME") ? s.values.at("FieldData/TIME") : 0.0;
        s.block = block_rows.empty() ? "" : block_rows[0];
        out.push_back(std::move(s));
        return out;
    }
    // A series in one file: each step's rows from the Steps group
    Handle steps(H5Gopen2(top, "Steps", H5P_DEFAULT), H5Gclose, "opening Steps");
    const uint64_t n = std::min<uint64_t>(h5::n_rows(steps, "Values"), max_steps);
    const std::vector<double> times = h5::read_rows<double>(steps, "Values", 0, n, 1, H5P_DEFAULT);
    const std::vector<int64_t> parts = read_column(steps, "PartOffsets", n);
    const std::vector<int64_t> point_offsets = read_column(steps, "PointOffsets", n);
    const uint32_t kinds = poly ? 4 : 1;
    const std::vector<int64_t> cell_offsets = read_column(steps, "CellOffsets", n, kinds);
    const std::vector<int64_t> id_offsets = read_column(steps, "ConnectivityIdOffsets", n, kinds);
    std::map<std::string, Source> whole;
    for (const std::string & path : paths) {
        if (!is_value(path)) whole[path] = locate(top, name, path);
    }
    const auto offset = [&](const std::string & group, const std::string & array, uint64_t s) {
        return static_cast<uint64_t>(read_value(steps, group + "Offsets/" + array, s));
    };
    for (uint64_t s = 0; s < n; s++) {
        Step step;
        step.file = name;
        step.t = times[s];
        step.block = s < block_rows.size() ? block_rows[s] : "";
        const uint64_t p = static_cast<uint64_t>(parts[s]);
        for (const std::string & path : paths) {
            if (!is_value(path)) continue;
            const bool field = path.starts_with("FieldData/");
            step.values[path] = read_value(top, path, field ? offset("FieldData", leaf(path), s) : p);
        }
        const auto count = [&](const std::string & path) { return static_cast<uint64_t>(step.values.at(path)); };
        const auto range = [&](const std::string & path, uint64_t row, uint64_t rows) {
            Source src = whole.at(path);
            src.row = row;
            src.rows = rows;
            step.arrays[path] = src;
        };
        const uint64_t points = count("NumberOfPoints");
        uint64_t cells = 0;
        for (uint32_t k = 0; k < kinds; k++) {
            const std::string group = poly ? std::string(POLY_KINDS[k]) + "/" : "";
            const uint64_t n_cells = count(group + "NumberOfCells");
            const uint64_t at = static_cast<uint64_t>(cell_offsets[s * kinds + k]);
            cells += n_cells;
            range(group + "Offsets", at + p, n_cells + 1);
            range(group + "Connectivity", static_cast<uint64_t>(id_offsets[s * kinds + k]), count(group + "NumberOfConnectivityIds"));
            if (!poly) range("Types", at, n_cells);
        }
        range("Points", static_cast<uint64_t>(point_offsets[s]), points);
        for (const auto & [path, src] : whole) {
            if (is_mesh(path)) continue;
            const bool cell_data = path.starts_with("CellData/");
            range(path, offset(cell_data ? "CellData" : "PointData", leaf(path), s), cell_data ? cells : points);
        }
        out.push_back(std::move(step));
    }
    return out;
}

void write_index(const std::string & index, const std::vector<Step> & steps) {
    if (steps.empty()) throw std::logic_error("vtkhdf_series: an index of no snapshots");
    const std::lock_guard<std::mutex> lock(h5::mutex());
    const std::filesystem::path dir = std::filesystem::path(index).parent_path();
    const Step & last = steps.back();
    Handle last_file(H5Fopen((dir / last.file).string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose,
                     "opening " + last.file);
    Handle last_root(H5Gopen2(last_file, "/", H5P_DEFAULT), H5Gclose, "opening /");
    Handle last_top(H5Gopen2(last_file, "VTKHDF", H5P_DEFAULT), H5Gclose, "opening VTKHDF");
    const bool poly = h5::read_strings_attribute(last_top, "Type").at(0) == "PolyData";
    const size_t n = steps.size();

    {
        Handle fapl(H5Pcreate(H5P_FILE_ACCESS), H5Pclose, "H5Pcreate");
        Handle fcpl(h5::file_creation(fapl, largest_attribute(last_root)), H5Pclose, "H5Pcreate");
        Handle file(H5Fcreate(partial_name(index).c_str(), H5F_ACC_TRUNC, fcpl, fapl), H5Fclose, "creating " + index);
        {
            Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "opening /");
            copy_attributes(last_root, root);
            h5::write_int_attribute(root, INDEX_ATTRIBUTE, 1);
        }
        Handle top(H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating VTKHDF");
        copy_attributes(last_top, top);
        for (const char * name : {"CellData", "PointData", "FieldData"}) {
            Handle g(H5Gcreate2(top, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, name);
        }

        // A mesh that every step shares is written once, at offsets 0
        bool shared_mesh = true;
        for (const auto & [path, src] : last.arrays) {
            if (!is_mesh(path)) continue;
            for (const Step & s : steps) shared_mesh = shared_mesh && s.arrays.contains(path) && s.arrays.at(path) == src;
        }
        for (const auto & [path, value] : last.values) {
            const bool field = path.starts_with("FieldData/");
            std::vector<double> values;
            for (const Step & s : steps) {
                values.push_back(s.values.contains(path) ? s.values.at(path) : 0.0);
                if (!field && shared_mesh) break;
            }
            Handle source(H5Dopen2(last_top, path.c_str(), H5P_DEFAULT), H5Dclose, "opening " + path);
            Handle type(H5Dget_type(source), H5Tclose, "H5Dget_type");
            Handle group = group_of(top, path);
            write_values(group, leaf(path), type, values);
            Handle dset(H5Dopen2(group, leaf(path).c_str(), H5P_DEFAULT), H5Dclose, "opening " + path);
            copy_attributes(source, dset);
        }
        std::map<std::string, Layout> layouts;
        for (const auto & [path, src] : last.arrays) {
            bool shared = shared_mesh;
            if (!is_mesh(path)) {
                shared = std::all_of(steps.begin(), steps.end(),
                                     [&](const Step & s) { return s.arrays.contains(path) && s.arrays.at(path) == src; });
            }
            for (const Step & s : steps) {
                if (!s.arrays.contains(path)) throw std::runtime_error(s.file + " has no " + path + ", which " + last.file + " has");
            }
            layouts[path] = write_array(top, path, steps, shared, dir);
        }

        Handle group(H5Gcreate2(top, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating Steps");
        h5::write_attribute<int64_t>(group, "NSteps", static_cast<int64_t>(n));
        std::vector<double> times;
        for (const Step & s : steps) times.push_back(s.t);
        h5::write_rows(group, "Values", n, 1, 0, n, times.data(), H5P_DEFAULT);
        if (H5Lexists(last_top, "FieldData", H5P_DEFAULT) > 0 && H5Lexists(last_top, "FieldData/TIME", H5P_DEFAULT) > 0) {
            Handle time(H5Dopen2(last_top, "FieldData/TIME", H5P_DEFAULT), H5Dclose, "opening FieldData/TIME");
            Handle values(H5Dopen2(group, "Values", H5P_DEFAULT), H5Dclose, "opening Steps/Values");
            copy_attributes(time, values);
        }
        const auto column = [&](const char * name, auto row_of, int cols = 1) {
            std::vector<int64_t> values;
            for (size_t s = 0; s < n; s++) {
                for (int k = 0; k < cols; k++) values.push_back(static_cast<int64_t>(row_of(s, k)));
            }
            h5::write_rows(group, name, n, cols, 0, n, values.data(), H5P_DEFAULT);
        };
        const auto first_row = [&](const std::string & path, size_t s) { return layouts.at(path).first[s]; };
        column("NumberOfParts", [](size_t, int) { return 1; });
        column("PartOffsets", [&](size_t s, int) { return shared_mesh ? 0 : s; });
        column("PointOffsets", [&](size_t s, int) { return first_row("Points", s); });
        if (poly) {
            // Each step's offsets of a kind of cells have one more row than its cells
            column("CellOffsets", [&](size_t s, int k) {
                return first_row(std::string(POLY_KINDS[size_t(k)]) + "/Offsets", s) - (shared_mesh ? 0 : s);
            }, 4);
            column("ConnectivityIdOffsets",
                   [&](size_t s, int k) { return first_row(std::string(POLY_KINDS[size_t(k)]) + "/Connectivity", s); }, 4);
        } else {
            column("CellOffsets", [&](size_t s, int) { return first_row("Types", s); });
            column("ConnectivityIdOffsets", [&](size_t s, int) { return first_row("Connectivity", s); });
        }
        for (const char * data : {"CellData", "PointData", "FieldData"}) {
            Handle offsets(H5Gcreate2(group, (std::string(data) + "Offsets").c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                           H5Gclose, data);
            const std::string prefix = std::string(data) + "/";
            for (const auto & [path, layout] : layouts) {
                if (!path.starts_with(prefix)) continue;
                std::vector<int64_t> rows(layout.first.begin(), layout.first.end());
                h5::write_rows(offsets, leaf(path).c_str(), n, 1, 0, n, rows.data(), H5P_DEFAULT);
            }
            if (prefix != "FieldData/") continue;
            for (const auto & [path, value] : last.values) {
                if (!path.starts_with(prefix)) continue;
                std::vector<int64_t> rows(n);
                for (size_t s = 0; s < n; s++) rows[s] = static_cast<int64_t>(s);
                h5::write_rows(offsets, leaf(path).c_str(), n, 1, 0, n, rows.data(), H5P_DEFAULT);
            }
        }
        std::vector<std::string> blocks, files;
        for (const Step & s : steps) {
            blocks.push_back(s.block);
            files.push_back(s.file);
        }
        if (std::any_of(blocks.begin(), blocks.end(), [](const std::string & b) { return !b.empty(); })) {
            write_strings(file, snapshot_block::DATASET, blocks);
        }
        write_strings(file, FILES_DATASET, files);
    }
    publish_file("the VTKHDF series index", index);
}

bool is_index(const std::string & file) {
    const std::lock_guard<std::mutex> lock(h5::mutex());
    Handle h5file(H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + file);
    Handle root(H5Gopen2(h5file, "/", H5P_DEFAULT), H5Gclose, "opening /");
    return H5Aexists(root, INDEX_ATTRIBUTE) > 0;
}

std::vector<std::string> members(const std::string & index) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    {
        const std::lock_guard<std::mutex> lock(h5::mutex());
        Handle h5file(H5Fopen(index.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + index);
        files = h5::read_string_rows(h5file, FILES_DATASET);
    }
    const fs::path path(index), dir = path.parent_path();
    for (std::string & f : files) f = (dir / f).string();
    const fs::path mesh = dir / (path.stem().string() + "_mesh.vtkhdf");
    if (fs::exists(mesh)) files.push_back(mesh.string());
    return files;
}

std::vector<Step> resume(const std::string & index, double t, const std::string & legacy) {
    namespace fs = std::filesystem;
    if (!fs::exists(index)) return {};
    const auto kept = [t](double t_entry) { return t_entry <= t * (1.0 + 1.0e-12) + 1.0e-300; };
    std::vector<Step> out;
    if (!is_index(index)) {
        std::error_code error;
        fs::rename(index, legacy, error);
        if (error) throw std::runtime_error("Could not rename the series file " + index + " to " + legacy + ": " + error.message());
        for (Step & s : describe(legacy)) {
            if (kept(s.t)) out.push_back(std::move(s));
        }
        return out;
    }
    std::vector<std::string> files;
    std::vector<double> times;
    {
        const std::lock_guard<std::mutex> lock(h5::mutex());
        Handle h5file(H5Fopen(index.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + index);
        files = h5::read_string_rows(h5file, FILES_DATASET);
        Handle steps(H5Gopen2(h5file, "VTKHDF/Steps", H5P_DEFAULT), H5Gclose, "opening VTKHDF/Steps");
        times = h5::read_rows<double>(steps, "Values", 0, files.size(), 1, H5P_DEFAULT);
    }
    // A file of several steps (a series of Mallard 0.11 and earlier) is described once
    const fs::path dir = fs::path(index).parent_path();
    std::map<std::string, std::vector<Step>> described;
    std::map<std::string, size_t> taken;
    for (size_t i = 0; i < files.size() && kept(times[i]); i++) {
        auto it = described.find(files[i]);
        if (it == described.end()) it = described.emplace(files[i], describe((dir / files[i]).string())).first;
        out.push_back(it->second.at(taken[files[i]]++));
    }
    return out;
}

#else

std::vector<Step> describe(const std::string &, size_t) { throw std::logic_error("VTKHDF series without HDF5"); }
void write_index(const std::string &, const std::vector<Step> &) { throw std::logic_error("VTKHDF series without HDF5"); }
bool is_index(const std::string &) { return false; }
std::vector<std::string> members(const std::string &) { return {}; }
std::vector<Step> resume(const std::string &, double, const std::string &) { return {}; }

#endif

} // namespace vtkhdf_series
