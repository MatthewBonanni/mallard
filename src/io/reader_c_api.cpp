/**
 * @file reader_c_api.cpp
 * @brief The output reader as a C library: no C++ exception crosses it.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "reader_c_api.h"

#include <exception>
#include <string>
#include <vector>

#include "build_info.h"
#include "common_typedef.h"
#include "json.h"
#include "output_reader.h"

struct mallard_file {
    output::Contents contents;
    std::string json;
};

struct mallard_array {
    std::vector<double> values;
    output::Geometry geometry;
    std::string text;
};

namespace {

thread_local std::string last_error;

template <typename F>
auto guarded(F && f) -> decltype(f()) {
    try {
        return f();
    } catch (const std::exception & e) {
        last_error = e.what();
    } catch (...) {
        last_error = "an unknown error";
    }
    return nullptr;
}

} // namespace

extern "C" {

const char * mallard_error(void) { return last_error.c_str(); }

const char * mallard_library_version(void) {
    static const std::string version = mallard_version();
    return version.c_str();
}

int mallard_dim(void) { return N_DIM; }

mallard_file * mallard_open(const char * path) {
    return guarded([&] {
        auto * f = new mallard_file{output::open(path), ""};
        try {
            f->json = output::to_json(f->contents);
        } catch (...) {
            delete f;
            throw;
        }
        return f;
    });
}

const char * mallard_contents(const mallard_file * file) { return file->json.c_str(); }

void mallard_close(mallard_file * file) { delete file; }

mallard_array * mallard_read(const mallard_file * file, const char * field, uint64_t snapshot) {
    return guarded([&] { return new mallard_array{output::read(file->contents, field, snapshot), {}, ""}; });
}

mallard_array * mallard_read_all(const mallard_file * file, const char * field) {
    return guarded([&] { return new mallard_array{output::read_all(file->contents, field), {}, ""}; });
}

mallard_array * mallard_read_geometry(const mallard_file * file, uint64_t snapshot) {
    return guarded([&] { return new mallard_array{{}, output::read_geometry(file->contents, snapshot), ""}; });
}

mallard_array * mallard_run_files(const char * dir) {
    return guarded([&] {
        std::string text = "[";
        for (const auto & f : output::run_files(dir)) text += (text.size() > 1 ? "," : "") + json::quote(f);
        return new mallard_array{{}, {}, text + "]"};
    });
}

const double * mallard_array_values(const mallard_array * array, uint64_t * n) {
    *n = array->values.size();
    return array->values.data();
}

const char * mallard_array_text(const mallard_array * array) { return array->text.c_str(); }

void mallard_array_geometry(const mallard_array * array, const double ** points, uint64_t * n_points,
                            const uint64_t ** offsets, uint64_t * n_offsets, const uint64_t ** connectivity,
                            uint64_t * n_connectivity, const uint8_t ** types, uint64_t * n_types) {
    const output::Geometry & g = array->geometry;
    *points = g.points.data();
    *n_points = g.points.size();
    *offsets = g.offsets.data();
    *n_offsets = g.offsets.size();
    *connectivity = g.connectivity.data();
    *n_connectivity = g.connectivity.size();
    *types = g.types.data();
    *n_types = g.types.size();
}

void mallard_array_free(mallard_array * array) { delete array; }

} // extern "C"
