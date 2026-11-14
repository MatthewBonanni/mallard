/**
 * @file reader_c_api.h
 * @brief The output reader (output_reader.h) as a C library, libmallard2d or
 *        libmallard3d, which the Python package loads with ctypes. Every
 *        function returns 0, or a pointer, on success; on an error -1 or
 *        null, and mallard_error() says why.
 *
 * Design: docs/design/output_reader.md.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef READER_C_API_H
#define READER_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mallard_file mallard_file;
typedef struct mallard_array mallard_array;

/** @brief The message of the last error on this thread. */
const char * mallard_error(void);
/** @brief The version of the build, as runs record it (`git describe`, else "0.10.0"). */
const char * mallard_library_version(void);
/** @brief The dimension of the build, 2 or 3: that of the mesh files it reads. */
int mallard_dim(void);

/** @brief output::open(); null on an error. */
mallard_file * mallard_open(const char * path);
/** @brief What the file holds, as the JSON of mallard info --format json; valid until mallard_close. */
const char * mallard_contents(const mallard_file * file);
void mallard_close(mallard_file * file);

/** @brief output::read(): a field at a snapshot; null on an error. */
mallard_array * mallard_read(const mallard_file * file, const char * field, uint64_t snapshot);
/** @brief output::read_all(): a field at every snapshot, one after the other. */
mallard_array * mallard_read_all(const mallard_file * file, const char * field);
/**
 * @brief output::read_geometry(): points, offsets, connectivity and types,
 *        the arrays mallard_array_geometry() gives.
 */
mallard_array * mallard_read_geometry(const mallard_file * file, uint64_t snapshot);
/** @brief output::run_files(): the output files of a run directory, as a JSON array of paths. */
mallard_array * mallard_run_files(const char * dir);

/** @brief The values of an array from mallard_read or mallard_read_all, and their count. */
const double * mallard_array_values(const mallard_array * array, uint64_t * n);
/** @brief The text of an array from mallard_run_files. */
const char * mallard_array_text(const mallard_array * array);
/** @brief The arrays of a geometry, each with its length (points: three coordinates per point). */
void mallard_array_geometry(const mallard_array * array, const double ** points, uint64_t * n_points,
                            const uint64_t ** offsets, uint64_t * n_offsets, const uint64_t ** connectivity,
                            uint64_t * n_connectivity, const uint8_t ** types, uint64_t * n_types);
void mallard_array_free(mallard_array * array);

#ifdef __cplusplus
}
#endif

#endif // READER_C_API_H
