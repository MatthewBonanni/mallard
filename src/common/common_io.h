/**
 * @file common_io.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Common IO header file.
 * @version 0.1
 * @date 2023-12-22
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef COMMON_IO_H
#define COMMON_IO_H

#include <cstdint>
#include <stdexcept>
#include <string>

#define LEN_STEP 6

/**
 * @brief Return the endianness of the system for VTU file writing.
 * @return std::string 
 */
std::string endianness();

/**
 * @brief Return the VTK name for the floating point type.
 * @return std::string 
 */
std::string vtk_float_type();

/**
 * @brief Local node of a cell at VTK (and XDMF) position k. Mallard prisms have
 *        the (0, 1, 2) normal pointing toward (3, 4, 5); VTK wedges point it away.
 */
uint32_t vtk_local_node(uint32_t n_nodes, uint32_t k);

/** @brief VTK cell type of a linear cell or boundary face, by dimension and node count. */
uint8_t vtk_cell_type(uint32_t dim, uint32_t n_nodes);

/** @brief Whether setup creates output files and directories (not in a pre-flight check). */
bool output_files_enabled();
void set_output_files_enabled(bool on);

/**
 * @brief The error of a file that cannot be read: "cannot open the mesh file
 *        mesh.msh: No such file or directory", with the directory relative paths
 *        start from and what to check. Reads errno; call right after the failure.
 */
std::runtime_error cannot_read(const std::string & what, const std::string & file);

/**
 * @brief The error of a file that cannot be written: the file, the reason
 *        (errno), and what to check (directory, permissions, disk space).
 */
std::runtime_error cannot_write(const std::string & what, const std::string & file);

/** @brief Create the directories a file goes in, or throw cannot_write's error. */
void create_parent_directories(const std::string & what, const std::string & file);

/** @brief Suffix of a file being written: it gets its name once complete, so no file of that name is incomplete. */
inline constexpr const char * PARTIAL = ".partial";

/** @brief The name file is written under until it is complete. */
inline std::string partial_name(const std::string & file) { return file + PARTIAL; }

/** @brief Give a file written under partial_name(file) its name, or throw cannot_write's error. */
void publish_file(const std::string & what, const std::string & file);

/** @brief A file a run wrote: its size and SHA-256, for the run log's output records. */
struct WrittenFile {
    std::string path;
    uint64_t bytes = 0;
    std::string sha256;
};

/**
 * @brief An output of a run as its setup banner lists it: what it writes, to which files, how often (every steps
 *        steps, or every seconds of simulated time) and what the files hold.
 */
struct OutputSummary {
    std::string kind;
    std::string file;
    uint64_t steps = 0;
    double seconds = 0.0;
    std::string contents{};
};

/** @brief Size and (contents) SHA-256 of a complete file. */
WrittenFile written_file(const std::string & path, bool contents = true);

#endif // COMMON_IO_H