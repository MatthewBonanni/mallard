/**
 * @file render_main.cpp
 * @brief `mallard post render`: images and movies of snapshots, headless, by Mallard's own reader
 *        and rasterizer; movie formats other than GIF through an ffmpeg on PATH.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cli.h"
#include "commands.h"
#include "diagnostic_code.h"
#include "image.h"
#include "log.h"
#include "output_reader.h"
#include "parallel/comm.h"
#include "render.h"
#include "snapshot_selection.h"

extern char ** environ;

namespace fs = std::filesystem;

namespace {

using cli::UsageError;
using logging::count;
using logging::format;
using logging::item;
using snapshots::parse_count;
using snapshots::parse_number;
using snapshots::Selected;
using snapshots::split_list;

/** @brief Whether a name matches a pattern of * and ?. */
bool matches(const std::string & pattern, const std::string & name) {
    size_t p = 0, n = 0, star = std::string::npos, mark = 0;
    while (n < name.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) {
            p++;
            n++;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            mark = n;
        } else if (star != std::string::npos) {
            p = star + 1;
            n = ++mark;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') p++;
    return p == pattern.size();
}

/** @brief The files of the arguments: each path, or the files a quoted pattern of its last part matches, sorted. */
std::vector<std::string> expand(const std::vector<std::string> & arguments) {
    std::vector<std::string> out;
    for (const std::string & a : arguments) {
        const fs::path path(a);
        const std::string leaf = path.filename().string();
        if (leaf.find_first_of("*?") == std::string::npos) {
            out.push_back(a);
            continue;
        }
        const fs::path dir = path.has_parent_path() ? path.parent_path() : fs::path(".");
        std::vector<std::string> found;
        std::error_code error;
        for (const auto & entry : fs::directory_iterator(dir, error)) {
            const std::string name = entry.path().filename().string();
            if (entry.is_regular_file() && matches(leaf, name)) {
                found.push_back(path.has_parent_path() ? (dir / name).string() : name);
            }
        }
        if (found.empty()) throw std::runtime_error("no file matches " + a);
        std::sort(found.begin(), found.end());
        out.insert(out.end(), found.begin(), found.end());
    }
    return out;
}

/** @brief The field drawn: a field of the files, a component of one, or a vector's magnitude. */
struct FieldChoice {
    std::string name;     // as the files name it
    int component = -1;   // -1: the value, or a vector's magnitude
    uint32_t components = 1;
    std::string label;    // "RHO [kg m-3]"
};

FieldChoice choose_field(const output::Contents & c, const std::string & wanted) {
    FieldChoice f;
    std::string base = wanted;
    if (wanted.size() > 3 && wanted.back() == ']' && wanted[wanted.size() - 3] == '[') {
        base = wanted.substr(0, wanted.size() - 3);
        f.component = wanted[wanted.size() - 2] - '0';
        if (f.component < 0 || f.component > 2) throw UsageError("--field's component is 0, 1 or 2, not " + wanted);
    } else if (!c.field(wanted) && wanted.size() > 2 && wanted[wanted.size() - 2] == '_' &&
               std::string("XYZ").find(wanted.back()) != std::string::npos) {
        base = wanted.substr(0, wanted.size() - 2);  // U_X: the x component of U, as the field catalog names it
        f.component = int(std::string("XYZ").find(wanted.back()));
    }
    const output::Field * field = c.field(base);
    if (!field) {
        std::string names;
        for (const auto & x : c.fields) names += (names.empty() ? "" : ", ") + x.name;
        throw UsageError(c.path + " has no field " + wanted + "; it has " + (names.empty() ? "none" : names));
    }
    f.name = base;
    f.components = field->components;
    if (f.component >= 0 && uint32_t(f.component) >= field->components) {
        throw UsageError(base + " has one component: give --field " + base);
    }
    const std::string shown = f.component >= 0 ? wanted : field->components > 1 ? "|" + base + "|" : base;
    f.label = shown + (field->units.empty() || field->units == "1" ? "" : " [" + field->units + "]");
    return f;
}

/** @brief The values of the field at a snapshot, one per cell. */
std::vector<double> field_values(const output::Contents & c, const FieldChoice & f, size_t snapshot) {
    std::vector<double> v = output::read(c, f.name, snapshot);
    if (f.components == 1) return v;
    const size_t n = v.size() / f.components;
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++) {
        if (f.component >= 0) {
            out[i] = v[i * f.components + size_t(f.component)];
        } else {
            double s = 0.0;
            for (size_t k = 0; k < f.components; k++) s += v[i * f.components + k] * v[i * f.components + k];
            out[i] = std::sqrt(s);
        }
    }
    return out;
}

std::vector<double> numbers(const std::string & text, const std::string & option, size_t n) {
    std::vector<double> out;
    for (const std::string & s : split_list(text)) out.push_back(parse_number(s, option));
    if (out.size() != n) throw UsageError(option + " takes " + std::to_string(n) + " numbers separated by commas, not " + text);
    return out;
}

/** @brief Where images go: one PNG, numbered PNGs, a GIF, or a movie through ffmpeg. */
class Sink {
    public:
        virtual ~Sink() = default;
        virtual void add(const image::Indexed & frame) = 0;
        virtual void close() {}
};

void write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + path);
}

/** @brief Whether a name holds one printf-style number, %d or %0Nd. */
bool is_numbered(const std::string & pattern) {
    const size_t p = pattern.find('%');
    if (p == std::string::npos) return false;
    size_t q = p + 1;
    while (q < pattern.size() && std::isdigit(static_cast<unsigned char>(pattern[q]))) q++;
    if (q >= pattern.size() || pattern[q] != 'd' || pattern.find('%', q) != std::string::npos) {
        throw UsageError("numbered frames take one %d or %0Nd (frame_%04d.png), not " + pattern);
    }
    return true;
}

/** @brief The pattern's %d or %0Nd replaced by n. */
std::string numbered_path(const std::string & pattern, int n) {
    if (!is_numbered(pattern)) return pattern;
    const size_t p = pattern.find('%'), d = pattern.find('d', p);
    const size_t width = d > p + 1 ? size_t(std::stoul(pattern.substr(p + 1, d - p - 1))) : 0;
    std::string digits = std::to_string(n);
    if (digits.size() < width) digits.insert(0, width - digits.size(), '0');
    return pattern.substr(0, p) + digits + pattern.substr(d + 1);
}

class PngSink : public Sink {
    public:
        explicit PngSink(std::string pattern_in) : pattern(std::move(pattern_in)) {}
        void add(const image::Indexed & frame) override {
            write_file(numbered_path(pattern, n), image::png(frame));
            n++;
        }

    private:
        std::string pattern;
        int n = 0;
};

class GifSink : public Sink {
    public:
        GifSink(const std::string & path, double fps) : path_(path), fps_(fps) {}
        void add(const image::Indexed & frame) override {
            if (!gif) {
                const auto delay = uint16_t(std::clamp(std::lround(100.0 / fps_), 2L, 65535L));
                gif = std::make_unique<image::Gif>(path_, frame.width, frame.height, frame.palette, delay);
            }
            gif->add(frame);
        }
        void close() override {
            if (gif) gif->close();
        }

    private:
        std::string path_;
        double fps_;
        std::unique_ptr<image::Gif> gif;
};

/** @brief A movie encoded by ffmpeg, which reads the frames as raw RGB from a pipe; it opens no socket. */
class FfmpegSink : public Sink {
    public:
        FfmpegSink(const std::string & path_in, double fps_in) : path(path_in), fps(fps_in) {}
        ~FfmpegSink() override {
            if (fd >= 0) ::close(fd);
            if (pid > 0) waitpid(pid, nullptr, 0);
        }
        void add(const image::Indexed & frame) override {
            if (pid <= 0) start(frame.width, frame.height);
            const std::vector<uint8_t> rgb = frame.rgb();
            size_t done = 0;
            while (done < rgb.size()) {
                const ssize_t n = ::write(fd, rgb.data() + done, rgb.size() - done);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) throw std::runtime_error("ffmpeg stopped while writing " + path + finish());
                done += size_t(n);
            }
        }
        void close() override {
            if (pid <= 0) return;
            const std::string why = finish();
            if (!why.empty()) throw std::runtime_error("ffmpeg could not write " + path + why);
        }

    private:
        std::string path;
        double fps;
        pid_t pid = -1;
        int fd = -1;

        void start(uint32_t width, uint32_t height) {
            const std::string ext = fs::path(path).extension().string();
            std::vector<std::string> a = {"ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-y",
                                          "-f", "rawvideo", "-pix_fmt", "rgb24",
                                          "-s", std::to_string(width) + "x" + std::to_string(height),
                                          "-r", format("%g", fps), "-i", "pipe:0"};
            if (ext == ".webm") {
                a.insert(a.end(), {"-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-pix_fmt", "yuv420p"});
            } else {
                a.insert(a.end(), {"-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"});
                if (ext == ".mp4" || ext == ".mov") a.insert(a.end(), {"-movflags", "+faststart"});
            }
            a.push_back(path);
            std::vector<char *> argv;
            for (std::string & s : a) argv.push_back(s.data());
            argv.push_back(nullptr);
            int pipe_fds[2];
            if (pipe(pipe_fds) != 0) throw std::runtime_error("cannot start ffmpeg: " + std::string(std::strerror(errno)));
            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_adddup2(&actions, pipe_fds[0], STDIN_FILENO);
            posix_spawn_file_actions_addclose(&actions, pipe_fds[1]);
            const int status = posix_spawnp(&pid, "ffmpeg", &actions, nullptr, argv.data(), environ);
            posix_spawn_file_actions_destroy(&actions);
            ::close(pipe_fds[0]);
            if (status != 0) {
                ::close(pipe_fds[1]);
                pid = -1;
                throw std::runtime_error(
                    "writing " + path + " needs ffmpeg on PATH" +
                    (status == ENOENT ? std::string(", and there is none") : ": " + std::string(std::strerror(status))) +
                    ": install ffmpeg, or write a .gif, or PNG frames (-o frames/frame_%04d.png)");
            }
            fd = pipe_fds[1];
            // A broken pipe is an error writing, not a signal that ends mallard
            std::signal(SIGPIPE, SIG_IGN);
        }

        /** @brief Waits for ffmpeg; why it failed, or empty. */
        std::string finish() {
            if (fd >= 0) ::close(fd);
            fd = -1;
            int status = 0;
            const pid_t done = waitpid(pid, &status, 0);
            pid = -1;
            if (done < 0) return ": " + std::string(std::strerror(errno));
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return "";
            return WIFEXITED(status) ? ": it exited with status " + std::to_string(WEXITSTATUS(status)) + " (its errors are above)"
                                     : ": it was stopped by a signal";
        }
};

bool is_movie(const std::string & ext) { return ext == ".mp4" || ext == ".mov" || ext == ".mkv" || ext == ".webm"; }

/** @brief A probe file's geometry on the plane, as an overlay labeled with its name. */
render::Overlay probe_overlay(const std::string & path, bool volume, int axis) {
    const output::Contents c = output::open(path);
    if (c.kind != output::Kind::probe) {
        throw UsageError("--probes takes probe files, and " + path + " is a " + output::kind_name(c.kind) + " file");
    }
    const output::Geometry g = output::read_geometry(c, c.snapshots.empty() ? 0 : c.snapshots.size() - 1);
    const auto [a, b] = render::plane_axes(volume ? axis : 2);
    const auto at = [&](uint64_t p) { return std::array<double, 2>{g.points[3 * p + size_t(a)], g.points[3 * p + size_t(b)]}; };
    render::Overlay o;
    std::string stem = fs::path(path).filename().string();
    stem = stem.substr(0, stem.find('.'));
    o.label = stem.starts_with("probe_") ? stem.substr(6) : stem;
    // Edges of lines and polylines, and of surfaces those that bound them (used by one face)
    std::map<std::pair<uint64_t, uint64_t>, int> face_edges;
    for (size_t cell = 0; cell < g.types.size(); cell++) {
        const uint64_t first = g.offsets[cell], n = g.offsets[cell + 1] - first;
        const uint64_t * ids = g.connectivity.data() + first;
        const uint8_t type = g.types[cell];
        if (type == 3 || type == 4) {
            for (uint64_t k = 0; k + 1 < n; k++) o.segments.push_back({at(ids[k]), at(ids[k + 1])});
        } else if (n >= 3) {
            for (uint64_t k = 0; k < n; k++) {
                const uint64_t p = ids[k], q = ids[(k + 1) % n];
                face_edges[{std::min(p, q), std::max(p, q)}]++;
            }
        } else if (n == 1) {
            o.points.push_back(at(ids[0]));
        }
    }
    for (const auto & [e, uses] : face_edges) {
        if (uses == 1) o.segments.push_back({at(e.first), at(e.second)});
    }
    if (g.types.empty()) {
        for (uint64_t p = 0; p < g.points.size() / 3; p++) o.points.push_back(at(p));
    }
    return o;
}

/** @brief Whether two snapshots are on one mesh, cell for cell, so the pixel map carries over. */
bool same_mesh(const output::Contents & a, const Selected & ea, const output::Contents & b, const Selected & eb) {
    if (&a == &b && ea.snapshot.file == eb.snapshot.file && a.global_order) return true;
    return a.global_order && b.global_order && !ea.snapshot.run_id.empty() && ea.snapshot.run_id == eb.snapshot.run_id &&
           a.mesh.n_cells == b.mesh.n_cells && a.mesh.n_cells > 0;
}

int render_files(const cli::Args & args) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<output::Contents> files;
    for (const std::string & path : expand(args.positional)) {
        output::Contents c = output::open(path);
        if (c.kind != output::Kind::snapshot) {
            throw std::runtime_error(
                path + " is a " + output::kind_name(c.kind) + " file; mallard post render draws snapshots" +
                (c.kind == output::Kind::restart ? " (a restart file holds no mesh: write one with mallard post extract "
                                                   "FILE --snapshot vtkhdf)"
                                                 : ""));
        }
        files.push_back(std::move(c));
    }
    const std::vector<Selected> selected = snapshots::select(files, args);
    const output::Contents & first = files[selected.front().file];

    std::string wanted = args.value_or("--field", "");
    if (wanted.empty()) {
        for (const char * name : {"RHO", "T", "P"}) {
            if (first.field(name)) {
                wanted = name;
                break;
            }
        }
        if (wanted.empty() && !first.fields.empty()) wanted = first.fields[0].name;
        if (wanted.empty()) throw std::runtime_error(first.path + " holds no field to draw");
    }
    const FieldChoice field = choose_field(first, wanted);

    const unsigned threads = args.has("--threads") ? unsigned(std::max<uint64_t>(1, parse_count(args.value_or("--threads", ""), "--threads")))
                                                   : std::max(1u, std::thread::hardware_concurrency());
    uint32_t width = 1280, height = 0;
    if (args.has("--size")) {
        const std::string text = args.value_or("--size", "");
        const size_t x = text.find('x');
        width = uint32_t(parse_count(text.substr(0, x), "--size"));
        if (x != std::string::npos) height = uint32_t(parse_count(text.substr(x + 1), "--size"));
        if (width < 64 || width > 16384 || (x != std::string::npos && (height < 64 || height > 16384))) {
            throw UsageError("--size takes WIDTH or WIDTHxHEIGHT, from 64 to 16384 pixels, not " + text);
        }
    }
    const std::string theme_name = args.value_or("--theme", "light");
    if (theme_name != "light" && theme_name != "dark") throw UsageError("--theme takes light or dark, not " + theme_name);
    const double fps = args.has("--fps") ? parse_number(args.value_or("--fps", ""), "--fps") : 20.0;
    if (!(fps > 0.0 && fps <= 240.0)) throw UsageError("--fps takes a rate from 0 to 240 frames per second");

    // The output: its kind by its name
    const bool several = selected.size() > 1;
    std::string sanitized = field.name + (field.component >= 0 ? "_" + std::string(1, "XYZ"[field.component]) : "");
    const std::string out = args.value_or("--output", sanitized + (several ? ".gif" : ".png"));
    const std::string ext = fs::path(out).extension().string();
    const bool numbered = is_numbered(out);
    if (ext != ".png" && ext != ".gif" && !is_movie(ext)) {
        throw UsageError("-o takes a .png, .gif, .mp4, .mov, .mkv or .webm file, or numbered PNG frames "
                         "(frames/frame_%04d.png), not " + out);
    }
    if (numbered && ext != ".png") throw UsageError("numbered frames are PNG files: " + out);
    std::vector<Selected> frames = selected;
    std::string note;
    if (ext == ".png" && !numbered && several) {
        frames = {selected.back()};
        note = "the last of the " + count(selected.size()) + " snapshots selected; name frames " +
               fs::path(out).stem().string() + "_%04d.png for every one";
    }

    // The plane: 2D cells as they are, 3D cells cut by the slice
    output::Geometry geometry = output::read_geometry(first, frames.front().index);
    const bool volume = render::has_volume_cells(geometry);
    int axis = 2;
    double at = std::numeric_limits<double>::quiet_NaN();
    if (args.has("--slice")) {
        if (!volume) throw UsageError("--slice cuts 3D cells, and " + first.path + " holds 2D cells");
        const std::string text = args.value_or("--slice", "");
        const size_t eq = text.find('=');
        const std::string name = text.substr(0, eq);
        if (name != "x" && name != "y" && name != "z") {
            throw UsageError("--slice takes x, y or z, with =POSITION in m (z=0.5), not " + text);
        }
        axis = name[0] - 'x';
        if (eq != std::string::npos) at = parse_number(text.substr(eq + 1), "--slice");
    }
    if (volume && std::isnan(at)) {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (size_t p = size_t(axis); p < geometry.points.size(); p += 3) {
            lo = std::min(lo, geometry.points[p]);
            hi = std::max(hi, geometry.points[p]);
        }
        at = 0.5 * (lo + hi);
    }
    auto polygons = std::make_unique<render::Polygons>(render::polygons(geometry, axis, at));
    if (polygons->size() == 0) {
        throw std::runtime_error(volume ? "no cell of " + first.path + " is on the plane " + std::string(1, char('x' + axis)) +
                                              " = " + render::number(at) + " m"
                                        : first.path + " holds no 2D cells to draw");
    }
    render::Window window{polygons->bounds[0], polygons->bounds[1], polygons->bounds[2], polygons->bounds[3]};
    if (args.has("--view")) {
        const std::vector<double> v = numbers(args.value_or("--view", ""), "--view", 4);
        if (!(v[1] > v[0] && v[3] > v[2])) throw UsageError("--view takes XMIN,XMAX,YMIN,YMAX with XMIN < XMAX and YMIN < YMAX");
        window = {v[0], v[1], v[2], v[3]};
    }
    if (!(window.x1 > window.x0 && window.y1 > window.y0)) throw std::runtime_error("the cells drawn have no extent on the plane");
    const std::string title = args.value_or("--title", "");
    const render::Layout layout = render::layout(window, width, height, args.has("--bare"), !title.empty());
    // The window the plot shows, with the plot's aspect ratio
    {
        const double cx = 0.5 * (window.x0 + window.x1), cy = 0.5 * (window.y0 + window.y1);
        const double scale = std::max((window.x1 - window.x0) / layout.plot_width, (window.y1 - window.y0) / layout.plot_height);
        window = {cx - 0.5 * scale * layout.plot_width, cx + 0.5 * scale * layout.plot_width,
                  cy - 0.5 * scale * layout.plot_height, cy + 0.5 * scale * layout.plot_height};
    }
    auto map = std::make_unique<render::PixelMap>(render::pixel_map(*polygons, window, layout.plot_width, layout.plot_height, threads));

    // Each frame's pixels, on the mesh of its snapshot
    size_t mapped = frames.front().file;
    Selected mapped_snapshot = frames.front();
    const auto pixels = [&](const Selected & e) {
        const output::Contents & c = files[e.file];
        if (!same_mesh(files[mapped], mapped_snapshot, c, e)) {
            geometry = output::read_geometry(c, e.index);
            polygons = std::make_unique<render::Polygons>(render::polygons(geometry, axis, at));
            map = std::make_unique<render::PixelMap>(render::pixel_map(*polygons, window, layout.plot_width, layout.plot_height, threads));
        }
        mapped = e.file;
        mapped_snapshot = e;
        const FieldChoice f = choose_field(c, wanted);
        return map->apply(field_values(c, f, e.index), threads);
    };

    // The color scale: over every frame, unless given
    std::vector<float> kept;
    double lo = std::numeric_limits<double>::infinity(), hi = -lo, lowest_positive = lo;
    std::vector<double> range;
    if (args.has("--range")) {
        range = numbers(args.value_or("--range", ""), "--range", 2);
        if (!(range[1] > range[0])) throw UsageError("--range takes MIN,MAX with MIN < MAX");
        lo = range[0];
        hi = range[1];
    } else {
        for (const Selected & e : frames) {
            std::vector<float> v = pixels(e);
            for (const float x : v) {
                if (std::isnan(x)) continue;
                lo = std::min(lo, double(x));
                hi = std::max(hi, double(x));
                if (x > 0.0f) lowest_positive = std::min(lowest_positive, double(x));
            }
            if (frames.size() == 1) kept = std::move(v);
        }
        if (!std::isfinite(lo)) throw std::runtime_error("no cell is in the view --view gives");
    }
    const render::FieldStyle style = render::default_style(field.name + (field.component >= 0 ? "_" + std::string(1, "XYZ"[field.component]) : ""), lo, hi);
    render::Scale scale{lo, hi, args.has("--log") || (style.log && !args.has("--colormap"))};
    if (range.empty() && style.symmetric && !args.has("--colormap")) {
        scale.lo = -std::max(std::abs(lo), std::abs(hi));
        scale.hi = -scale.lo;
    }
    if (scale.log) {
        if (!(scale.hi > 0.0)) throw std::runtime_error("--log needs positive values, and " + field.label + " has none in the view");
        if (!args.has("--range")) scale.lo = std::max(lowest_positive, scale.hi * 1e-3);
        if (!(scale.lo > 0.0)) throw UsageError("--log takes a positive --range");
    }
    if (!(scale.hi > scale.lo)) {
        const double pad = std::max(std::abs(scale.lo) * 1e-3, 1e-30);
        scale.lo -= pad;
        scale.hi += pad;
    }
    const std::string colormap_name = args.value_or("--colormap", style.colormap);
    std::array<image::Rgb, 256> colors;
    try {
        colors = render::colormap(colormap_name);
    } catch (const std::invalid_argument & e) {
        throw UsageError(std::string("--colormap: ") + e.what());
    }

    std::vector<render::Overlay> overlays;
    for (const std::string & path : expand(split_list(args.value_or("--probes", "")))) overlays.push_back(probe_overlay(path, volume, axis));

    std::unique_ptr<Sink> sink;
    if (ext == ".png") {
        sink = std::make_unique<PngSink>(out);
    } else if (ext == ".gif") {
        sink = std::make_unique<GifSink>(out, fps);
    } else {
        sink = std::make_unique<FfmpegSink>(out, fps);
    }
    if (const fs::path dir = fs::path(out).parent_path(); !dir.empty()) fs::create_directories(dir);

    // Frames are drawn while the previous one is encoded
    std::future<void> encoding;
    for (const Selected & e : frames) {
        render::Frame f;
        f.window = window;
        f.values = kept.empty() ? pixels(e) : std::move(kept);
        f.map = map.get();
        f.colormap = colors;
        f.scale = scale;
        f.colors = render::theme(theme_name == "dark");
        f.title = title;
        f.time = "t = " + format("%.6g", e.snapshot.t) + " s" + (e.step_recorded ? ", step " + std::to_string(e.step) : "");
        f.label = field.label;
        f.mesh = args.has("--mesh") ? polygons.get() : nullptr;
        f.overlays = overlays;
        auto img = std::make_shared<image::Indexed>(render::draw(layout, f, threads));
        if (encoding.valid()) encoding.get();
        encoding = std::async(std::launch::async, [&sink, img] { sink->add(*img); });
    }
    if (encoding.valid()) encoding.get();
    sink->close();

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    logging::section("mallard post render");
    item("Snapshots", count(frames.size()) + ", t = " + format("%.6g", frames.front().snapshot.t) +
                          (frames.size() > 1 ? " to " + format("%.6g", frames.back().snapshot.t) : "") + " s" +
                          (note.empty() ? "" : " (" + note + ")"));
    item("Field", field.label);
    item("Plane", volume ? std::string(1, char('x' + axis)) + " = " + render::number(at) + " m, " + count(polygons->size(), "cell") + " cut"
                         : count(polygons->size(), "cell"));
    item("View", render::number(window.x0) + " to " + render::number(window.x1) + " by " + render::number(window.y0) + " to " +
                     render::number(window.y1) + " m");
    item("Colors", colormap_name + (scale.log ? ", logarithmic" : "") + ", " + render::number(scale.lo) + " to " +
                       render::number(scale.hi));
    item("Image", std::to_string(layout.width) + " x " + std::to_string(layout.height) + " pixels, " + theme_name +
                      (frames.size() > 1 && ext != ".png" ? ", " + format("%g", fps) + " frames per second" : ""));
    item("Written", out + (numbered ? " (" + count(frames.size(), "frame") + ")" : ""));
    item("Wall time", logging::duration(seconds));
    return 0;
}

} // namespace

int post_render(const cli::Args & args) {
    try {
        return render_files(args);
    } catch (const UsageError & e) {
        return cli::usage_error(*cli::command("post render"), e.what());
    } catch (const std::exception & e) {
        logging::error(diagnostic_code::RENDER, std::string("mallard post render: ") + e.what());
        return 1;
    }
}

int render_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    if (comm::size() > 1) {
        if (comm::is_root()) {
            logging::error(diagnostic_code::ONE_RANK, "mallard post render: runs on one rank, with threads (--threads), "
                                                      "not " + std::to_string(comm::size()) + "; start it without mpirun.");
        }
        return 1;
    }
    return post_render(args);
}
