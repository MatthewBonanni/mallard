// TOML input files: parsing, the lookups setup makes, number and unit conversions
// with their error messages, and the unused-key report (with suggestions) of --check
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "input.h"

namespace {

// Read about half of the keys (chosen by name), so the rest are reported as unused
void read_some(const Input & v, const std::string & path) {
    if (v.is_table()) {
        std::vector<std::string> keys;
        for (const auto & entry : v.as_table().entries()) keys.push_back(entry.first);
        for (const auto & key : keys) {
            if (std::hash<std::string>{}(key) % 2 == 0) continue;
            if (v.contains(key)) read_some(v.at(key), path.empty() ? key : path + "." + key);
        }
        static_cast<void>(v.contains("Nx"));
    } else if (v.is_array()) {
        for (const auto & item : v.as_array()) read_some(item, path + "[]");
    } else {
        try {
            const Dimension dims[] = {dim::none, dim::length, dim::temperature};
            static_cast<void>(as_real(v, path, dims[std::hash<std::string>{}(path) % 3]));
            static_cast<void>(toml::get<uint32_t>(v));
        } catch (const std::exception &) {
        }
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    const std::string text(reinterpret_cast<const char *>(data), size);
    try {
        const Input input = parse_input_string(text, "fuzz.toml");
        read_some(input, "");
        for (const Diagnostic & d : unused_keys(input)) static_cast<void>(describe(d));
        static_cast<void>(toml::format(input));
    } catch (const std::exception &) {
    }
    return 0;
}
