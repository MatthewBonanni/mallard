// Expressions of x, y, z and t in the input (initial conditions, sources, profiles)
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include "expression.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    const std::string text(reinterpret_cast<const char *>(data), size);
    try {
        const Expression f("fuzz", text);
        static_cast<void>(f.depends_on_time());
        static_cast<void>(f(0.5, -1.0, 2.0, 0.25));
    } catch (const std::exception &) {
    }
    return 0;
}
