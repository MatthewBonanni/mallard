/**
 * @file compiled_expression.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Expressions compiled once on the host into a stack program that
 *        kernels evaluate on the device (and the host) with the same code.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef COMPILED_EXPRESSION_H
#define COMPILED_EXPRESSION_H

#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

namespace expr {

/** @brief Deepest evaluation stack a program may need. */
constexpr int MAX_STACK = 32;

enum class Op : uint8_t {
    CONST, VAR,
    NEG, ADD, SUB, MUL, DIV, MOD, POW, IPOW, IPOWINV,
    LT, LE, GT, GE, EQ, NE, AND, OR, XOR, NAND, NOR, XNOR, NOT,
    SELECT, CLAMP, INRANGE,
    MIN, MAX, ATAN2, HYPOT, LOGN, ROOT, ROUNDN,
    ABS, ACOS, ACOSH, ASIN, ASINH, ATAN, ATANH, CEIL, COS, COSH, COT, CSC, SEC, ERF, ERFC, EXP, EXPM1,
    FLOOR, FRAC, LOG, LOG10, LOG1P, LOG2, NCDF, ROUND, SGN, SIN, SINC, SINH, SQRT, TAN, TANH, TRUNC,
    DEG2RAD, RAD2DEG,
};

/** @brief One instruction: push a constant or variable, or apply an operation to the top of the stack. */
struct Instr {
    double value = 0.0;   // CONST
    uint16_t index = 0;   // VAR: the variable; IPOW, IPOWINV: the exponent
    Op op = Op::CONST;
};

namespace detail {

KOKKOS_INLINE_FUNCTION double p3(const double a) { return a * a * a; }

/** @brief exprtk's integer powers (numeric::fast_exp), operation for operation. */
KOKKOS_INLINE_FUNCTION double ipow(double v, const uint32_t n) {
    switch (n) {
        case 0: return 1.0;
        case 1: return v;
        case 2: return v * v;
        case 3: return p3(v);
        case 4: { const double v2 = v * v; return v2 * v2; }
        case 5: { const double v2 = v * v; return (v2 * v2) * v; }
        case 6: { const double v3 = p3(v); return v3 * v3; }
        case 7: { const double v3 = p3(v); return (v3 * v3) * v; }
        case 8: { const double v2 = v * v; const double v4 = v2 * v2; return v4 * v4; }
        case 9: { const double v2 = v * v; const double v4 = v2 * v2; return (v4 * v4) * v; }
        case 10: { const double v2 = v * v; const double v5 = (v2 * v2) * v; return v5 * v5; }
        default: {
            uint32_t k = n;
            double l = 1.0;
            while (k) {
                if (k % 2 == 1) {
                    l *= v;
                    --k;
                }
                v *= v;
                k /= 2;
            }
            return l;
        }
    }
}

KOKKOS_INLINE_FUNCTION double truth(const bool b) { return b ? 1.0 : 0.0; }

/** @brief a * b rounded on its own, never contracted into a fused multiply-add. */
KOKKOS_INLINE_FUNCTION double product(const double a, const double b) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IF_ON_DEVICE((return __dmul_rn(a, b);))
#endif
    volatile double p = a * b;
    return p;
}

KOKKOS_INLINE_FUNCTION double unary(const Op op, const double v) {
    constexpr double log2_e = 0.69314718055994530941723212145817656807550013436026;
    constexpr double pi_180 = 0.01745329251994329576923690768488612713442871888542;
    constexpr double d180_pi = 57.29577951308232087679815481410517033240547246656443;
    constexpr double sqrt2 = 1.41421356237309504880168872420969807856967187537695;
    switch (op) {
        case Op::NEG: return -v;
        case Op::NOT: return truth(v == 0.0);
        case Op::ABS: return v < 0.0 ? -v : v;
        case Op::ACOS: return Kokkos::acos(v);
        case Op::ACOSH: return Kokkos::acosh(v);
        case Op::ASIN: return Kokkos::asin(v);
        case Op::ASINH: return Kokkos::asinh(v);
        case Op::ATAN: return Kokkos::atan(v);
        case Op::ATANH: return Kokkos::atanh(v);
        case Op::CEIL: return Kokkos::ceil(v);
        case Op::COS: return Kokkos::cos(v);
        case Op::COSH: return Kokkos::cosh(v);
        case Op::COT: return 1.0 / Kokkos::tan(v);
        case Op::CSC: return 1.0 / Kokkos::sin(v);
        case Op::SEC: return 1.0 / Kokkos::cos(v);
        case Op::ERF: return Kokkos::erf(v);
        case Op::ERFC: return Kokkos::erfc(v);
        case Op::EXP: return Kokkos::exp(v);
        case Op::EXPM1: return Kokkos::expm1(v);
        case Op::FLOOR: return Kokkos::floor(v);
        case Op::FRAC: return v - Kokkos::trunc(v);
        case Op::LOG: return Kokkos::log(v);
        case Op::LOG10: return Kokkos::log10(v);
        case Op::LOG1P: return Kokkos::log1p(v);
        case Op::LOG2: return Kokkos::log(v) / log2_e;
        case Op::NCDF: return 0.5 * Kokkos::erfc(-(v / sqrt2));
        case Op::ROUND: return v < 0.0 ? Kokkos::ceil(v - 0.5) : Kokkos::floor(v + 0.5);
        case Op::SGN: return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
        case Op::SIN: return Kokkos::sin(v);
        case Op::SINC: return Kokkos::fabs(v) >= 2.220446049250313e-16 ? Kokkos::sin(v) / v : 1.0;
        case Op::SINH: return Kokkos::sinh(v);
        case Op::SQRT: return Kokkos::sqrt(v);
        case Op::TAN: return Kokkos::tan(v);
        case Op::TANH: return Kokkos::tanh(v);
        case Op::TRUNC: return Kokkos::trunc(v);
        case Op::DEG2RAD: return v * pi_180;
        case Op::RAD2DEG: return v * d180_pi;
        default: return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION double binary(const Op op, const double a, const double b) {
    switch (op) {
        case Op::ADD: return a + b;
        case Op::SUB: return a - b;
        case Op::MUL: return a * b;
        case Op::DIV: return a / b;
        case Op::MOD: return Kokkos::fmod(a, b);
        case Op::POW: return Kokkos::pow(a, b);
        case Op::LT: return truth(a < b);
        case Op::LE: return truth(a <= b);
        case Op::GT: return truth(a > b);
        case Op::GE: return truth(a >= b);
        case Op::EQ: return truth(a == b);
        case Op::NE: return truth(a != b);
        case Op::AND: return truth(a != 0.0 && b != 0.0);
        case Op::OR: return truth(a != 0.0 || b != 0.0);
        case Op::XOR: return truth((a == 0.0) != (b == 0.0));
        case Op::NAND: return truth(a == 0.0 || b == 0.0);
        case Op::NOR: return truth(a == 0.0 && b == 0.0);
        case Op::XNOR: return truth((a != 0.0) == (b != 0.0));
        case Op::MIN: return b < a ? b : a;
        case Op::MAX: return a < b ? b : a;
        case Op::ATAN2: return Kokkos::atan2(a, b);
        case Op::HYPOT: return Kokkos::sqrt(product(a, a) + product(b, b));
        case Op::LOGN: return Kokkos::log(a) / Kokkos::log(b);
        case Op::ROOT:
            if (a < 0.0) {
                return (b == Kokkos::trunc(b) && Kokkos::fmod(b, 2.0) != 0.0) ? -Kokkos::pow(-a, 1.0 / b)
                                                                              : Kokkos::nan("");
            }
            return Kokkos::pow(a, 1.0 / b);
        case Op::ROUNDN: {
            const double f = Kokkos::floor(b);
            const int n = f < 0.0 ? 0 : (f > 16.0 ? 16 : static_cast<int>(f));
            double p10 = 1.0;
            for (int i = 0; i < n; i++) p10 *= 10.0;
            return a < 0.0 ? Kokkos::ceil(product(a, p10) - 0.5) / p10 : Kokkos::floor(product(a, p10) + 0.5) / p10;
        }
        default: return 0.0;
    }
}

} // namespace detail

/**
 * @brief Evaluate the program code[0, n) with variables vars(i) (any
 *        callable from a variable index to its value).
 */
template <typename Vars>
KOKKOS_INLINE_FUNCTION double evaluate(const Instr * code, const uint32_t n, const Vars & vars) {
    double s[MAX_STACK] = {};
    int top = -1;
    for (uint32_t i = 0; i < n; i++) {
        const Instr & in = code[i];
        switch (in.op) {
            case Op::CONST: s[++top] = in.value; break;
            case Op::VAR: s[++top] = vars(in.index); break;
            case Op::IPOW: s[top] = detail::ipow(s[top], in.index); break;
            case Op::IPOWINV: s[top] = 1.0 / detail::ipow(s[top], in.index); break;
            case Op::SELECT:
                top -= 2;
                s[top] = s[top] != 0.0 ? s[top + 1] : s[top + 2];
                break;
            case Op::CLAMP:
                top -= 2;
                s[top] = s[top + 1] < s[top] ? s[top] : (s[top + 1] > s[top + 2] ? s[top + 2] : s[top + 1]);
                break;
            case Op::INRANGE:
                top -= 2;
                s[top] = detail::truth(s[top] <= s[top + 1] && s[top + 1] <= s[top + 2]);
                break;
            default:
                if (in.op >= Op::NEG && in.op <= Op::POW) {
                    if (in.op == Op::NEG) {
                        s[top] = -s[top];
                    } else {
                        top--;
                        s[top] = detail::binary(in.op, s[top], s[top + 1]);
                    }
                } else if (in.op >= Op::LT && in.op <= Op::XNOR) {
                    top--;
                    s[top] = detail::binary(in.op, s[top], s[top + 1]);
                } else if (in.op >= Op::MIN && in.op <= Op::ROUNDN) {
                    top--;
                    s[top] = detail::binary(in.op, s[top], s[top + 1]);
                } else {
                    s[top] = detail::unary(in.op, s[top]);
                }
                break;
        }
    }
    return s[0];
}

/** @brief Variables read from an array. */
struct ArrayVariables {
    const double * values;
    KOKKOS_INLINE_FUNCTION double operator()(const uint16_t i) const { return values[i]; }
};

/**
 * @brief An expression compiled from exprtk syntax (the subset of
 *        arithmetic, comparisons, logic, the ternary operator and exprtk's
 *        scalar functions), evaluated as exprtk evaluates it with strength
 *        reduction off: the same operations in the same order, integer powers
 *        by exprtk's multiplications, constants folded. Unlike exprtk,
 *        variable names are case sensitive (T is not t).
 */
class Program {
    public:
        /**
         * @param name Name for error messages.
         * @param text The expression.
         * @param variables Names of the variables it may read, by index.
         */
        Program(const std::string & name, const std::string & text, const std::vector<std::string> & variables);

        const std::vector<Instr> & code() const { return m_code; }

        /** @brief Whether the expression reads variable i. */
        bool uses(const uint16_t i) const { return i < m_uses.size() && m_uses[i]; }

        /** @brief Whether the value is a constant (no variables). */
        bool is_constant() const { return m_code.size() == 1 && m_code[0].op == Op::CONST; }

        /** @brief Evaluate on the host with variable values vars[0, n_variables). */
        double operator()(const double * vars) const {
            return evaluate(m_code.data(), static_cast<uint32_t>(m_code.size()), ArrayVariables{vars});
        }

    private:
        std::vector<Instr> m_code;
        std::vector<bool> m_uses;
};

/**
 * @brief Programs gathered into one device array, each evaluated by its
 *        index in kernels. Copyable by value into kernels.
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct ProgramTable {
    Kokkos::View<const Instr *, MemorySpace> code;
    Kokkos::View<const uint32_t *, MemorySpace> offsets;  // program p is code[offsets(p), offsets(p + 1))

    ProgramTable() = default;

    explicit ProgramTable(const std::vector<const Program *> & programs) {
        std::vector<Instr> all;
        std::vector<uint32_t> off(1, 0);
        for (const Program * p : programs) {
            all.insert(all.end(), p->code().begin(), p->code().end());
            off.push_back(static_cast<uint32_t>(all.size()));
        }
        Kokkos::View<Instr *, MemorySpace> c("expr_code", all.size());
        Kokkos::View<uint32_t *, MemorySpace> o("expr_offsets", off.size());
        Kokkos::deep_copy(c, Kokkos::View<const Instr *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(all.data(), all.size()));
        Kokkos::deep_copy(o, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(off.data(), off.size()));
        code = c;
        offsets = o;
    }

    KOKKOS_INLINE_FUNCTION uint32_t size() const {
        return offsets.extent(0) > 0 ? static_cast<uint32_t>(offsets.extent(0) - 1) : 0u;
    }

    template <typename Vars>
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t p, const Vars & vars) const {
        const uint32_t begin = offsets(p);
        return evaluate(code.data() + begin, offsets(p + 1) - begin, vars);
    }
};

} // namespace expr

#endif // COMPILED_EXPRESSION_H
