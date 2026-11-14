/**
 * @file gated.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Kokkos's parallel_for, parallel_reduce and parallel_scan, whose kernels do nothing while the device
 *        flag of the innermost gated::Scope is 0, and copies and fills as such kernels. Every top-level kernel in
 *        src/ launches through them.
 * @version 0.1
 * @date 2026-11-14
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 * A step graph's launches after the device stopped for the host run every kernel of the step, and each exits at
 * once (Solver::device_if). A reduction that does nothing leaves its destination as it was: its final step
 * copies the destination's value into the result Kokkos then writes there.
 *
 */

#ifndef GATED_H
#define GATED_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace gated {

/** @brief The device flag that kernels launched now test; null: they run. */
inline const uint32_t *& flag() {
    static const uint32_t * f = nullptr;
    return f;
}

/** @brief While alive, kernels launched through this header do nothing when *on is 0 (null: they run). */
class Scope {
    public:
        explicit Scope(const uint32_t * on) : saved(flag()) { flag() = on; }
        ~Scope() { flag() = saved; }
        Scope(const Scope &) = delete;
        Scope & operator=(const Scope &) = delete;

    private:
        const uint32_t * saved;
};

namespace detail {

template <class F>
concept DefinesValueType = requires { typename F::value_type; };

template <class F, class V>
concept HasFinal = requires(const F f, V && v) { f.final(static_cast<V &&>(v)); };

template <class P>
void check_policy() {
    static_assert(Kokkos::is_execution_policy_v<P> || std::is_integral_v<P>,
                  "gated: top-level kernels only; nested parallel patterns use Kokkos's own");
}

/** @brief Kokkos's work count for an integer, else the policy. */
template <class P>
decltype(auto) work(const P & policy) {
    if constexpr (std::is_integral_v<P>) {
        return static_cast<size_t>(policy);
    } else {
        return (policy);
    }
}

template <class P>
struct Space {
    using type = typename P::execution_space;
};
template <class P>
    requires std::is_integral_v<P>
struct Space<P> {
    using type = Kokkos::DefaultExecutionSpace;
};

template <class P, class M>
constexpr bool accessible = Kokkos::SpaceAccessibility<typename Space<P>::type, M>::accessible;

inline void unsupported(const char * what) {
    if (flag()) throw std::logic_error(std::string("gated: ") + what + " under a gated::Scope");
}

/** @brief Copies the destination old (count values for arrays) into the result v. */
template <class V>
KOKKOS_FUNCTION void restore(V && v, const void * old, const size_t count) {
    using R = std::remove_reference_t<V>;
    if constexpr (std::is_pointer_v<R>) {
        using T = std::remove_cv_t<std::remove_pointer_t<R>>;
        for (size_t i = 0; i < count; i++) v[i] = static_cast<const T *>(old)[i];
    } else {
        v = *static_cast<const std::remove_cv_t<R> *>(old);
    }
}

// Kokkos passes its index types to the functors, as Kokkos itself does from its system headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"

template <class F>
struct For {
    F f;
    const uint32_t * on;

    template <class... A>
    KOKKOS_FUNCTION void operator()(A &&... a) const {
        if (on && !*on) return;
        f(static_cast<A &&>(a)...);
    }
};

/** @brief A reduction's functor (a lambda); old: its destination, or null. */
template <class F>
struct Reduce {
    F f;
    const uint32_t * on;
    const void * old;
    size_t count;

    template <class... A>
    KOKKOS_FUNCTION void operator()(A &&... a) const {
        if (on && !*on) return;
        f(static_cast<A &&>(a)...);
    }
    template <class V>
    KOKKOS_FUNCTION void final(V && v) const {
        if (on && !*on && old) restore(static_cast<V &&>(v), old, count);
    }
};

/** @brief A reduction's functor that defines its value type (and its init, join and final) itself. */
template <DefinesValueType F>
struct Reduce<F> : F {
    const uint32_t * on;
    const void * old;
    size_t count;

    Reduce(const F & f, const uint32_t * on_, const void * old_, const size_t count_)
        : F(f), on(on_), old(old_), count(count_) {}
    template <class... A>
    KOKKOS_FUNCTION void operator()(A &&... a) const {
        if (on && !*on) return;
        static_cast<const F &>(*this)(static_cast<A &&>(a)...);
    }
    template <class V>
    KOKKOS_FUNCTION void final(V && v) const {
        if (on && !*on) {
            if (old) restore(static_cast<V &&>(v), old, count);
        } else if constexpr (HasFinal<F, V>) {
            static_cast<const F &>(*this).final(static_cast<V &&>(v));
        }
    }
};

/** @brief A Kokkos reducer whose destination keeps its value when the kernel does nothing. */
template <class R>
struct Reducer {
    using reducer = Reducer;
    using value_type = typename R::value_type;
    using result_view_type = typename R::result_view_type;

    R r;
    const uint32_t * on;

    KOKKOS_FUNCTION void join(value_type & dst, const value_type & src) const { r.join(dst, src); }
    KOKKOS_FUNCTION void init(value_type & v) const { r.init(v); }
    KOKKOS_FUNCTION void final(value_type & v) const {
        if (on && !*on) v = *r.view().data();
    }
    KOKKOS_FUNCTION value_type & reference() const { return r.reference(); }
    KOKKOS_FUNCTION result_view_type view() const { return r.view(); }
    KOKKOS_FUNCTION bool references_scalar() const { return r.references_scalar(); }
};

template <class F, class P>
struct Scan {
    using value_type = typename Kokkos::Impl::FunctorAnalysis<Kokkos::Impl::FunctorPatternInterface::SCAN,
                                                              std::conditional_t<std::is_integral_v<P>,
                                                                                 Kokkos::RangePolicy<>, P>,
                                                              F, void>::value_type;
    F f;
    const uint32_t * on;

    template <class... A>
    KOKKOS_FUNCTION void operator()(A &&... a) const {
        if (on && !*on) return;
        f(static_cast<A &&>(a)...);
    }
};

#pragma GCC diagnostic pop

template <class... R>
struct Results {
    static constexpr bool reducer = false, view = false;
};
template <class R>
struct Results<R> {
    using T = std::remove_cvref_t<R>;
    static constexpr bool reducer = Kokkos::is_reducer_v<T>, view = Kokkos::is_view_v<T>;
};

template <class P, class F, class... R>
void reduce(const std::string * label, const P & policy, const F & f, R &&... results) {
    check_policy<P>();
    const auto run = [&](const auto & functor, auto &&... r) {
        if (label) {
            Kokkos::parallel_reduce(*label, work(policy), functor, static_cast<decltype(r) &&>(r)...);
        } else {
            Kokkos::parallel_reduce(work(policy), functor, static_cast<decltype(r) &&>(r)...);
        }
    };
    const uint32_t * const on = flag();
    if constexpr (Results<R...>::reducer) {
        using Result = typename Results<R...>::T;
        if constexpr (!accessible<P, typename Result::result_view_type::memory_space>) {
            unsupported("a reduction into host memory");
        }
        const Reducer<Result> reducer{results..., on};
        if constexpr (DefinesValueType<F>) {
            run(Reduce<F>(f, on, nullptr, 0), reducer);
        } else {
            run(For<F>{f, on}, reducer);
        }
    } else if constexpr (Results<R...>::view) {
        using Result = typename Results<R...>::T;
        if constexpr (!accessible<P, typename Result::memory_space>) unsupported("a reduction into host memory");
        const Result result = (results, ...);
        if constexpr (DefinesValueType<F>) {
            run(Reduce<F>(f, on, result.data(), result.span()), result);
        } else {
            run(Reduce<F>{f, on, result.data(), result.span()}, result);
        }
    } else {
        unsupported("a reduction into host values");
        if constexpr (DefinesValueType<F>) {
            run(Reduce<F>(f, on, nullptr, 0), static_cast<R &&>(results)...);
        } else {
            run(For<F>{f, on}, static_cast<R &&>(results)...);
        }
    }
}

} // namespace detail

template <class P, class F>
    requires(!std::is_convertible_v<const P &, std::string>)
void parallel_for(const P & policy, const F & f) {
    detail::check_policy<P>();
    Kokkos::parallel_for(detail::work(policy), detail::For<F>{f, flag()});
}

template <class P, class F>
void parallel_for(const std::string & label, const P & policy, const F & f) {
    detail::check_policy<P>();
    Kokkos::parallel_for(label, detail::work(policy), detail::For<F>{f, flag()});
}

template <class P, class F, class... R>
    requires(!std::is_convertible_v<const P &, std::string>)
void parallel_reduce(const P & policy, const F & f, R &&... results) {
    detail::reduce(nullptr, policy, f, static_cast<R &&>(results)...);
}

template <class P, class F, class... R>
void parallel_reduce(const std::string & label, const P & policy, const F & f, R &&... results) {
    detail::reduce(&label, policy, f, static_cast<R &&>(results)...);
}

template <class P, class F, class... T>
    requires(!std::is_convertible_v<const P &, std::string>)
void parallel_scan(const P & policy, const F & f, T &&... total) {
    detail::check_policy<P>();
    if constexpr (sizeof...(T) > 0) detail::unsupported("a scan's total");
    Kokkos::parallel_scan(detail::work(policy), detail::Scan<F, P>{f, flag()}, static_cast<T &&>(total)...);
}

template <class P, class F, class... T>
void parallel_scan(const std::string & label, const P & policy, const F & f, T &&... total) {
    detail::check_policy<P>();
    if constexpr (sizeof...(T) > 0) detail::unsupported("a scan's total");
    Kokkos::parallel_scan(label, detail::work(policy), detail::Scan<F, P>{f, flag()}, static_cast<T &&>(total)...);
}

/** @brief Kokkos::deep_copy(exec, dst, src) of contiguous Views of one layout and span, as a kernel on exec. */
template <class E, class D, class S>
void copy(const E & exec, const D & dst, const S & src) {
    static_assert(D::rank <= 1 || std::is_same_v<typename D::array_layout, typename S::array_layout>,
                  "gated::copy: Views of one layout");
    if (dst.span() != src.span() || !dst.span_is_contiguous() || !src.span_is_contiguous()) {
        throw std::logic_error("gated::copy: \"" + src.label() + "\" to \"" + dst.label() +
                               "\" are not contiguous Views of one span");
    }
    const auto d = dst.data();
    const auto s = src.data();
    gated::parallel_for("gated_copy", Kokkos::RangePolicy<E>(exec, 0, dst.span()), KOKKOS_LAMBDA(const size_t i) { d[i] = s[i]; });
}

template <class D, class S>
void copy(const D & dst, const S & src) {
    copy(Kokkos::DefaultExecutionSpace(), dst, src);
}

/** @brief Kokkos::deep_copy(exec, dst, value) of a contiguous View, as a kernel on exec. */
template <class E, class D>
void fill(const E & exec, const D & dst, const typename D::non_const_value_type value) {
    if (!dst.span_is_contiguous()) throw std::logic_error("gated::fill: \"" + dst.label() + "\" is not contiguous");
    const auto d = dst.data();
    gated::parallel_for("gated_fill", Kokkos::RangePolicy<E>(exec, 0, dst.span()), KOKKOS_LAMBDA(const size_t i) { d[i] = value; });
}

template <class D>
void fill(const D & dst, const typename D::non_const_value_type value) {
    fill(Kokkos::DefaultExecutionSpace(), dst, value);
}

} // namespace gated

#endif // GATED_H
