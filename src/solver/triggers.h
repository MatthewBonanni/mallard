/**
 * @file triggers.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[triggers]]: conditions on the run (an expression of t, the step,
 *        the wall time and the monitored values) that stop it, write a
 *        restart or a snapshot, or start the statistics.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TRIGGERS_H
#define TRIGGERS_H

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "data_writer.h"
#include "input.h"
#include "kexpr.h"

/** @brief The time until which a condition true since `since` must hold for `hold` (within round-off), the same
 *         on the host and the device: each product rounded on its own. */
KOKKOS_INLINE_FUNCTION double trigger_hold_end(const double since, const double hold) {
    volatile double h = hold * (1.0 - 1e-9);
    return since + h;
}

/**
 * @brief The [[triggers]] of a run. Each step the solver evaluates every
 *        condition on the host from values already reduced (where only rank 0
 *        holds them, it evaluates them and broadcasts the outcome), and
 *        advance() moves each trigger's state the same way on every rank.
 *        A trigger fires when its condition has held for `for` seconds of
 *        simulated time; it fires once, or with repeat again each time the
 *        condition turns true after being false. Restart files carry the state.
 */
class Triggers {
    public:
        enum class Action { STOP, CHECKPOINT, WRITE, START_STATISTICS };

        struct Act {
            Action action = Action::STOP;
            std::string writer;  // WRITE: the [[write_data]] prefix
        };

        struct Trigger {
            std::string name;
            std::string when;
            std::vector<Act> actions;
            bool repeat = false;
            double hold = 0.0;
            kexpr::Program condition;
            std::vector<uint16_t> used;  // variables the condition reads
            bool armed = true;
            bool holding = false;  // the condition has been true since `since`
            double since = 0.0;
            uint64_t firings = 0;
            double t_first = 0.0;  // of the first firing
        };

        /** @brief Names of the variables before the monitored values, in this order. */
        static const std::vector<std::string> & built_in();

        /**
         * @brief Read [[triggers]]. monitored names the values the solver passes after
         *        the built-in ones; writers the [[write_data]] prefixes; statistics
         *        whether the run has [statistics]. Throws InputError.
         */
        void init(const Input & input, const std::vector<std::string> & monitored,
                  const std::vector<std::string> & writers, bool statistics);

        bool empty() const { return triggers.empty(); }
        size_t size() const { return triggers.size(); }
        const Trigger & operator[](size_t i) const { return triggers[i]; }
        const std::vector<std::string> & variables() const { return names; }

        /**
         * @brief Each condition at values (by variables(), NaN where not yet known),
         *        as '1' or '0'; a condition reading an unknown value is false.
         */
        std::string evaluate(const std::vector<double> & values) const;

        /** @brief Advance every trigger with its condition (from evaluate) at time t; those that fire. */
        std::vector<size_t> advance(const std::string & conditions, double t);

        /** @brief A trigger's state between firings, as a step graph advances it (firings stay the host's). */
        struct DeviceState {
            uint32_t armed;
            uint32_t holding;
            uint32_t repeat;
            uint32_t n_used;
            double since;
            double hold;
            uint16_t used[32];  // the variables its condition reads (n_used of them)
        };
        /** @brief Whether some condition reads variable v. */
        bool reads(uint16_t v) const;
        /** @brief Whether every condition reads at most 32 variables (the step graph's limit). */
        bool fit_device() const;
        std::vector<const kexpr::Program *> conditions() const;
        std::vector<DeviceState> device_states() const;
        /** @brief The armed, holding and since of each trigger, from a step graph's states. */
        void set_states(const std::vector<DeviceState> & states);

        /** @brief Whether a trigger starts the statistics (which wait for it). */
        bool starts_statistics() const;
        /** @brief Time a trigger started the statistics, if one has. */
        std::optional<double> statistics_started() const;

        RestartAttributes attributes() const;
        /** @brief Continue the state of a restart file's attributes (fresh where it has none). */
        void restore(const RestartAttributes & attributes);

        /** @brief "when -> do" of a trigger, for the log. */
        static std::string describe(const Trigger & trigger);
        std::vector<std::pair<std::string, std::string>> summary() const;

    private:
        std::vector<Trigger> triggers;
        std::vector<std::string> names;
};

/** @brief A monitored value's name in conditions: characters other than letters, digits and _ become _. */
std::string trigger_variable(const std::string & name);

#endif // TRIGGERS_H
