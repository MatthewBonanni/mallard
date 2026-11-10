# Triggers

`[[triggers]]` (#544) pair a condition on the run with actions. User docs:
[Running and controlling jobs](../guide/running.md#triggers).

## Decisions

1. **Conditions are kexpr expressions of values already reduced.** The
   variables are `t`, `step`, `dt`, `wall` and the latest row of every
   monitor: `[integrals]`, `[[forces]]`, the reductions of `[[probes]]` and
   the values of one-point probes, named `<monitor>_<column>`. kexpr has no
   strings, so a monitor's name is part of the variable name rather than an
   argument (`combustor_max_T`, not `max(T, "combustor")`). Nothing new is
   computed on the device: a trigger reads what the monitors wrote.
2. **Evaluated on the host after every step, once that step's probes are
   written.** Probes are now written before the scheduled snapshots and
   restarts of the step (their files are independent), so a trigger reads
   this step's values and a restart written at the step carries the state
   after it; a run resumed from it continues exactly. A monitor written every
   n steps keeps its value in between; before its first value a condition
   reading it is false.
3. **Ranks agree by construction.** Probe values live on rank 0, and `wall`
   differs between ranks, so on a distributed mesh rank 0 evaluates every
   condition and broadcasts one byte per trigger; every rank then advances
   the same state machines with the same `t`. Undistributed runs evaluate
   locally. A run without triggers does nothing new: no evaluation, no
   message.
4. **State machine.** A trigger is armed until it fires; it fires once its
   condition has held for `for` seconds of simulated time, counted from the
   step it last turned true (a lapse restarts the clock, the hysteresis
   against chatter). With `repeat`, it re-arms when the condition is false,
   so it fires on each rising edge rather than every step. Restart attributes
   `TRIGGER_<name>_{FIRINGS,ARMED,HOLDING,SINCE,T_FIRST}` carry the state; a
   restart file without them starts the trigger afresh, as does a remap.
5. **Actions reuse existing paths.** `stop` is a stop reason like `t_stop`
   (final output, a restart, exit status 0); `checkpoint` is the run-control
   checkpoint; `write <prefix>` forces that writer once (time-based writers
   keep their schedule); `start statistics` holds `[statistics]` (and its
   spectra and phases) at `t_start = inf` until the trigger fires, then starts
   them as `t_start` would, so the averages equal those of a run with that
   `t_start` bit for bit (tested), also across a restart (the first firing's
   time is restored).
6. **Record.** Each firing is a log line and an `event` record of kind
   `trigger` with the trigger's name, firing count and the values its
   condition read, so a reader can see why it fired.

## Measurements

A 2D case (38,400 triangles, MUSCL, 400 steps, a 50-point and a 400-point
line probe, forces and integrals; no triggers) gives the same probe,
reduction, force and integral rows, bit for bit, as `main` before this
change; stepping time over three alternating runs on 2 threads of a loaded
machine was 4.52 s (`main`) and 4.59 s (this change), within the run-to-run
spread of 4.15-5.06 s.

## Not done (in #544 and later)

- `[run] t_stop`, `n_steps` and `t_wall_stop` as built-in triggers, and
  landing the step exactly on a time condition.
- Reductions over regions inside a condition (`max(T, "domain")`), particle
  counts, run-health values and a steady-state test (`change(q, over = ...)`);
  today a `[[probes]]` region reduction gives the first.
- `stop statistics`, `log` and `set` actions; each trigger's state in
  `mallard ctl status` (it lists the stopping triggers among the stop
  conditions) and checks of conditions against the case in
  `mallard check` beyond names.
