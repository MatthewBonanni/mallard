# The cluster tutorial at reduced size, for tutorials/ci.sh: a coarser mesh,
# sized with mallard check, run through tools/jobs/mpirun.sh, queried,
# checkpointed, stopped as by a scheduler and resumed to the end
cluster() {
    cp "$here/cluster/input.toml" "$here/../tools/jobs/mpirun.sh" .
    set_key input.toml Nx 240
    set_key input.toml Ny 60
    "$mallard" check input.toml --ranks 4 --throughput 5e7 > check.log || return 1
    export MALLARD="$mallard" NRANKS=2 THREADS=1
    ./mpirun.sh > run.log 2>&1 &
    local job=$!
    local i
    for i in $(seq 300); do [ -S mallard.sock ] && break; sleep 0.2; done
    "$mallard" ctl . status || { kill -TERM "$job"; return 1; }
    "$mallard" ctl . checkpoint || { kill -TERM "$job"; return 1; }
    kill -TERM "$job"
    local status=0
    wait "$job" || status=$?
    [ "$status" -eq 3 ] || { echo "stopped job exited with $status, not 3"; return 1; }
    ./mpirun.sh > resume.log 2>&1
    grep -q "Initial state *restart from restart_" resume.log || return 1
    grep -q "t_stop = 0.2 reached" resume.log || return 1
    python3 "$here/cluster/plot_run.py" . --out figures
}
