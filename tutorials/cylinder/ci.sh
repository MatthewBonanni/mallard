# The cylinder tutorial at reduced size, for tutorials/ci.sh: a coarser mesh
# and a short run (no shedding yet)
cylinder() {
    cp "$here/cylinder/input.toml" .
    gmsh "$here/cylinder/cylinder.geo" -2 -setnumber n_theta 48 -setnumber n_ring 8 \
        -setnumber dn0 0.02 -setnumber h_wake 0.3 -o cylinder.msh > gmsh.log
    set_key input.toml t_stop 4.0
    set_key input.toml time_interval 2.0
    mpi "$mallard" -i input.toml > run.log
    python3 "$here/cylinder/plot_mesh.py" cylinder.msh --out figures
    python3 "$here/cylinder/plot_results.py" . --t-start 0 --out figures
}
