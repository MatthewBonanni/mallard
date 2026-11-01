# The measuring tutorial at reduced size, for tutorials/ci.sh: the cylinder
# tutorial's coarse CI mesh and a short run, averaged from t = 2 (no shedding
# yet; every file and figure is still written, the box snapshot at the end)
measuring() {
    cp "$here/measuring/input.toml" .
    gmsh "$here/cylinder/cylinder.geo" -2 -setnumber n_theta 48 -setnumber n_ring 8 \
        -setnumber dn0 0.02 -setnumber h_wake 0.3 -o cylinder.msh > gmsh.log
    set_key input.toml t_stop 6.0
    set_key input.toml t_start 2.0
    mpi "$mallard" -i input.toml > run.log
    python3 "$here/measuring/plot_measurements.py" . --t-start 2 --out figures
}
