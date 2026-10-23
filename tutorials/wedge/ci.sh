# The wedge tutorial at reduced size, for tutorials/ci.sh: a coarser mesh
wedge() {
    cp "$here/wedge/input.toml" .
    set_key input.toml Nx 80
    set_key input.toml Ny 60
    python3 "$here/wedge/oblique_shock.py" input.toml
    mpi "$m2d" -i input.toml > run.log
    python3 "$here/wedge/plot_mesh.py" input.toml --out figures
    python3 "$here/wedge/plot_results.py" . --out figures
}
