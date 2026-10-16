# The airfoil tutorial at reduced size, for tutorials/ci.sh: a quarter of the
# cells and two time units
airfoil() {
    cp "$here/airfoil/input.toml" .
    set_key input.toml t_stop 2.0
    set_key input.toml time_interval 1.0
    gmsh "$here/airfoil/naca0012.geo" -2 -setnumber refine 0.5 -o naca0012.msh > gmsh.log
    mpi "$m2d" -i input.toml > run.log
    python3 "$here/airfoil/plot_mesh.py" naca0012.msh --out figures
    python3 "$here/airfoil/plot_results.py" . --out figures
}
