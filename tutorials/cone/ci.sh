# The cone tutorial at reduced size, for tutorials/ci.sh: a coarse mesh
cone() {
    cp "$here/cone/input.toml" .
    gmsh "$here/cone/cone.geo" -3 -setnumber h 0.06 -setnumber h_out 0.12 -o cone.msh > gmsh.log
    mpi "$mallard" -i input.toml > run.log
    python3 "$here/cone/taylor_maccoll.py" --csv taylor_maccoll.csv
    python3 "$here/cone/plot_mesh.py" cone.msh --out figures
    python3 "$here/cone/plot_results.py" . --out figures
}
