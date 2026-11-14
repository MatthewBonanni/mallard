# The premixed-flame tutorial at reduced size, for tutorials/ci.sh: 10 cells
# per thermal thickness, one flame time
premixed_flame() {
    python3 "$here/premixed_flame/flame_setup.py" 1.0 . --cells-per-delta 10 --flame-times 1 \
        --mechanism "$here/../mechanisms/h2o2.yaml"
    sh mesh.sh > gmsh.log
    mpi "$m2d" -i input.toml > run.log
    python3 "$here/premixed_flame/plot_flame.py" setup . --out figures
    python3 "$here/premixed_flame/plot_flame.py" results . --out figures
}
