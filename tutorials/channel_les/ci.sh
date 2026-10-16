# The channel LES tutorial at reduced size, for tutorials/ci.sh: 8 x 16 x 8
# cells, statistics over the last part of half a time unit
channel_les() {
    cp "$here/channel_les/input.toml" .
    set_key input.toml t_stop 0.5
    set_key input.toml t_start 0.2
    set_key input.toml interval 1
    set_key input.toml time_interval 0.25
    gmsh "$here/channel_les/channel.geo" -3 -setnumber nx 8 -setnumber ny 16 -setnumber nz 8 -o channel.msh > gmsh.log
    mpi "$m3d" -i input.toml > run.log
    python3 "$here/channel_les/plot_channel_les.py" mesh channel.msh --out figures
    python3 "$here/channel_les/plot_channel_les.py" results . --t-start 0.2 --out figures
}
