# The from-run-to-figure tutorial at reduced size, for tutorials/ci.sh: two
# coarse meshes and a short run (the wake still forming), then every
# post-processing step and figure; mallard post convergence exits with 1
# when a quantity of these unconverged runs diverges, still a report
post_processing() {
    export MALLARD_EXECUTABLE="$mallard"
    for mesh in coarse medium; do
        mkdir -p $mesh
        refine=$([ $mesh = coarse ] && echo 0.5 || echo 0.7071)
        gmsh "$here/cylinder/cylinder.geo" -2 -setnumber n_theta 64 -setnumber n_ring 10 \
            -setnumber dn0 0.04 -setnumber h_wake 0.2 -setnumber refine $refine -o $mesh/cylinder.msh > gmsh.log
        cp "$here/post_processing/input.toml" $mesh/
        set_key $mesh/input.toml t_stop 4.0
        set_key $mesh/input.toml time_interval 2.0
        (cd $mesh && mpi "$mallard" -i input.toml > run.log)
    done
    "$mallard" info medium > info.log
    "$mallard" info medium/solut/wake.vtkhdf >> info.log
    "$mallard" info medium/solut/wake.vtkhdf --input > resolved.toml
    mv medium/report.html report_of_run.html
    "$mallard" info medium --html > report.log
    cmp report_of_run.html medium/report.html
    for mesh in coarse medium; do
        (cd $mesh
         cat input.toml "$here/post_processing/probes.toml" > measure.toml
         "$mallard" post extract solut/wake.vtkhdf -i measure.toml > extract.log
         "$mallard" post average solut/wake.vtkhdf --times 2:4 --fields U_X,U_Y,P > average.log)
    done
    (cd medium && "$mallard" post derive solut/wake.vtkhdf --latest --fields Q_CRITERION,ENSTROPHY,CP \
         --reference '{rho = 1.0, u = [0.2, 0.0], p = 0.7142857142857143}' > derive.log)
    "$mallard" post convergence coarse medium --quantity Fx_pressure,Fx_viscous > convergence.log || [ $? -eq 1 ]
    python3 "$here/post_processing/plot_post_processing.py" coarse medium --out figures
}
