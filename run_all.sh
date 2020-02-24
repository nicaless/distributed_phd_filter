#/bin/bash
#python -W ignore run_sim_small_test.py 3 3_nodes_small_test
#python -W ignore run_sim_large_test.py 3 3_nodes_large_test
#python -W ignore run_sim_large_test.py 5 5_nodes_large_test

function run_sim {
    python -W ignore run_sim.py $1 $1_nodes >> $1_nodes.out
}

export -f run_sim

#python -W ignore run_sim.py 3 3_nodes
#python -W ignore run_sim.py 5 5_nodes
#python -W ignore run_sim.py 6 6_nodes
#python -W ignore run_sim.py 7 7_nodes
#python -W ignore run_sim.py 10 10_nodes
#python -W ignore run_sim.py 12 12_nodes
parallel --progress -j 0 << EOF
run_sim 15
run_sim 20
run_sim 25
run_sim 30
EOF
