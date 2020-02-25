#/bin/bash

#python -W ignore run_sim_small_test.py 3 3_nodes_small_test
#python -W ignore run_sim_large_test.py 3 3_nodes_large_test
#python -W ignore run_sim_large_test.py 5 5_nodes_large_test

function run_sim {
    python -W ignore run_sim.py $1 $1_nodes 42 >> $1_nodes.out
}
export -f run_sim

function run_sim_single {
    python -W ignore run_sim.py $1 $1_nodes_single 42 --single_node_fail >> $1_nodes_single.out
}
export -f run_sim_single

function run_sim_1 {
    python -W ignore run_sim.py $1 $1_nodes_1 43 >> $1_nodes_1.out
}
export -f run_sim_1

function run_sim_single_1 {
    python -W ignore run_sim.py $1 $1_nodes_single_1 43 --single_node_fail >> $1_nodes_single_1.out
}
export -f run_sim_single_1

function run_sim_2 {
    python -W ignore run_sim.py $1 $1_nodes_2 44 >> $1_nodes_2.out
}
export -f run_sim_2

function run_sim_single_2 {
    python -W ignore run_sim.py $1 $1_nodes_single_2 44 --single_node_fail >> $1_nodes_single_1.out
}
export -f run_sim_single_2

parallel --progress -j 0 << EOF
run_sim 10
run_sim 12
run_sim 15
run_sim 20
run_sim 25
run_sim 30
EOF
