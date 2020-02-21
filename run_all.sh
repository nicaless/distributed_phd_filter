#/bin/bash

# python -W ignore run_sim_small_test.py 3 3_nodes_small_test
# python -W ignore run_sim_large_test.py 3 3_nodes_large_test
# python -W ignore run_sim_large_test.py 5 5_nodes_large_test

# python -W ignore run_sim.py 3 3_nodes
# python -W ignore run_sim.py 5 5_nodes
# python -W ignore run_sim.py 6 6_nodes
# python -W ignore run_sim.py 7 7_nodes

# python -W ignore run_sim.py 10 10_nodes
# python -W ignore run_sim.py 12 12_nodes
# python -W ignore run_sim.py 15 15_nodes
# python -W ignore run_sim.py 20 20_nodes
# python -W ignore run_sim.py 25 25_nodes
# python -W ignore run_sim.py 30 30_nodes

# python -W ignore run_sim.py 10 10_nodes_single --single_node_fail
# python -W ignore run_sim.py 12 12_nodes_single --single_node_fail
# python -W ignore run_sim.py 15 15_nodes_single --single_node_fail
# python -W ignore run_sim.py 20 20_nodes_single --single_node_fail
# python -W ignore run_sim.py 25 25_nodes_single --single_node_fail
# python -W ignore run_sim.py 30 30_nodes_single --single_node_fail

function run_sim {
    python -W ignore run_sim.py $1 $1_nodes 42 >> $1_nodes.out
}
export -f run_sim

function run_sim_single {
    python -W ignore run_sim.py $1 $1_nodes_single 42 --single_node_fail >> $1_nodes.out
}
export -f run_sim_single

function run_sim_1 {
    python -W ignore run_sim.py $1 $1_nodes_1 43
}
export -f run_sim_1

function run_sim_single_1 {
    python -W ignore run_sim.py $1 $1_nodes_single_1 43 --single_node_fail
}
export -f run_sim_single_1

function run_sim_2 {
    python -W ignore run_sim.py $1 $1_nodes_2 44 >> $1_nodes.out
}
export -f run_sim_2

function run_sim_single_2 {
    python -W ignore run_sim.py $1 $1_nodes_single_2 44 --single_node_fail
}
export -f run_sim_single_2

parallel --progress -j 0 << EOF
run_sim 5
run_sim 6
run_sim 7
run_sim 10
run_sim 12
run_sim 15
run_sim 20
run_sim 25
run_sim 30
run_sim_single 5
run_sim_single 6
run_sim_single 7
run_sim_single 10
run_sim_single 12
run_sim_single 15
run_sim_single 20
run_sim_single 25
run_sim_single 30
run_sim_1 5
run_sim_1 6
run_sim_1 7
run_sim_1 10
run_sim_1 12
run_sim_1 15
run_sim_1 20
run_sim_1 25
run_sim_1 30
run_sim_single_1 5
run_sim_single_1 6
run_sim_single_1 7
run_sim_single_1 10
run_sim_single_1 12
run_sim_single_1 15
run_sim_single_1 20
run_sim_single_1 25
run_sim_single_1 30
run_sim_2 5
run_sim_2 6
run_sim_2 7
run_sim_2 10
run_sim_2 12
run_sim_2 15
run_sim_2 20
run_sim_2 25
run_sim_2 30
run_sim_single_2 5
run_sim_single_2 6
run_sim_single_2 7
run_sim_single_2 10
run_sim_single_2 12
run_sim_single_2 15
run_sim_single_2 20
run_sim_single_2 25
run_sim_single_2 30

EOF
