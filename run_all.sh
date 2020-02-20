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
    python -W ignore run_sim.py $1 $1_nodes >> $1_nodes.out
}
export -f run_sim

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
EOF
