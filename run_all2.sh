#/bin/bash

parallel --progress -j 0 << EOF
#python -W ignore run_sim2_dkf.py 7 4 7_nodes_4_targets
#python -W ignore run_sim2_dkf.py 10 4 10_nodes_4_targets
python -W ignore run_sim2_dkf.py 15 4 15_nodes_4_targets
python -W ignore run_sim2_dkf.py 20 4 20_nodes_4_targets
EOF

