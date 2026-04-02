#!/bin/bash

bash eval/eval_replica.sh  # check: res_path_root
bash eval/eval_tum.sh
bash eval/eval_waymo.sh

python eval/read_metrics.py  # check: save_path_parent
