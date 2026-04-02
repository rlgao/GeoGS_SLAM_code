#!/bin/bash

dataset_path="/home/grl/datasets/euroc"

# dataset="MH_01_easy"
# dataset="MH_02_easy"
# dataset="MH_03_medium"
dataset="V1_01_easy"
# dataset="V1_02_medium"
# dataset="V2_02_medium"
# dataset="V2_03_difficult"



dataset_folder="${dataset_path}/${dataset}/mav0/cam0/data"
log_path="$(pwd)/results/euroc/${dataset}/poses.txt"

python main.py --image_folder "$dataset_folder" \
    --log_results --skip_dense_log \
    --log_path "${log_path}" \
    --use_sim3 --vis_map

