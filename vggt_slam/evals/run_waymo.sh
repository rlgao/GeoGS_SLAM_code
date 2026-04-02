#!/bin/bash

dataset_path="/home/grl/datasets/waymo"

# dataset="13476"
# dataset="100613"
# dataset="106762"
dataset="132384"


dataset_folder="${dataset_path}/${dataset}/FRONT/rgb"
log_path="$(pwd)/results/waymo/${dataset}/poses.txt"

python main.py --image_folder "$dataset_folder" \
    --log_results --skip_dense_log \
    --log_path "${log_path}" \
    --use_sim3 --vis_map

