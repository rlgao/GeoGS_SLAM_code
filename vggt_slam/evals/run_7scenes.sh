#!/bin/bash

dataset_path="/home/grl/datasets/7-scenes"

dataset="chess"
# dataset="fire"
# dataset="heads"
# dataset="office"
# dataset="pumpkin"
# dataset="redkitchen"
# dataset="stairs"


dataset_folder="${dataset_path}/${dataset}/seq-01"
log_path="$(pwd)/results/7scenes/${dataset}/poses.txt"

# python main.py --image_folder "$dataset_folder" \
#     --log_results --skip_dense_log \
#     --log_path "${log_path}" \
#     --use_sim3 --vis_map

python main.py --image_folder "$dataset_folder" \
    --use_sim3 --vis_map
