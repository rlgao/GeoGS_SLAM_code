#!/bin/bash

dataset_path="/home/grl/datasets/tum"


# dataset="rgbd_dataset_freiburg1_360"
dataset="rgbd_dataset_freiburg1_desk"
# dataset="rgbd_dataset_freiburg1_desk2"
# dataset="rgbd_dataset_freiburg1_floor"
# dataset="rgbd_dataset_freiburg1_plant"
# dataset="rgbd_dataset_freiburg1_room"
# dataset="rgbd_dataset_freiburg1_rpy"
# dataset="rgbd_dataset_freiburg1_teddy"
# dataset="rgbd_dataset_freiburg1_xyz"

# dataset="rgbd_dataset_freiburg2_xyz"
# dataset="rgbd_dataset_freiburg3_long_office_household"



dataset_folder="${dataset_path}/${dataset}/rgb"
# log_path="$(pwd)/results/tum/${dataset}/poses.txt"
log_path="$(pwd)/results/tum/${dataset}/poses_no_flow_downsample.txt"

python main.py --image_folder "$dataset_folder" \
    --log_results --skip_dense_log \
    --log_path "${log_path}" \
    --use_sim3 --vis_map
