#!/bin/bash

# python run.py --config config/tum/360.yaml --save_path_parent "results/full"

ablations=(
    # full
    # wo_lc
    wo_poseopt
    wo_sample
    wo_depthprior
    wo_poseprior
)

config_path_root="config/tum/"
scenes=(
    360
    desk
    desk2
    floor
    plant
    room
    rpy
    teddy
    xyz
)

for ablation in ${ablations[@]}; do
    for scene in ${scenes[@]}; do
        config_path="$config_path_root""$scene"".yaml"
        echo "config_path: "$config_path

        save_path_parent="results/""$ablation"
        echo "save_path_parent: "$save_path_parent

        python run.py --config $config_path --save_path_parent $save_path_parent
        echo "============================"
    done
done
