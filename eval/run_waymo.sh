#!/bin/bash

# python run.py --config config/waymo/13476.yaml --save_path_parent "results/full"

ablations=(
    full
    wo_lc
    wo_poseopt
    wo_sample
    wo_depthprior
    wo_poseprior
)

config_path_root="config/waymo/"
scenes=(
    13476
    100613
    106762
    132384
    152706
    153495
    158686
    163453
    405841
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
