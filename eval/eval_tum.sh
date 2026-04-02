#!/bin/bash

# python eval/eval.py --resfile results/full/tum/360/metadata.json

#####################################################  
# res_path_root="results/full/tum/"
# res_path_root="results/wo_lc/tum/"
# res_path_root="results/wo_poseopt/tum/"
# res_path_root="results/wo_sample/tum/"
# res_path_root="results/wo_depthprior/tum/"
# res_path_root="results/wo_poseprior/tum/"
#####################################################  

ablations=(
    # full
    # wo_lc
    wo_poseopt
    wo_sample
    wo_depthprior
    wo_poseprior
)

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
        res_path_root="results/""$ablation""/tum/" ###
        echo "res_path_root: "$res_path_root

        res_path="$res_path_root""$scene""/metadata.json"
        echo "res_path: "$res_path

        python eval/eval.py --resfile $res_path
        echo "============================"
    done
done
