#!/bin/bash

# python eval/eval.py --resfile results/full/replica/office0/metadata.json

#####################################################  
# res_path_root="results/full/replica/"
# res_path_root="results/wo_lc/replica/"
# res_path_root="results/wo_poseopt/replica/"
# res_path_root="results/wo_sample/replica/"
# res_path_root="results/wo_depthprior/replica/"
# res_path_root="results/wo_poseprior/replica/"
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
    office0
    office1
    office2
    office3
    office4
    room0
    room1
    room2
)

for ablation in ${ablations[@]}; do
    for scene in ${scenes[@]}; do
        res_path_root="results/""$ablation""/replica/" ###
        echo "res_path_root: "$res_path_root

        res_path="$res_path_root""$scene""/metadata.json"
        echo "res_path: "$res_path

        python eval/eval.py --resfile $res_path
        echo "============================"
    done
done
