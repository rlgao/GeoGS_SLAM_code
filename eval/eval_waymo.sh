#!/bin/bash

# python eval/eval.py --resfile results/full/waymo/13476/metadata.json

#####################################################  
# res_path_root="results/full/waymo/"
# res_path_root="results/wo_lc/waymo/"
# res_path_root="results/wo_poseopt/waymo/"
# res_path_root="results/wo_sample/waymo/"
# res_path_root="results/wo_depthprior/waymo/"
# res_path_root="results/wo_poseprior/waymo/"
#####################################################  

ablations=(
    full
    wo_lc
    wo_poseopt
    wo_sample
    wo_depthprior
    wo_poseprior
)

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
        res_path_root="results/""$ablation""/waymo/" ###
        echo "res_path_root: "$res_path_root

        res_path="$res_path_root""$scene""/metadata.json"
        echo "res_path: "$res_path

        python eval/eval.py --resfile $res_path
        echo "============================"
    done
done

