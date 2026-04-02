import json
import numpy as np
import argparse
from evo.core.trajectory import PosePath3D
from evo.core import metrics
from evo.tools import plot
from matplotlib import pyplot as plt
import os
import re
import trimesh

###########################################
# evaluate one scene at a time

# python eval/eval.py --resfile results/full/replica/office0/metadata.json
# python eval/eval.py --resfile results/full/tum/360/metadata.json
# python eval/eval.py --resfile results/full/waymo/13476/metadata.json
###########################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resfile", default="")
    args = parser.parse_args()

    resfile = args.resfile
    print(f'Evaluating resfile: {resfile}')
    
    ############ Result poses
    with open(resfile, 'r') as f:
        metadata = json.load(f)
    # print(f"metadata keys: {list(metadata.keys())}")
    keyframes = metadata['keyframes']
    # print(f"Number of keyframes: {len(keyframes)}")

    poses_est = []
    stamps_est = []
    for kf in keyframes:
        Rt = np.array(kf['Rt'])
        pose = np.linalg.inv(Rt)
        poses_est.append(pose)
        
        stamp = kf['info']['frame_id']  # float
        stamps_est.append(stamp)
    # print(len(poses_est))


    ############ dataset type
    if 'replica' in resfile:
        dataset_type = 'replica'
    elif 'tum' in resfile:      
        dataset_type = 'tum'
    elif 'waymo' in resfile:      
        dataset_type = 'waymo'
    print(f'Evaluating dataset_type: {dataset_type}')
    

    ############ GT poses Replica
    if dataset_type == 'replica':
        scene_match = re.search(r'replica/([^/]+)/', resfile)
        if scene_match:
            scene_name = scene_match.group(1)  # office0
            gt_path = f'/home/grl/datasets/replica/{scene_name}/traj.txt'
            print(f'Evaluating scene_name: {scene_name}')
        else:
            raise ValueError(f"Could not extract scene name from resfile: {resfile}")
        
        poses_gt = []
        with open(gt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            poses_gt.append(pose)
        # print(f'len(poses_gt): {len(poses_gt)}')

        poses_gt_selected = []
        for idx in stamps_est:
            idx = int(idx)
            poses_gt_selected.append(poses_gt[idx])
        # print(f'len(poses_gt_selected): {len(poses_gt_selected)}')
        
    ############ GT poses TUM
    elif dataset_type == 'tum':
        scene_match = re.search(r'tum/([^/]+)/', resfile)
        if scene_match:
            scene_name = scene_match.group(1)  # 360
            gt_path = f'/home/grl/datasets/tum/rgbd_dataset_freiburg1_{scene_name}/groundtruth.txt'
            print(f'Evaluating scene_name: {scene_name}')
        else:
            raise ValueError(f"Could not extract scene name from resfile: {resfile}")
        
        gt_data = np.loadtxt(gt_path, delimiter=" ", dtype=np.str_, skiprows=3)
        pose_vecs = gt_data[:, 0:].astype(np.float64)
        # print(f'pose_vecs.shape: {pose_vecs.shape}')
        
        poses_gt = []
        stamps_gt = []
        for k in range(len(pose_vecs)):
            stamp = pose_vecs[k][0]
            stamps_gt.append(stamp)
            
            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            poses_gt.append(T)
        # print(len(poses_gt))

        stamps_gt_selected = []
        poses_gt_selected = []
        for stamp_est in stamps_est:
            idx = np.argmin(np.abs(np.array(stamps_gt) - stamp_est))
            stamps_gt_selected.append(stamps_gt[idx])
            poses_gt_selected.append(poses_gt[idx])
    
    ############ GT poses Waymo
    elif dataset_type == 'waymo':
        scene_match = re.search(r'waymo/([^/]+)/', resfile)
        if scene_match:
            scene_name = scene_match.group(1)  # 13476
            gt_folder = f'/home/grl/datasets/waymo/{scene_name}/FRONT/gt'
            print(f'Evaluating scene_name: {scene_name}')
        else:
            raise ValueError(f"Could not extract scene name from resfile: {resfile}")

        gt_files = [f for f in os.listdir(gt_folder)]
        gt_files = sorted(gt_files, key=lambda x: int(os.path.splitext(x)[0]))
        
        poses_gt = []
        for gt_file in gt_files:
            pose = np.loadtxt(os.path.join(gt_folder, gt_file))
            poses_gt.append(pose)
        # print(f'len(poses_gt): {len(poses_gt)}')
        
        poses_gt_selected = []
        for idx in stamps_est:
            idx = int(idx)
            poses_gt_selected.append(poses_gt[idx])
        # print(f'len(poses_gt_selected): {len(poses_gt_selected)}')



    ############ EVO
    traj_ref = PosePath3D(poses_se3=poses_gt_selected)
    traj_est = PosePath3D(poses_se3=poses_est)
    ## Align
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)
    traj_est_aligned = traj_est
    ## ATE RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    print(f"RMSE ATE (m): {ape_rmse}, scale: {s}")

    ## Plot
    # plot_mode = plot.PlotMode.xy
    plot_mode = plot.PlotMode.xyz

    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE (m): {ape_rmse:.3f}, scale: {s:.3f}")
    plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    # plt.show()
    fig_dir = os.path.dirname(resfile)
    fig_path = os.path.join(fig_dir, "evo_traj.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')


    ############ save all metrics
    # Load PSNR, SSIM, LPIPS from resfile
    psnr = metadata['PSNR']
    ssim = metadata['SSIM']
    lpips = metadata['LPIPS']

    # Prepare metrics dictionary
    metrics_dict = {
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips,
        'ATE': ape_rmse,
    }

    metrics_path = os.path.join(fig_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
