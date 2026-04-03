import os
import time
import glob
import argparse
from termcolor import colored

import numpy as np
import torch
from tqdm import tqdm
from threading import Thread
import json
import cv2
import copy

from vggt_slam import slam_utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT

from config.config import load_config, config

from scene.keyframe import Keyframe
from scene.scene_model import SceneModel

from gaussianviewer import GaussianViewer
from webviewer.webviewer import WebViewer
from graphdecoviewer.types import ViewerMode
from utils import increment_runtime, get_vggt_image_size


###########################################
# run one scene at a time
# run type: see config/base.yaml [save_path_parent]

# python run.py --config config/replica/room2.yaml --save_path_parent "none" --viewer_mode "local"
# python run.py --config config/tum/desk.yaml --save_path_parent "none" --viewer_mode "local"
# python run.py --config config/waymo/106762.yaml --save_path_parent "none" --viewer_mode "local"
###########################################



def test():
    ###### load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save_path_parent", default="none")  ######
    args = parser.parse_args()
    
    load_config(args.config)
    print(f'Using config: {args.config}')
    
    config['dataset']['save_path_parent'] = args.save_path_parent
    print(f'save_path_parent: {args.save_path_parent}')


def main():
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    
    ###### load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save_path_parent", default="none")  ######
    parser.add_argument("--viewer_mode", default="none")  ######
    args = parser.parse_args()
    
    load_config(args.config)
    print(f'Using config: {args.config}')
    
    config['dataset']['save_path_parent'] = args.save_path_parent
    config['mapping']['viewer_mode'] = args.viewer_mode
    print(f'save_path_parent: {args.save_path_parent}')
    print(f'viewer_mode: {args.viewer_mode}')
    
    # ====================================================
    ###### ablation ######
    save_path_parent = config['dataset']['save_path_parent']
    if 'wo_lc' in save_path_parent:
        config['tracking']['enable_loop_closure'] = False
    if 'wo_depthprior' in save_path_parent:
        config['mapping']['depth_loss_weight_init'] = 0.0
    # ====================================================
       
    
    # prevent OOM
    use_optical_flow_downsample = config['tracking']['use_optical_flow_downsample']
    if not use_optical_flow_downsample:
        config['tracking']['submap_size'] = 4
        config['tracking']['max_loops'] = 1
    
    # save config
    save_path_parent = config['dataset']['save_path_parent']
    save_path = config['dataset']['save_path']
    if save_path_parent != "none":
        save_path = os.path.join(save_path_parent, save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        config_json_path = os.path.join(save_path, "config.json")
        with open(config_json_path, "w") as f:
            json.dump(config, f, indent=4)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ###### VGGT solver
    solver = Solver(
        init_conf_threshold = config['tracking']['conf_threshold'],
        use_point_map       = config['tracking']['use_point_map'],
        use_sim3            = config['tracking']['use_sim3'],
        enable_loop_closure = config['tracking']['enable_loop_closure'],
        solver_vis          = config['tracking']['solver_vis'],
    )

    print("Initializing and loading VGGT model...")
    vggt_model = VGGT()
    model_path = "/home/grl/GeoGS_SLAM/checkpoints/vggt_model_1B.pt"
    vggt_model.load_state_dict(torch.load(model_path))
    vggt_model.eval()
    vggt_model = vggt_model.to(device)
    print(f'Loaded VGGT model from {model_path}!')


    ###### load image dataset
    print(colored(
        f"Loading images from {config['dataset']['image_folder']}...",
        "green"
    ))
    image_paths = [
        f for f in glob.glob(os.path.join(config['dataset']['image_folder'], "*")) 
        if "depth" not in os.path.basename(f).lower() 
        and "txt" not in os.path.basename(f).lower() 
        and "db" not in os.path.basename(f).lower()
        and "json" not in os.path.basename(f).lower()
    ]
    image_paths = slam_utils.sort_images_by_number(image_paths)
    
    downsample_factor = config['dataset']['downsample_factor']
    if downsample_factor > 0:
        image_paths = slam_utils.downsample_images(
            image_paths, 
            downsample_factor
        )
    print(colored(
        f"Found {len(image_paths)} images", 
        "green"
    ))

    # resized images size for the whole pipeline (VGGT inference, GS)
    height, width = get_vggt_image_size(image_paths[0:1])
    print(f'height: {height}, width: {width}')

    ###### Gaussian scene model
    scene_model = SceneModel(
        width=width, 
        height=height,
        config=config['mapping'],
        save_path_parent=config['dataset']['save_path_parent']
    )

    ###### Initialize the viewer
    viewer_mode = config['mapping']['viewer_mode']
    ip = config['mapping']['ip']
    port = config['mapping']['port']
    if viewer_mode in ["server", "local"]:
        viewer_mode = ViewerMode.SERVER if viewer_mode == "server" else ViewerMode.LOCAL
        
        viewer = GaussianViewer.from_scene_model(scene_model, viewer_mode)
        viewer_thd = Thread(target=viewer.run, args=(ip, port), daemon=True)
        viewer_thd.start()
        viewer.throttling = True  # Enable throttling when training
    # elif args.viewer_mode == "web":
    #     ...


    image_paths_subset = []
    n_keyframes = 0
    mapped_frame_ids = []
    metrics = {}
    pbar = tqdm(range(0, len(image_paths)))
    recon_start_time = time.time()
    
    ###### main loop
    for frameID in pbar:
        torch.cuda.empty_cache()
        
        image_path = image_paths[frameID]
        
        ###### keyframe selection with LK optical flow
        if use_optical_flow_downsample:
            img = cv2.imread(image_path)
            
            # ------------------
            time_start = time.time()  ###
            
            enough_disparity = solver.flow_tracker.compute_disparity(
                img, 
                config['tracking']['min_disparity'],
                config['tracking']['vis_flow']
            )
            if enough_disparity:
                image_paths_subset.append(image_path)
                
                time_end = time.time()  ###
                runtime = (time_end - time_start) * 1000  ### ms
                # print(colored(
                #     f'runtime [keyframe decision]: {runtime:.1f} [ms]',
                #     'red', 'on_white', ['bold']
                # ))  ###
                # ------------------
        else:
            image_paths_subset.append(image_path)


        ###### Run submap processing if enough images are collected or if it's the last group of images.
        subset_ready = (
            len(image_paths_subset) == config['tracking']['submap_size']
                                     + config['tracking']['overlap_size'] 
            or image_path == image_paths[-1]
        )
        
        if subset_ready:
            ###### process the current submap
            print(colored("image_paths_subset: ", "green"))
            print([os.path.basename(f) for f in image_paths_subset])
            
            # ===================================================
            ###### VGGT-based pose & geometry estimation
            predictions = solver.run_predictions(
                image_names   = image_paths_subset, 
                model         = vggt_model, 
                max_loops     = config['tracking']['max_loops'],
                min_sim_thres = config['tracking']['min_sim_thres'],
            )
            
            solver.add_points(predictions)
            
            
            
            graph_before_opt = copy.deepcopy(solver.graph)
            
            # -------***--------
            time_start = time.time()  ###
            
            solver.graph.optimize()
            
            time_end = time.time()  ###
            runtime = (time_end - time_start) * 1000  ### ms
            # -------***--------
            
            solver.map.update_submap_homographies(solver.graph)

            loop_detected = (
                config['tracking']['enable_loop_closure'] 
                and len(predictions["detected_loops"]) > 0
            )
            
            if config['tracking']['solver_vis']:
                if config['tracking']['vis_map']:
                    if loop_detected:
                        solver.update_all_submap_vis()
                    else:
                        solver.update_latest_submap_vis()
            
            # Reset for next submap, retain overlapping images
            image_paths_subset = image_paths_subset[-config['tracking']['overlap_size'] : ]
            # ===================================================
            
            if loop_detected:
                # -------***--------
                # print(colored(
                #     f'runtime [PGO]: {runtime:.1f} [ms]',
                #     (255, 0, 255), 'on_white', ['bold']
                # ))  ###
                # -------***--------
                
                # ------------------
                time_start = time.time()  ###
                
                scene_model.adjust_for_loop_closure(graph_before_opt, solver.graph)
                
                time_end = time.time()  ###
                runtime = (time_end - time_start) * 1000  ### ms
                # print(colored(
                #     f'runtime [global adjustment]: {runtime:.1f} [ms]',
                #     'yellow', 'on_white', ['underline']
                # ))  ###
                # ------------------
                
                
            ###### get keyframes from the current submap
            submap       = solver.map.get_latest_submap()
            submap_id    = submap.get_id()
            num_kfs      = submap.get_len_frames()
            frame_ids    = submap.get_frame_ids()
            frames       = submap.get_all_frames(ignore_loop_closure_frames=True)
            # pointmaps    = submap.get_all_pointmaps(ignore_loop_closure_frames=True).astype(np.float32)
            depths, conf = submap.get_all_depths_confs(ignore_loop_closure_frames=True)
            poses        = submap.get_all_poses_world2(ignore_loop_closure_frames=True).astype(np.float32)
            
            # get focal length
            if submap_id == 0:
                intrinsics = submap.get_all_intrinsics(ignore_loop_closure_frames=True)
                intrinsic = intrinsics[0]
                f = torch.tensor(
                    (intrinsic[0, 0] + intrinsic[1, 1]) / 2.0
                )
            
            print("Adding kf to Gaussian map and optimize...")
            for i in range(num_kfs):
                if frame_ids[i] in mapped_frame_ids:
                    continue
                
                ###### keyframing
                #####################################################
                # ablation: wo_poseprior, w/o pose prior for Gaussian init/opt, instead using last kf's pose
                if 'wo_poseprior' in save_path_parent:
                    if n_keyframes == 0:
                        Rt = np.linalg.inv(poses[i])  # eye
                    else:
                        Rt = scene_model.keyframes[-1].get_Rt()
                        
                    keyframe = Keyframe(
                        submap_id = submap_id,
                        frame_id = frame_ids[i],
                        index    = n_keyframes,
                        image    = frames[i],
                        # pointmap   = torch.from_numpy(pointmaps[i]).cuda(),
                        depth      = depths[i],
                        depth_conf = conf[i],
                        Rt         = Rt,
                        f          = f,
                        config     = config['mapping']
                    )
                else:
                    keyframe = Keyframe(
                        submap_id = submap_id,
                        frame_id = frame_ids[i],
                        index    = n_keyframes,
                        image    = frames[i],
                        # pointmap   = torch.from_numpy(pointmaps[i]).cuda(),
                        depth      = depths[i],
                        depth_conf = conf[i],
                        Rt         = np.linalg.inv(poses[i]),
                        f          = f,
                        config     = config['mapping']
                    )
                #####################################################
                
                n_keyframes += 1
                mapped_frame_ids.append(frame_ids[i])
                
                ###### add keyframes, add Gaussians, and optimization
                scene_model.add_keyframe(
                    keyframe=keyframe, 
                    f=f if keyframe.index == 0 else None
                )
                
                # ------------------
                time_start = time.time()  ###
                
                scene_model.add_new_gaussians()
                
                time_end = time.time()  ###
                runtime = (time_end - time_start) * 1000  ### ms
                # print(colored(
                #     f'runtime [primitive sampling per frame]: {runtime:.1f} [ms]',
                #     'cyan', 'on_white', ['bold']
                # ))  ###
                # ------------------
                
                
                # ------------------
                time_start = time.time()  ###
                
                scene_model.optimization_loop(config['mapping']['num_iterations'])
                
                time_end = time.time()  ###
                runtime = (time_end - time_start) * 1000  ### ms
                # print(colored(
                #     f'runtime [joint opt per frame]: {runtime:.1f} [ms]',
                #     'blue', 'on_white', ['bold']
                # ))  ###
                # ------------------
                
                
                ##############################
                # scene_model.place_anchor_if_needed()  
                ##############################
                
                if keyframe.index == 0:
                    if viewer_mode not in ["none", "web"]:
                        viewer.reset_intrinsics("point_view")
                
                
                ###### Intermediate evaluation
                test_frequency = config['mapping']['test_frequency']
                if test_frequency > 0 and n_keyframes % test_frequency == 0:
                    metrics = scene_model.evaluate(
                        eval_poses=False, 
                        with_LPIPS=False, 
                        all_kfs=True,  #False
                    )
                    
                ###### Display optimization progress and metrics
                bar_postfix = []
                for key, value in metrics.items():
                    bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
            
                bar_postfix += [
                    # f"\033[36mFocal:{focal:.1f}",
                    f"\033[36mKeyframes:{n_keyframes}\033[0m",
                    f"\033[36mGaussians:{scene_model.n_active_gaussians}\033[0m",
                    f"\033[36mAnchors:{len(scene_model.anchors)}\033[0m",
                ]
                pbar.set_postfix_str(",".join(bar_postfix), refresh=False)
            
            del graph_before_opt
            del frames, depths, conf, poses
            torch.cuda.empty_cache()
            
            
    recon_time = time.time() - recon_start_time
        
    print("Total number of submaps: ", solver.map.get_num_submaps())
    print("Total number of loop closures: ", solver.graph.get_num_loops())
    if config['tracking']['solver_vis']:
        if not config['tracking']['vis_map']:
            # just show the map after all submaps have been processed
            solver.update_all_submap_vis()
    
    
    # Set to inference mode so that the model can be rendered properly
    # scene_model.enable_inference_mode()
    
    ###### Save the scene model and metrics
    if save_path_parent != "none":
        print(f"Saving reconstruction to: {save_path}...")
        metrics = scene_model.save(
            path=save_path, 
            reconstruction_time=recon_time, 
            n_frames=len(image_paths)
        )
        # =============================================
        ### save render results
        scene_model.save_render_results(path=save_path)
        
        ### save traj
        scene_model.save_traj(path=save_path)
        # =============================================
        
    else:
        print(f"Evaluating reconstruction without saving...")
        metrics = scene_model.evaluate(
            eval_poses=False, 
            with_LPIPS=True, 
            all_kfs=True,
        )
        
    metrics_str = ", ".join(
        f"{metric}: {value:.3f}"
        if isinstance(value, float)
        else f"{metric}: {value}"
        for metric, value in metrics.items()
    )
    print(colored(
        metrics_str, 
        'red', 'on_cyan', ['bold']
    ))


    if viewer_mode != "none":
        if viewer_mode == "web":
            while True:
                time.sleep(1)
        else:
            viewer.throttling = False  # Disable throttling when done training
            # Loop to keep the viewer alive
            while viewer.running:
                time.sleep(1)


if __name__ == "__main__":
    main()
    # test()
    