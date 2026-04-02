#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# from argparse import Namespace
import gc
import os
import json
import math
import threading
import time
import warnings
import cv2
import torch
import torch.nn.functional as F
import numpy as np

import lpips
from fused_ssim import fused_ssim
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)
from simple_knn._C import distIndex2

# from poses.feature_detector import DescribedKeypoints
# from poses.matcher import Matcher
# from poses.guided_mvs import GuidedMVS

from scene.optimizers import SparseGaussianAdam
from scene.keyframe import Keyframe
from scene.anchor import Anchor

from utils import (
    RGB2SH,
    depth2points,
    focal2fov,
    get_dog_norm, #get_lapla_norm,
    getProjectionMatrix,
    inverse_sigmoid,
    align_poses,
    make_torch_sampler,
    psnr,
    rotation_distance,
    unitquat2mtx,
    mtx2unitquat,
    quat_mul,
)
from dataloaders.read_write_model import write_model


class SceneModel:
    """
    Scene Model class that contains the scene's Gaussians, anchors, keyframes, and methods for rendering and optimization.
    """
    def __init__(
        self,
        width: int,
        height: int,
        config: dict,
        save_path_parent: str = 'none',
        inference_mode: bool = False,
    ):
        self.save_path_parent = save_path_parent
        
        self.width = width
        self.height = height
        self.centre = torch.tensor(
            [(width - 1) / 2, (height - 1) / 2], 
            device="cuda"
        )
        self.anchor_overlap = config['anchor_overlap']
        self.optimization_thread = None

        try:
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            warnings.filterwarnings("ignore")
            
            # self.lpips = lpips.LPIPS(net="vgg").cuda()
            self.lpips = lpips.LPIPS(net="alex").cuda()
            
            sys.stdout = original_stdout
        except:
            self.lpips = None

        if not inference_mode:
            self.active_sh_degree = config['sh_degree']
            self.max_sh_degree = config['sh_degree']
            
            self.lambda_dssim = config['lambda_dssim']
            self.init_proba_scaler = config['init_proba_scaler']
            self.max_active_keyframes = config['max_active_keyframes']
            self.use_last_frame_proba = config['use_last_frame_proba']
            
            self.active_frames_cpu = []
            self.active_frames_gpu = []
            
            self.lr_dict = {
                "xyz": {
                    "lr_init": config['position_lr_init'],
                    "lr_decay": config['position_lr_decay'],
                }
            }

            ## Initialize Gaussian parameters
            self.gaussian_params = {
                "xyz": {
                    "val": torch.empty(0, 3, device="cuda"),
                    "lr": config['position_lr_init'],
                },
                "f_dc": {
                    "val": torch.empty(0, 1, 3, device="cuda"),
                    "lr": config['feature_lr'],
                },
                "f_rest": {
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device="cuda",
                    ),
                    "lr": config['feature_lr'] / 20.0,
                },
                "scaling": {
                    "val": torch.empty(0, 3, device="cuda"),
                    "lr": config['scaling_lr'],
                },
                "rotation": {
                    "val": torch.empty(0, 4, device="cuda"),
                    "lr": config['rotation_lr'],
                },
                "opacity": {
                    "val": torch.empty(0, 1, device="cuda"),
                    "lr": config['opacity_lr'],
                },
            }
            self.submap_ids = torch.empty(0, dtype=torch.int32, device="cuda")  ######
            
            self.active_anchor = Anchor(self.gaussian_params)
            self.anchors = [self.active_anchor]
            
            ## Initialize optimizer
            self.reset_optimizer()

        self.keyframes: list[Keyframe] = []
        self.anchor_weights = [1.0]
        self.f = 0.7 * width
        self.init_intrinsics()
        
        self.n_kept_frames = config['n_kept_frames']

        self.approx_cam_centres = None
        
        self.gt_Rts = torch.empty(0, 4, 4, device="cuda")
        self.gt_Rts_mask = torch.empty(0, device="cuda", dtype=bool)
        self.gt_f = self.f
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")
        self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)
        
        # self.sorted_frame_indices = None
        self.last_trained_id = 0
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)
        self.lock = threading.Lock()
        self.inference_mode = inference_mode

        ## Initialize helpers for Gaussian initialization
        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[
            0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5
        ] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()

        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), 
                    torch.arange(0, height), 
                    indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .cuda()
        )  # [H, W, 2]


    def reset_optimizer(self):
        for key in self.gaussian_params:
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
                
        self.optimizer = SparseGaussianAdam(
            self.submap_ids,
            self.gaussian_params, 
            (0.5, 0.99), 
            lr_dict=self.lr_dict
        )


    @property
    def xyz(self):
        return self.gaussian_params["xyz"]["val"]

    @property
    def f_dc(self):
        return self.gaussian_params["f_dc"]["val"]

    @property
    def f_rest(self):
        return self.gaussian_params["f_rest"]["val"]

    @property
    def scaling(self):
        return torch.exp(self.gaussian_params["scaling"]["val"])

    @property
    def rotation(self):
        return F.normalize(self.gaussian_params["rotation"]["val"])

    @property
    def opacity(self):
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def n_active_gaussians(self):
        return self.xyz.shape[0]


    # @classmethod
    # def from_scene(
    #     cls, 
    #     scene_dir: str, 
    #     # args
    #     config
    # ):
    #     with open(os.path.join(scene_dir, "metadata.json")) as f:
    #         metadata = json.load(f)

    #     width = metadata["config"]["width"]
    #     height = metadata["config"]["height"]
        
    #     # scene_model = cls(width, height, args, inference_mode=True)
    #     scene_model = cls(width, height, config, inference_mode=True)
        
    #     scene_model.active_sh_degree = metadata["config"]["sh_degree"]
    #     scene_model.max_sh_degree = metadata["config"]["sh_degree"]

    #     # Load anchors
    #     scene_model.anchors = []
    #     for i in range(len(metadata["anchors"])):
    #         scene_model.anchors.append(
    #             Anchor.from_ply(
    #                 os.path.join(scene_dir, "point_clouds", f"anchor_{i}.ply"),
    #                 torch.tensor(metadata["anchors"][i]["position"]),
    #                 metadata["config"]["sh_degree"],
    #             )
    #         )
    #     scene_model.active_anchor = scene_model.anchors[0]

    #     # Load keyframes
    #     for i in range(len(metadata["keyframes"])):
    #         keyframe = Keyframe.from_json(metadata["keyframes"][i], i, width, height)
    #         scene_model.add_keyframe(keyframe)

    #     return scene_model


    @property
    def first_active_frame(self):
        return self.active_anchor.keyframe_ids[0]

    @property
    def last_active_frame(self):
        return self.active_anchor.keyframe_ids[-1]

    @property
    def n_active_keyframes(self):
        return self.last_active_frame - self.first_active_frame + 1


    def optimization_step(self, finetuning=False):
        if self.xyz.shape[0] == 0:
            return
        
        # Select which keyframe to train on
        # We train on the latest keyframe with self.use_last_frame_proba probability or a random keyframe otherwise
        if (
            np.random.rand() > self.use_last_frame_proba
            or self.last_trained_id == -1
            or finetuning
        ):
            keyframe_id = np.random.choice(self.active_frames_gpu)
        else:
            keyframe_id = -1
            
        keyframe = self.keyframes[keyframe_id]
        lvl = keyframe.pyr_lvl

        # Zero gradients
        keyframe.zero_grad()
        self.optimizer.zero_grad()

        # Render image and depth
        render_pkg = self.render_from_id(
            keyframe_id, 
            pyr_lvl=lvl,
            bg=torch.rand(3, device="cuda")
        )
        rendered_image = render_pkg["render"]
        rendered_depth = 1.0 / render_pkg["invdepth"][0].clamp_min(1e-8)

        gt_image = keyframe.image_pyr[lvl]
        gt_depth = keyframe.get_depth(lvl)

        # Losses
        l1_loss = (rendered_image - gt_image).abs().mean()
        ssim_loss = 1 - fused_ssim(rendered_image[None], gt_image[None])
        depth_loss = (rendered_depth - gt_depth).abs().mean()
        
        loss = (
            self.lambda_dssim * ssim_loss
            + (1 - self.lambda_dssim) * l1_loss
            + keyframe.depth_loss_weight * depth_loss
        )
        loss.backward()

        # Optimizers
        with torch.no_grad():
            # Pose optimization
            #####################################################  
            if 'wo_poseopt' in self.save_path_parent:
                pass
            else:
                keyframe.step()  # if commented: ablation: wo_poseopt, w/o pose optimization using GS
            #####################################################  
  
            # Skip the scene optimization if the current keyframe is a test keyframe
            # if not keyframe.info["is_test"]:
            if not keyframe.is_test:
                # Scene Gaussian optimization
                self.optimizer.step(
                    render_pkg["visibility_filter"], 
                    render_pkg["radii"].shape[0]
                )

            # keyframe.latest_invdepth = render_pkg["invdepth"].detach()
            # keyframe.latest_depth = 1 / render_pkg["invdepth"].detach()[0].clamp_min(1e-8)

        self.valid_Rt_cache[keyframe_id] = False
        self.last_trained_id = keyframe_id


    def optimization_loop(self, n_iters: int, run_until_interupt: bool = False):
        """
        Runs at least n_iters optimization steps.
        If run_until_interupt, also runs until join_optimization_thread is called 
        (Useful to run the optimization until the next keyframe is added in streaming mode).
        """
        # print("Running Gaussian map optimization...")
        self.interupt_optimization = False
        i = 0
        while i < n_iters or (run_until_interupt and not self.interupt_optimization):
            torch.cuda.empty_cache()
            self.optimization_step()
            i += 1
        
        
    def join_optimization_thread(self):
        """
        Interupts the optimization loop and waits for the thread to finish.
        """
        if self.optimization_thread is not None:
            self.interupt_optimization = True
            self.optimization_thread.join()
            self.optimization_thread = None
    
    def optimize_async(self, n_iters: int):
        """
        Starts an optimization thread that runs at least n_iters optimization steps.
        """
        self.join_optimization_thread()
        self.optimization_thread = threading.Thread(
            target=self.optimization_loop, args=(n_iters, True)
        )
        self.optimization_thread.start()


    @torch.no_grad()
    def harmonize_test_exposure(self):
        """Harmonizes the exposure matrices of test keyframes by averaging the exposure of the previous and next keyframes."""
        for index, keyframe in enumerate(self.keyframes):
            # if keyframe.info["is_test"]:
            if keyframe.is_test:
                idxm = index - 1 if index != 0 else 1
                idxp = (
                    index + 1
                    if index != len(self.keyframes) - 1
                    else len(self.keyframes) - 2
                )
                keyframe.exposure = (
                    self.keyframes[idxm].exposure + self.keyframes[idxp].exposure
                ) / 2


    @torch.no_grad()
    def evaluate(
        self, 
        eval_poses=False, 
        with_LPIPS=False, 
        all_kfs=False
    ):
        # Make sure test keyframes have similar exposure matrices compared to their neighbors
        self.harmonize_test_exposure()

        # Compute image quality metrics
        metrics = {"PSNR": 0, "SSIM": 0}
        if with_LPIPS:
            metrics["LPIPS"] = 0
            
        n_test_frames = 0
        start_index = 0 if all_kfs else self.active_anchor.keyframe_ids[0]
        
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            # if keyframe.info["is_test"]:
            # if index % 10 == 0:
            if True:
                gt_image = keyframe.image_pyr[0].cuda()
                
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = render_pkg["render"]
                
                mask = torch.ones_like(image[:1] > 0)
                mask = mask.expand_as(image)
                
                image = image * mask
                gt_image = gt_image * mask
                
                metrics["PSNR"] += psnr(
                    image[mask], 
                    gt_image[mask]
                )
                
                metrics["SSIM"] += fused_ssim(
                    image[None], 
                    gt_image[None], 
                    train=False
                ).item()
                
                # print(f"image min: {image.min().item()}, max: {image.max().item()}")
                # print(f"gt_image min: {gt_image.min().item()}, max: {gt_image.max().item()}")
                # gt_image min: 0.0, max: 1.0
                # image min: 0.0, max: 1.0
                
                if with_LPIPS and self.lpips is not None:
                    metrics["LPIPS"] += self.lpips(
                        image[None],
                        gt_image[None],
                        ########################################
                        normalize=True,  # since input is [0,1]
                        ########################################  
                    ).item()
                    
                n_test_frames += 1

        if n_test_frames > 0:
            for metric in metrics:
                metrics[metric] /= n_test_frames
        else:
            metrics = {}

        # Compute pose errors
        if eval_poses:
            Rts = self.get_Rts()
            gt_Rts = self.get_gt_Rts(align=False)
            if len(Rts) == len(gt_Rts):
                Rts_aligned = torch.linalg.inv(align_poses(Rts, gt_Rts))
                gt_Rts = torch.linalg.inv(gt_Rts)
                
                R_error = rotation_distance(Rts_aligned[:, :3, :3], gt_Rts[:, :3, :3])
                t_error = (Rts_aligned[:, :3, 3] - gt_Rts[:, :3, 3]).norm(dim=-1)

                metrics["R°"] = R_error.mean().item() * 180 / math.pi
                metrics["t"] = t_error.mean().item()

        return metrics


    @torch.no_grad()
    def save_test_frames(self, out_dir):
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)
        for keyframe in self.keyframes:
            # if keyframe.info["is_test"]:
            if keyframe.is_test:
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = torch.clamp(render_pkg["render"], 0, 1) * 255
                image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                is_jpeg = os.path.splitext(keyframe.info["name"])[-1].lower() in [
                    ".jpg",
                    ".jpeg",
                ]
                write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
                cv2.imwrite(
                    os.path.join(out_dir, keyframe.info["name"]), image, write_flag
                )


    def render_from_id(
        self,
        keyframe_id,
        pyr_lvl=0,
        scaling_modifier=1,
        bg=torch.zeros(3, device="cuda"),
    ):
        """
        Render the scene from a given keyframe id at a specified resolution level (pyr_lvl).
        Applies the exposure matrix of the keyframe to the rendered image.
        """
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt().transpose(0, 1)
        
        scale = 2 ** pyr_lvl
        width, height = self.width // scale, self.height // scale
        
        render_pkg = self.render(
            width, 
            height, 
            view_matrix, 
            scaling_modifier, 
            bg
        )
        
        render_pkg["render"] = (
            keyframe.exposure[:3, :3] @ render_pkg["render"].view(3, -1)
        ) + keyframe.exposure[:3, 3, None]
        render_pkg["render"] = render_pkg["render"].clamp(0, 1).view(3, height, width)
        
        return render_pkg


    def render(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        scaling_modifier: float,
        bg: torch.Tensor = torch.zeros(3, device="cuda"),
        top_view: bool = False,
        fov_x: float = None,
        fov_y: float = None,
    ):
        cam_centre = view_matrix.detach().inverse()[3, :3]

        # Use the scene's intrinsic parameters if not provided
        if fov_x is None and fov_y is None:
            tanfovx, tanfovy = self.tanfovx, self.tanfovy
            projection_matrix = self.projection_matrix
        # Use the provided FOV values
        elif fov_x is not None and fov_y is not None:
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y)
                .transpose(0, 1)
                .cuda()
            )
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")

        raster_settings = GaussianRasterizationSettings(
            height,
            width,
            tanfovx,
            tanfovy,
            bg,
            1 if top_view else scaling_modifier,
            projection_matrix,
            self.active_sh_degree,
            cam_centre,
            False,
            False,
        )
        rasterizer = GaussianRasterizer(raster_settings)
        
        with self.lock:
            # Load and blend anchors if in inference mode 
            if self.inference_mode and not top_view:
                self.gaussian_params, self.anchor_weights = Anchor.blend(
                    cam_centre, 
                    self.anchors, 
                    self.anchor_overlap
                )
                
            screenspace_points = torch.zeros_like(self.xyz, requires_grad=True)
            if self.xyz.shape[0] > 0:
                # Set constant scaling and opacity to visualize the Gaussians' positions in the top view
                if top_view:
                    scaling = torch.ones_like(self.scaling) * scaling_modifier
                    opacity = torch.ones_like(self.opacity)
                else:
                    scaling = self.scaling
                    opacity = self.opacity
                    
                color, invdepth, mainGaussID, radii = rasterizer(
                    self.xyz,
                    screenspace_points,
                    opacity,
                    self.f_dc,
                    self.f_rest,
                    scaling,
                    self.rotation,
                    view_matrix,
                )
            else:
                # If no Gaussians are present, return empty tensors
                color = torch.zeros(
                    3, height, width, device="cuda"
                )
                invdepth = torch.zeros(
                    1, height, width, device="cuda"
                )
                mainGaussID = torch.zeros(
                    1, height, width, device="cuda", dtype=torch.int32
                )
                radii = torch.zeros(
                    1, height, width, device="cuda"
                )
                
        return {
            "render": color,
            "invdepth": invdepth,
            "mainGaussID": mainGaussID,
            "radii": radii,
            "visibility_filter": radii > 0,
            "screenspace_points": screenspace_points,
        }


    def get_closest_by_cam(self, cam_centre, k=3):
        closest_anchors = []
        closest_anchors_ids = []
        offset = 0
        approx_cam_centres = self.approx_cam_centres.clone()
        for l in range(min(k, len(self.anchors))):
            if approx_cam_centres.shape[0] == 0:
                break
            dists = torch.linalg.norm(
                approx_cam_centres - cam_centre[None], 
                dim=-1
            )
            min_dist, min_id = torch.min(dists, dim=0)

            if min_dist < 1e9:
                for anchor_id, anchor in enumerate(self.anchors):
                    if min_id in anchor.keyframe_ids:
                        closest_anchors.append(anchor)
                        closest_anchors_ids.append(anchor_id)
                        approx_cam_centres[
                            anchor.keyframe_ids[0] : anchor.keyframe_ids[-1] + 1
                        ] = 1e9
                        break

        return closest_anchors, closest_anchors_ids


    def get_Rts(self):
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    def get_gt_Rts(self, align):
        n_poses = min(self.gt_Rts_mask.shape[0], 
                      self.cached_Rts.shape[0])
        if align and n_poses > 0:
            Rts = self.get_Rts()[:n_poses][self.gt_Rts_mask[:n_poses]]
            return align_poses(self.gt_Rts[: len(Rts)], Rts)
        else:
            return self.gt_Rts


    # for pruning Gaussians
    def make_dummy_ext_tensor(self):
        return {
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
        }


    def reset(self, keyframe_id: int = -1):
        """Remove the Gaussians that are visible in the given keyframe."""
        valid_mask = self.opacity[:, 0] > 0.05
        render_pkg = self.render_from_id(keyframe_id)
        valid_mask[render_pkg["visibility_filter"]] = False
        
        self.submap_ids = self.optimizer.add_and_prune(
            None,
            self.make_dummy_ext_tensor(), 
            valid_mask
        )


    def init_intrinsics(self):
        self.FoVx = focal2fov(self.f, self.width)
        self.FoVy = focal2fov(self.f, self.height)
        
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        
        self.projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )


    def add_keyframe(
        self, 
        keyframe: Keyframe, 
        f=None
    ):
        """Add a keyframe to the scene, add and prune Gaussians"""

        # Make sure training is not running 
        self.join_optimization_thread()

        ## Add the keyframe and update the indices (sorted by distance to last keyframe)
        self.keyframes.append(keyframe)
        
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [
                    self.approx_cam_centres, 
                    keyframe.approx_centre[None]
                ], 
                dim=0
            )
            
        # dist_to_last = torch.linalg.vector_norm(
        #     self.approx_cam_centres - keyframe.approx_centre[None], 
        #     dim=-1
        # )
        # self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        ## Update intrinsics
        if f is not None:
            self.f = f.item()
            self.init_intrinsics()

        ## Update cached Rts for the viewer
        self.cached_Rts = torch.cat(
            [
                self.cached_Rts, 
                keyframe.get_Rt().unsqueeze(0)
            ],
            dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [
                self.valid_Rt_cache, 
                torch.ones(1, device="cuda", dtype=torch.bool)
            ],
            dim=0
        )
        
        # gt_pose = keyframe.info.get("Rt", None)
        gt_pose = None
        if gt_pose is not None:
            self.gt_Rts = torch.cat(
                [
                    self.gt_Rts, 
                    gt_pose.unsqueeze(0)
                ],
                dim=0
            )
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.Tensor([gt_pose is not None]).to(self.gt_Rts_mask),
            ],
            dim=0
        )
        
        # self.gt_f = keyframe.info.get("focal", self.f)
        self.gt_f = self.f

        if not self.inference_mode:
            ## Add keyframe to the active anchor
            self.active_anchor.add_keyframe(keyframe)
            self.active_frames_gpu.append(keyframe.index)

            ## Clear memory if there are many keyframes
            if len(self.active_frames_gpu) > self.max_active_keyframes:
                self.move_rand_keyframe_to_cpu()
                
                # Reshuffle the active keyframes and clear cache
                if len(self.active_frames_cpu) % 5 == 0:
                    self.move_rand_keyframe_to_cpu()
                    self.move_rand_keyframe_to_gpu()

                    gc.collect()
                    torch.cuda.empty_cache()


    @torch.no_grad()
    def add_new_gaussians(
        self, 
        keyframe_id: int = -1
    ):
        """Use the given keyframe to add new Gaussians to the scene model."""
        
        keyframe = self.keyframes[keyframe_id]

        ###### Skip if the keyframe is a test keyframe
        if keyframe.is_test:
            return

        ###### Get the pixel-wise probability to add a Gaussian
        img = keyframe.image_pyr[0]  # [3, H, W]
        dep = keyframe.get_depth(0)  # [1, H, W]
        
        init_proba_img = get_dog_norm(img, self.disc_kernel)  # eq. 1, [H, W]
        init_proba_dep = get_dog_norm(dep, self.disc_kernel)  # eq. 1, [H, W]
        init_proba = init_proba_img + init_proba_dep  # [H, W]

        #####################################################
        # ablation: wo_depthprior, w/o depth prior for Gaussian init/opt, instead using random depth map
        if 'wo_depthprior' in self.save_path_parent:
            # dep = torch.rand_like(dep)
            dep = 2 * torch.ones((1, img.shape[1], img.shape[2])).to(img.device)
            dep += torch.randn_like(dep) * 0.3
            
            init_proba = init_proba_img
        #####################################################

        ###### Compute the penalty based on the rendering from the new keyframe's point of view
        penalty = 0
        rendered_depth = None
        if self.xyz.shape[0] > 0:
            render_pkg = self.render_from_id(keyframe_id)
            render = render_pkg["render"]  # [3, H, W]
            rendered_depth = 1.0 / render_pkg["invdepth"][0].clamp_min(1e-8)  # [H, W]
            penalty = get_dog_norm(render, self.disc_kernel)  # eq. 2, [H, W]

        ###### Define which pixels should become Gaussians
        init_proba *= self.init_proba_scaler
        penalty *= self.init_proba_scaler
        
        #####################################################  
        # ablation: wo_sample, w/o pixle sampling using DoG for Gaussian init, instead using uniform sampling
        if 'wo_sample' in self.save_path_parent:
            uniform_downsample_factor = 64
            # Create a boolean mask where 1/uniform_downsample_factor of the values are True
            sample_mask = torch.zeros_like(init_proba, dtype=torch.bool)
            num_pixels = init_proba.numel()
            num_samples = max(1, num_pixels // uniform_downsample_factor)
            # Flatten indices, randomly select num_samples indices to set True
            flat_indices = torch.randperm(num_pixels)[:num_samples]
            sample_mask.view(-1)[flat_indices] = True
        else:
            sample_mask = torch.rand_like(init_proba) < (init_proba - penalty)  # eq. 3, [H, W]
        #####################################################
        
        sampled_uv = self.uv[sample_mask]  # [N0, 2]
        
        
        ###### Initialize positions
        u = sampled_uv.cpu()[:, 0].int()
        v = sampled_uv.cpu()[:, 1].int()
        samlped_depth = dep[0, v, u]  # [N0,]
        accurate_mask = torch.ones_like(samlped_depth, dtype=torch.bool)  # [N0,]
        
        #####################################################
        # filter low conf
        if 'wo_depthprior' in self.save_path_parent:
            valid_mask = (samlped_depth > 1e-6)  # ablation: wo_depthprior
        else:
            valid_mask = (keyframe.sample_conf(sampled_uv) > 0.5) * (samlped_depth > 1e-6)
        #####################################################
        
        sample_mask[sample_mask.clone()] = valid_mask
        samlped_depth = samlped_depth[valid_mask]
        sampled_uv = sampled_uv[valid_mask]
        accurate_mask = accurate_mask[valid_mask]

        ###### Remove Gaussians that are coarser than the newpoints
        if self.xyz.shape[0] > 0:
            main_gaussians_map = render_pkg["mainGaussID"]
            accurate_sample_mask = sample_mask.clone()
            accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
            selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]
            ids, counts = torch.unique(
                selected_main_gaussians[selected_main_gaussians >= 0],
                return_counts=True,
            )
            valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
            valid_gs_mask[ids] = counts < 10
            with self.lock:
                self.submap_ids = self.optimizer.add_and_prune(
                    None,
                    self.make_dummy_ext_tensor(),
                    valid_gs_mask
                )
            render_pkg = self.render_from_id(keyframe_id)
            rendered_depth = 1.0 / render_pkg["invdepth"][0].clamp_min(1e-8)

        ###### Check for occlusions
        if rendered_depth is not None:
            valid_mask = samlped_depth < rendered_depth[sample_mask]
            sample_mask[sample_mask.clone()] = valid_mask
            samlped_depth = samlped_depth[valid_mask]
            sampled_uv = sampled_uv[valid_mask]
            accurate_mask = accurate_mask[valid_mask]


        ###### Get the samples' 3D positions, new_pts: [N, 3]
        new_pts = depth2points(
            sampled_uv, 
            samlped_depth.unsqueeze(-1),
            self.f, 
            self.centre
        )  # from camera
        new_pts = (new_pts - keyframe.get_t()) @ keyframe.get_R()  # from world
        # print(f'new_pts.shape: {new_pts.shape}')
        

        ###### Initialize Colour, f_dc: [N, 1, 3]
        f_dc = img[:, sample_mask]
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))
        # print(f'f_dc.shape: {f_dc.shape}')
        

        ###### Initialize Scales, scales: [N, 3]
        sampled_init_proba = init_proba[sample_mask]
        
        # Expected distance to the nearest neighbour (eq. 4)
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.width / 10)
        # Scale by the distance to the camera centre
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(
            new_pts - keyframe.approx_centre[None], 
            dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)
        # print(f'scales.shape: {scales.shape}')
        

        ###### Initialize opacities, opacities: [N, 1]
        opacities = torch.ones(f_dc.shape[0], 1, device="cuda")
        # Lower inital opacity depending for innacurate points
        opacities[: sampled_uv.shape[0]] *= (
            0.07 * accurate_mask[..., None] + 
            0.02 * ~accurate_mask[..., None]
        )
        # High opacity for triangulated Gaussians
        opacities[sampled_uv.shape[0] :] *= 0.2
        opacities = inverse_sigmoid(opacities)
        # print(f'opacities.shape: {opacities.shape}')
        

        ###### Initialize SH, rotations as identity, f_rest: [N, 15, 3], rots: [N, 4]
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device="cuda",
        )
        
        rots = torch.zeros(f_dc.shape[0], 4, device="cuda")
        rots[:, 0] = 1
        # print(f'f_rest.shape: {f_rest.shape}')
        # print(f'rots.shape: {rots.shape}')


        ###### Get which Gaussians should be pruned
        if self.xyz.shape[0] > 0:
            # Only keep Gaussians with non neglectible opacity
            valid_gs_mask = self.opacity[:, 0] > 0.05

            # Discard huge Gaussians
            dist = torch.linalg.vector_norm(
                self.xyz - keyframe.approx_centre[None], 
                dim=-1
            )
            screen_size = self.f * self.scaling.max(dim=-1)[0] / dist
            valid_gs_mask *= (screen_size < 0.5 * self.width)
        else:
            valid_gs_mask = torch.ones(0, device="cuda", dtype=torch.bool)


        ###### Append the new Gaussians
        extension_tensors = {
            "xyz"     : new_pts,
            "f_dc"    : f_dc,
            "f_rest"  : f_rest,
            "opacity" : opacities,
            "scaling" : scales,
            "rotation": rots,
        }
        with self.lock:
            self.submap_ids = self.optimizer.add_and_prune(
                keyframe.submap_id,  ######
                extension_tensors, 
                valid_gs_mask
            )
        
        assert self.xyz.shape[0] == self.submap_ids.shape[0]
        # print(f'Gaussian num after adding kf {keyframe.index}: {self.xyz.shape[0]}')
        # print(f'self.submap_ids.shape: {self.submap_ids.shape}')



    def move_rand_keyframe_to_cpu(self):
        """Move a random keyframe to CPU memory"""
        frame_id = np.random.choice(
            self.active_frames_gpu[:-self.n_kept_frames]
        )
        self.keyframes[frame_id].to("cpu")
        self.active_frames_cpu.append(frame_id)
        self.active_frames_gpu.remove(frame_id) 

    def move_rand_keyframe_to_gpu(self):
        """Move a random keyframe to GPU memory"""
        if len(self.active_frames_cpu) > 0:
            frame_id = np.random.choice(self.active_frames_cpu)
            self.keyframes[frame_id].to("cuda")
            self.active_frames_gpu.insert(0, frame_id)
            self.active_frames_cpu.remove(frame_id) 


    @torch.no_grad()
    def adjust_for_loop_closure(
        self,
        graph_old,
        graph_new
    ):
        print('Running scene_model adjust_for_loop_closure...')
        
        # Pre-compute combined transformation matrices for maximum efficiency
        unique_submap_ids = torch.unique(self.submap_ids)
        transform_cache = {}
        
        for submap_id in unique_submap_ids:
            submap_id_int = submap_id.item()
            pose_old = torch.from_numpy(
                graph_old.get_homography(submap_id_int).matrix()
            ).float().to("cuda")
            pose_new = torch.from_numpy(
                graph_new.get_homography(submap_id_int).matrix()
            ).float().to("cuda")
            
            # Pre-compute combined transformation: pose_new @ inv(pose_old)
            pose_old_inv = torch.linalg.inv(pose_old)
            transform_pose = pose_new @ pose_old_inv  # [4, 4]
            transform_cache[submap_id_int] = transform_pose
        
        # Vectorized Gaussian adjustment
        if self.xyz.shape[0] > 0:
            # Group Gaussians by submap_id for batch processing
            for submap_id in unique_submap_ids:
                submap_id_int = submap_id.item()
                mask = (self.submap_ids == submap_id)
                if not mask.any():
                    continue
                    
                transform_pose = transform_cache[submap_id_int]   # [4, 4]
                R = transform_pose[:3, :3]  # [3, 3]
                t = transform_pose[:3, 3]   # [3]
                
                # Batch process all Gaussians in this submap
                # xyz_batch = self.xyz[mask]  # [N, 3]
                # rotation_batch = self.rotation[mask]  # [N, 4]
                
                self.xyz[mask] = self.xyz[mask] @ R.T + t  # [N, 3]
                # # Transform positions: single matrix multiplication with pre-computed transform
                # xyz_homo = torch.cat(
                #     [
                #         xyz_batch, 
                #         torch.ones(xyz_batch.shape[0], 1, device=xyz_batch.device)
                #     ], 
                #     dim=1
                # )  # [N, 4]
                # # Single batch matrix multiplication with pre-computed transformation
                # transform_pose_batch = transform_pose.unsqueeze(0).expand(xyz_homo.shape[0], -1, -1)  # [N, 4, 4]
                # xyz_homo_batch = xyz_homo.unsqueeze(-1)  # [N, 4, 1]
                # xyz_new_homo = torch.bmm(transform_pose_batch, xyz_homo_batch).squeeze(-1)  # [N, 4]
                # self.xyz[mask] = xyz_new_homo[:, :3]  # [N, 3]
                
                q_rot = mtx2unitquat(R)  # (w,x,y,z)
                q_rot_b = q_rot.expand(self.rotation[mask].shape[0], -1)  # [N,4]
                self.rotation[mask] = F.normalize(
                    quat_mul(q_rot_b, self.rotation[mask]), 
                    dim=-1
                )
                # Transform rotations: single matrix multiplication with pre-computed transform
                # Convert quaternions to rotation matrices (vectorized)
                # rotation_so3_batch = torch.stack([
                #     unitquat2mtx(rot) for rot in rotation_batch
                # ])  # [N, 3, 3]
                
                # # Single batch matrix multiplication with pre-computed rotation transformation
                # rotation_transform_batch = rotation_transform.unsqueeze(0).expand(rotation_so3_batch.shape[0], -1, -1)  # [N, 3, 3]
                # rotation_so3_new = torch.bmm(rotation_transform_batch, rotation_so3_batch)  # [N, 3, 3]
                
                # # Convert back to quaternions
                # new_rotations = torch.stack([
                #     mtx2unitquat(rot) for rot in rotation_so3_new
                # ])  # [N, 4]
                # self.rotation[mask] = new_rotations


        # Remove redundant Gaussians after transformation
        # if self.xyz.shape[0] > 0:
        #     self._prune_redundant_gaussians_after_loop_closure()


        # Vectorized keyframe adjustment
        if len(self.keyframes) > 0:
            # Group keyframes by submap_id
            keyframe_submap_ids = [kf.submap_id for kf in self.keyframes]
            unique_kf_submap_ids = list(set(keyframe_submap_ids))
            
            for submap_id in unique_kf_submap_ids:
                if submap_id not in transform_cache:
                    continue
                    
                transform_pose = transform_cache[submap_id]
                
                # Process all keyframes with this submap_id
                for keyframe in self.keyframes:
                    if keyframe.submap_id != submap_id:
                        continue
                        
                    Rt_old = torch.linalg.inv(keyframe.get_Rt())  # [4, 4]
                    Rt_new = transform_pose @ Rt_old  # [4, 4] - single matrix multiplication
                    keyframe.set_Rt(torch.linalg.inv(Rt_new))
                    
        print('scene_model adjust_for_loop_closure done!')



    def _prune_redundant_gaussians_after_loop_closure(self):
        """
        Remove redundant Gaussians that may have been created during loop closure transformation.
        Uses distance-based clustering and opacity-based filtering.
        """
        with torch.no_grad():
            # Parameters for pruning
            distance_threshold = 0.1  # meters - Gaussians closer than this are considered redundant
            opacity_threshold = 0.01  # Gaussians with opacity below this are removed
            
            # Filter by opacity first
            opacity_mask = self.opacity[:, 0] > opacity_threshold
            
            if not opacity_mask.any():
                return
                
            # Get valid Gaussians
            valid_xyz = self.xyz[opacity_mask]  # [N, 3]
            valid_indices = torch.where(opacity_mask)[0]
            
            if valid_xyz.shape[0] < 2:
                return
                
            # Find clusters of nearby Gaussians using KNN
            k = min(5, valid_xyz.shape[0] - 1)  # Look at 5 nearest neighbors
            _, nn_idx = distIndex2(valid_xyz, k)
            nn_idx = nn_idx.view(-1, k)  # [N, 5]
            
            # Calculate distances to nearest neighbors
            nn_xyz = valid_xyz[nn_idx]  # [N, 5, 3]
            distances = torch.linalg.norm(
                valid_xyz.unsqueeze(1) - nn_xyz, 
                dim=-1
            )  # [N, 5]
            
            # Find Gaussians that are too close to their neighbors
            close_neighbors = distances < distance_threshold  # [N, 5]
            redundant_mask = close_neighbors.sum(dim=1) > 1  # [N, ], More than 1 close neighbor
            
            # Keep the Gaussian with highest opacity in each redundant cluster
            if redundant_mask.any():
                redundant_indices = torch.where(redundant_mask)[0]  # [R, ]
                
                # Create a mask for Gaussians to keep (initialize all as keep)
                keep_mask = torch.ones(self.xyz.shape[0], dtype=torch.bool, device=self.xyz.device)
                
                # Process each redundant Gaussian
                for redundant_idx in redundant_indices:
                    # Find all Gaussians in this cluster
                    cluster_mask = close_neighbors[redundant_idx]  # [5,]
                    cluster_indices = torch.where(cluster_mask)[0]
                    
                    # if len(cluster_indices) > 1:
                    # Get the actual global indices for this cluster
                    global_cluster_indices = valid_indices[cluster_indices]
                        
                        # # Keep the Gaussian with highest opacity
                        # cluster_opacities = self.opacity[global_cluster_indices, 0]
                        # best_in_cluster_idx = global_cluster_indices[cluster_opacities.argmax()]
                        
                    # Mark all others in the cluster for removal
                    for global_idx in global_cluster_indices:
                        # if global_idx != best_in_cluster_idx:
                        keep_mask[global_idx] = False
                
                # Use add_and_prune to remove redundant Gaussians
                self.optimizer.add_and_prune(
                    None,  # No new submap_id
                    self.make_dummy_ext_tensor(),  # No new Gaussians
                    keep_mask
                )
                
                print(f"Pruned {(~keep_mask).sum().item()} redundant Gaussians after loop closure")
                print(f'Gaussian num now: {self.xyz.shape[0]}')


    def enable_inference_mode(self):
        """Enable inference mode and sets the anchor position to the mean of the active keyframes."""
        self.inference_mode = True
        self.update_anchor()


    def update_anchor(self, n_left_frames: int = 0):
        """Update the anchor position and remove the last n_left_frames keyframes from the active anchor."""
        anchor_position = self.approx_cam_centres[
            self.first_active_frame : self.last_active_frame - n_left_frames
        ].mean(dim=0)
        self.active_anchor.position = anchor_position
        
        if n_left_frames > 0:
            self.active_anchor.keyframes = self.active_anchor.keyframes[
                :-n_left_frames
            ]
            self.active_anchor.keyframe_ids = self.active_anchor.keyframe_ids[
                :-n_left_frames
            ]

    def place_anchor_if_needed(self):
        """
        Check if many Gaussians appear small on the screen. 
        If so, place a new anchor and merge the Gaussians.
        """
        small_prop_thresh = 0.4
        # self.n_kept_frames = 20
        k = 3
        
        if (
            self.xyz.shape[0] > 0
            and self.first_active_frame < len(self.keyframes) - 2 * self.n_kept_frames
        ):
            with torch.no_grad():
                dist = torch.linalg.vector_norm(
                    self.xyz - self.approx_cam_centres[-1][None], 
                    dim=-1
                )
                screen_size = self.f * self.scaling.mean(dim=-1) / dist
                small_mask = screen_size < 1
                small_prop = small_mask.float().mean()

            # if more than 40% of the Active Gaussians have a size 
            # S < tau_min from camera i-1's point of view (tau_min = 1 pixel)
            if small_prop > small_prop_thresh:
                with torch.no_grad():
                    small_mask = screen_size < 1.5
                    # Update anchor positions using the camera poses used to optimize it
                    self.update_anchor(self.n_kept_frames)

                    ## Merge fine Gaussians for the current active set
                    # Select a subset and get their nearest neighbours for merging
                    small_gaussians = {
                        name: self.gaussian_params[name]["val"][small_mask]
                        for name in self.gaussian_params
                    }
                    xyz = small_gaussians["xyz"].contiguous()
                    _, nn_idx = distIndex2(xyz, k)
                    nn_idx = nn_idx.view(-1, k)
                    perm = torch.randperm(xyz.shape[0], device=xyz.device)
                    idx = perm[: (xyz.shape[0] // (k + 1))]
                    selected_nn_idx = torch.cat(
                        [
                            idx[..., None], 
                            nn_idx[idx]
                        ], 
                        dim=-1
                    )

                    # Compute merging weights based on contribution to the rendering
                    weights = self.gaussian_params["opacity"]["val"][
                        selected_nn_idx, 0
                    ].sigmoid() * (screen_size[selected_nn_idx] ** 2)
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                    weights.unsqueeze_(-1)

                    # Merge the Gaussians by averaging their parameters
                    merged_gaussians = {
                        "xyz": (
                            self.gaussian_params["xyz"]['val'][selected_nn_idx, :] * weights
                        ).sum(dim=1),
                        "f_dc": (
                            self.gaussian_params["f_dc"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)
                        ).sum(dim=1),
                        "f_rest": (
                            self.gaussian_params["f_rest"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)
                        ).sum(dim=1),
                        "opacity": inverse_sigmoid(
                            self.gaussian_params["opacity"]['val'][selected_nn_idx, :].sigmoid() * weights
                        ).sum(dim=1),
                        "scaling": torch.log(
                            (
                                torch.exp(self.gaussian_params["scaling"]['val'][selected_nn_idx, :]) * weights * (k + 1)
                            ).sum(dim=1)
                        ),
                        "rotation": (
                            self.gaussian_params["rotation"]['val'][selected_nn_idx, :] * weights
                        ).sum(dim=1),
                    }

                    # Offload the previous Gaussians to the CPU
                    self.active_anchor.duplicate_param_dict()
                    self.active_anchor.to("cpu", with_keyframes=True)

                    ## Add the merged Gaussians to the set of Gaussians and reset the optimizer
                    with self.lock:
                        self.optimizer.add_and_prune(merged_gaussians, ~small_mask)

                    # Create a new active anchor with the merged Gaussians
                    self.active_anchor = Anchor(
                        self.gaussian_params,
                        self.approx_cam_centres[-1],
                        self.keyframes[-self.n_kept_frames :],
                    )
                    self.anchors.append(self.active_anchor)
                    self.active_frames_gpu = [
                        kf.index for kf in self.active_anchor.keyframes
                    ]
                    self.active_frames_cpu = []

                    # Visualization
                    self.anchor_weights = np.zeros(len(self.anchors))
                    self.anchor_weights[-1] = 1.0

                gc.collect()
                torch.cuda.empty_cache()


    @torch.no_grad()
    def save_render_results(
        self, 
        path: str
    ):
        from PIL import Image
        import matplotlib.pyplot as plt
        import io
            
        path_gt_image = os.path.join(path, 'gt_image')
        path_render_image = os.path.join(path, 'render_image')
        path_render_depth = os.path.join(path, 'render_depth')
        os.makedirs(path_gt_image, exist_ok=True)
        os.makedirs(path_render_image, exist_ok=True)
        os.makedirs(path_render_depth, exist_ok=True)
        
        # if 'replica' in path:
        #     W_full = 1200
        #     H_full = 680
        # elif 'tum' in path:
        #     W_full = 640
        #     H_full = 480
        # elif 'waymo' in path:
        #     W_full = 1920
        #     H_full = 1280
                
        
        for index, keyframe in enumerate(self.keyframes):
            frame_id = keyframe.frame_id
            
            gt_image = keyframe.image_pyr[0].cuda()  # [3, H, W]
            
            render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
            render_image = render_pkg["render"]  # [3, H, W]
            render_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)  # [H, W]

            psnr_value = psnr(render_image, gt_image)
            
            
            ###### Upsample gt_image, render_image, and render_depth to (H_full, W_full)
            # def upsample_tensor_img(img, size):
            #     # img: [3, H, W] or [1, H, W]
            #     img = img.unsqueeze(0)  # [1, C, H, W]
            #     img_upsampled = F.interpolate(
            #         img, 
            #         size=size, 
            #         mode='bicubic', 
            #         align_corners=False
            #     )
            #     return img_upsampled.squeeze(0)

            # gt_image = upsample_tensor_img(gt_image, (H_full, W_full))
            # render_image = upsample_tensor_img(render_image, (H_full, W_full))
            # render_depth = upsample_tensor_img(render_depth.unsqueeze(0), (H_full, W_full)).squeeze(0)
            
            
            ###### Save gt_image as PNG under path_gt_image, named by frame_id
            gt_image_np = (gt_image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # [3, H, W]
            gt_image_np = np.transpose(gt_image_np, (1, 2, 0))  # [H, W, 3]
            gt_image_filename = os.path.join(path_gt_image, f"{frame_id}.png")
            Image.fromarray(gt_image_np).save(gt_image_filename)
            
            ###### Save render_image as PNG under path_render_image, named by [frame_id]_[psnr_value]
            render_image_np = (render_image.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # [3, H, W]
            render_image_np = np.transpose(render_image_np, (1, 2, 0))  # [H, W, 3]
            psnr_str = f"{psnr_value:.2f}"
            render_image_filename = os.path.join(path_render_image, f"{frame_id}_{psnr_str}.png")
            Image.fromarray(render_image_np).save(render_image_filename)

            ###### Save render_depth as PNG under path_render_depth, named by frame_id, using jet colormap
            render_depth_np = render_depth.detach().cpu().numpy()  # [H, W]
            # Normalize depth for colormap: ignore inf/NaN, use min/max of valid values
            valid_mask = np.isfinite(render_depth_np)
            if np.any(valid_mask):
                min_depth = np.min(render_depth_np[valid_mask])
                max_depth = np.max(render_depth_np[valid_mask])
                # Avoid division by zero
                if max_depth > min_depth:
                    norm_depth = (render_depth_np - min_depth) / (max_depth - min_depth)
                else:
                    norm_depth = np.zeros_like(render_depth_np)
                norm_depth = np.clip(norm_depth, 0, 1)
            else:
                norm_depth = np.zeros_like(render_depth_np)
            # Apply jet colormap
            cmap = plt.get_cmap('jet')
            depth_colored = cmap(norm_depth)[:, :, :3]  # [H, W, 3], drop alpha
            depth_colored = (depth_colored * 255).astype(np.uint8)
            render_depth_filename = os.path.join(path_render_depth, f"{frame_id}.png")
            Image.fromarray(depth_colored).save(render_depth_filename)

        print(f"Saved in: {path_gt_image}!")
        print(f"Saved in: {path_render_image}!")
        print(f"Saved in: {path_render_depth}!")



    @torch.no_grad()
    def save_traj(
        self, 
        path: str
    ):
        from scipy.spatial.transform import Rotation as R
        
        traj_file = os.path.join(path, 'traj.txt')
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists, not the file

        with open(traj_file, 'w') as f:
            for keyframe in self.keyframes:
                frame_id = keyframe.frame_id
                
                Rt = keyframe.get_Rt().detach().cpu().numpy()  # [4, 4]
                pose = np.linalg.inv(Rt)  # [4, 4]
                
                # TUM format: timestamp tx ty tz qx qy qz qw
                # Convert rotation to quaternion (wxyz)
                Rmat = pose[:3, :3]
                tvec = pose[:3, 3]
                quat = R.from_matrix(Rmat).as_quat()  # [x, y, z, w]
                
                # TUM expects [qw, qx, qy, qz]
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                # Write line: frame_id tx ty tz qx qy qz qw
                f.write(f"{frame_id} {tvec[0]:.6f} {tvec[1]:.6f} {tvec[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
        print(f"Saved as: {traj_file}!")



    def save(
        self, 
        path: str, 
        reconstruction_time: float = 0, 
        n_frames: int = 0
    ):
        ###### Get metrics
        metrics = {
            "num anchors": len(self.anchors),
            "num keyframes": len(self.keyframes),
            "num Gaussians": self.xyz.shape[0],
        }
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                metrics["FPS"] = n_frames / reconstruction_time
                
        metrics.update(
            self.evaluate(
                eval_poses=False, 
                with_LPIPS=True, 
                all_kfs=True
            )
        )

        if path == "":
            print("No path provided, skipping save")
            return metrics
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        ###### Save anchors
        # pcd_path = os.path.join(path, "point_clouds")
        # os.makedirs(pcd_path, exist_ok=True)
        # for index, anchor in enumerate(self.anchors):
        #     anchor.save_ply(os.path.join(pcd_path, f"anchor_{index}.ply"))

        ###### Save metadata
        metadata = {
            "config": {
                "width": self.width,
                "height": self.height,
                "sh_degree": self.max_sh_degree,
                "f": self.f,
            },
            "anchors": [
                {
                    "position": anchor.position.cpu().numpy().tolist(),
                }
                for anchor in self.anchors
            ],
            "keyframes": [
                keyframe.to_json() for keyframe in self.keyframes
            ],
        }
        metadata = {
            **metrics, 
            **metadata
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # Save renders of test views
        # self.save_test_frames(os.path.join(path, "test_images"))

        # Saving cameras with COLMAP format
        # images = {}
        # cameras = {}
        # colmap_save_path = os.path.join(path, "colmap")
        # os.makedirs(colmap_save_path, exist_ok=True)
        # for index, keyframe in enumerate(self.keyframes):
        #     camera, image = keyframe.to_colmap(index)
        #     cameras[index] = camera
        #     images[index] = image
        # write_model(cameras, images, {}, colmap_save_path, ext=".bin")

        return metrics


    # for viewer
    def get_closest_keyframe(
        self, 
        position: torch.Tensor, 
        count: int = 1
    ) -> list[Keyframe]:
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], 
            dim=-1
        )
        closest_ids = dists.argsort()[:count]
        return [
            self.keyframes[closest_id] for closest_id in closest_ids
        ]


    def finetune_epoch(self):
        """
        Go through all anchors and optimize them one by one.
        This is used for finetuning after the initial training.
        """
        self.anchor_weights = np.zeros(len(self.anchors))
        
        for anchor_id, anchor in enumerate(self.anchors):
            self.active_anchor = anchor
            # Load the anchor and make its parameters optimizable
            anchor.to("cuda", with_keyframes=True)
            self.gaussian_params = anchor.gaussian_params
            self.anchor_weights[anchor_id] = 1
            self.reset_optimizer()

            # # Ensure other anchors are on cpu to save memory
            # if anchor_id >= 1:
            #     self.anchors[anchor_id-1].to("cpu", with_keyframes=True)

            # Optimize the anchor by going through its keyframes
            for _ in range(len(anchor.keyframes)):
                self.optimization_step(finetuning=True)

            # Update the anchor and store it on cpu
            anchor.gaussian_params = self.gaussian_params
            self.anchor_weights[anchor_id] = 0

