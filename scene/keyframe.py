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

from __future__ import annotations
from argparse import Namespace
import torch
import torch.nn.functional as F
import numpy as np

# from poses.feature_detector import DescribedKeypoints
# from poses.triangulator import Triangulator
# from scene.dense_extractor import DenseExtractor
# from scene.mono_depth import MonoDepthEstimator, align_depth

from scene.optimizers import BaseAdam
from utils import sample, sixD2mtx, make_torch_sampler, depth2points
from dataloaders.read_write_model import Camera, BaseImage, rotmat2qvec


class Keyframe:
    """
    A keyframe in the scene, 
    containing the image, camera parameters, and other information used for optimization.
    """
    def __init__(
        self,
        submap_id: int,
        frame_id: float,
        index: int,
        image: torch.Tensor,  # [3, H, W]
        # pointmap: torch.Tensor,  # [H, W, 3]
        depth: np.ndarray,  # [H, W, 1]
        depth_conf: np.ndarray,  # [H, W]
        Rt: np.ndarray,  # [4, 4], from camera to world
        f: torch.Tensor,
        config : dict,
        inference_mode: bool = False
    ):
        self.image_pyr = [image]
        
        if not inference_mode:  # Only extract depth and feature maps in training mode
            # self.pointmap = pointmap  # [H, W, 3]
            
            # self.depth = depth.squeeze(-1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            self.depth_conf = torch.from_numpy(depth_conf).cuda().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            self.width = image.shape[2]
            self.height = image.shape[1]
            self.centre = torch.tensor(
                [(self.width - 1) / 2, (self.height - 1) / 2], 
                device="cuda"
            )
            self.f = f
            
            self.depth_loss_weight = config['depth_loss_weight_init']
            self.depth_loss_weight_decay = config['depth_loss_weight_decay']
            
            # Build the multiscale pyramids
            self.pyr_lvl = config['pyr_levels'] - 1
            self.depth_pyr = [
                torch.from_numpy(depth).cuda().permute(2, 0, 1)  # [1, H, W]
            ]
            for _ in range(self.pyr_lvl):
                self.image_pyr.append(
                    F.avg_pool2d(self.image_pyr[-1], 2)
                )
                self.depth_pyr.append(
                    F.avg_pool2d(self.depth_pyr[-1], 2)
                )
        
        
        self.submap_id = submap_id
        self.frame_id = frame_id
        self.index = index
        self.is_test = False

        # Optimizable parameters
        if isinstance(Rt, np.ndarray):
            Rt = torch.from_numpy(Rt).cuda()
        self.rW2C = torch.nn.Parameter(Rt[:3, :2].clone())
        self.tW2C = torch.nn.Parameter(Rt[:3, 3].clone())
        
        exposure = torch.eye(3, 4, device="cuda")
        self.exposure = torch.nn.Parameter(exposure)
        
        self.depth_scale = torch.nn.Parameter(torch.ones(1, device="cuda"))
        self.depth_offset = torch.nn.Parameter(torch.zeros(1, device="cuda"))

        # Optimizer
        if not inference_mode:  # Only create optimizer in training mode
            params = {
                "rW2C": {
                    "val": self.rW2C, 
                    "lr": config['lr_poses']
                },
                "tW2C": {
                    "val": self.tW2C, 
                    "lr": config['lr_poses']
                },
                "depth_scale": {
                    "val": self.depth_scale,
                    "lr": config['lr_depth_scale_offset']
                },
                "depth_offset": {
                    "val": self.depth_offset,
                    "lr": config['lr_depth_scale_offset']
                },
            }
            
            if not self.is_test:
                params["exposure"] = {
                    "val": self.exposure, 
                    "lr": config['lr_exposure']
                }
                
            self.optimizer = BaseAdam(params, betas=(0.8, 0.99))
            self.num_steps = 0

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]


    def to(self, device: str, only_train=False):
        if self.device.type == device:
            return
        
        for i in range(len(self.image_pyr)):
            self.image_pyr[i] = self.image_pyr[i].to(device)
            self.depth_pyr[i] = self.depth_pyr[i].to(device)

        # if not only_train:
        #     self.depth = self.depth.to(device)


    @property
    def device(self):
        return self.image_pyr[0].device

    def get_R(self):
        return sixD2mtx(self.rW2C)

    def get_t(self):
        return self.tW2C

    def get_Rt(self):
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def set_Rt(self, Rt: torch.Tensor):
        self.rW2C.data.copy_(Rt[:3, :2])
        self.tW2C.data.copy_(Rt[:3, 3])
        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def get_centre(self, approx=False):
        if approx:
            return self.approx_centre
        else:
            return -self.get_R().T @ self.get_t()


    def get_depth(self, lvl=0):
        if self.depth_pyr[lvl].device.type != "cuda":
            self.depth_pyr[lvl] = self.depth_pyr[lvl].cuda()
        return self.depth_pyr[lvl] * self.depth_scale + self.depth_offset


    @torch.no_grad()
    def sample_conf(self, uv):
        return sample(
            self.depth_conf, 
            uv.view(1, 1, -1, 2), 
            self.width, 
            self.height
        )[0, 0, 0]


    def zero_grad(self):
        self.optimizer.zero_grad()


    @torch.no_grad()
    def step(self):
        # Optimizer step
        self.optimizer.step()
        
        self.depth_loss_weight *= self.depth_loss_weight_decay
        self.num_steps += 1
        
        # decrement pyr_lvl
        if self.num_steps % 5 == 0:
            if self.pyr_lvl > 0:
                self.image_pyr.pop()
                self.depth_pyr.pop()
                self.pyr_lvl -= 1

   
    def to_json(self):
        info = {
            # "is_test": self.info["is_test"],
            "is_test": self.is_test,
        }
        # if "name" in self.info:
        #     info["name"] = self.info["name"]
        # if "Rt" in self.info:
        #     info["gt_Rt"] = self.info["Rt"].cpu().numpy().tolist()

        info["frame_id"] = self.frame_id
        info["index"] = self.index

        return {
            "info": info,
            "Rt": self.get_Rt().detach().cpu().numpy().tolist(),
            "f": self.f.item(),
        }

    
    """ 
    @classmethod
    def from_json(cls, config, index, height, width):
        if "gt_Rt" in config["info"]:
            config["info"]["Rt"] = torch.tensor(config["info"]["gt_Rt"]).cuda()
        keyframe = cls(
            image=None,
            info=config["info"],
            desc_kpts=None,
            Rt=torch.tensor(config["Rt"]).cuda(),
            index=index,
            f=None, 
            feat_extractor=None,
            depth_estimator=None,
            triangulator=None,
            args=None,
            inference_mode=True,
        )
        keyframe.height = height
        keyframe.width = width
        keyframe.centre = torch.tensor(
            [(width - 1) / 2, (height - 1) / 2], device="cuda"
        )
        return keyframe 
    """

    '''
    def to_colmap(self, id):
        """
        Convert the keyframe to a colmap camera and image.
        """
        # first param of params is focal length in pixels
        camera = Camera(
            id=id,
            model="SIMPLE_PINHOLE",
            width=self.width,
            height=self.height,
            params=[self.f.item(), self.centre[0].item(), self.centre[1].item()],
        )

        image = BaseImage(
            id=id,
            name=self.info.get("name", str(id)),
            camera_id=id,
            qvec=-rotmat2qvec(self.get_R().cpu().detach().numpy()),
            tvec=self.get_t().flatten().cpu().detach().numpy(),
            xys=[],
            point3D_ids=[],
        )

        return camera, image
    '''
    
    