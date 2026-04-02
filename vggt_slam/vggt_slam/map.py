import os
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from vggt_slam.submap import Submap


class GraphMap:
    def __init__(self):
        self.submaps: dict[int, Submap] = dict()
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap: Submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
    
    def get_largest_key(self):
        if len(self.submaps) == 0:
            return -1
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self):
        return self.get_submap(self.get_largest_key())
    
    
    # def retrieve_best_score_frame(
    def retrieve_best_sim_frame(
        self, 
        query_vector, 
        current_submap_id, 
        ignore_last_submap=True
    ):
        # overall_best_score = 1000
        overall_best_sim = 0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        
        # search for best image to target image
        for submap_key in self.submaps.keys():
            if submap_key == current_submap_id:
                continue

            if ignore_last_submap and (submap_key == current_submap_id - 1):
                continue

            else:
                submap = self.submaps[submap_key]
                submap_embeddings = submap.get_all_retrieval_vectors()
                
                # scores = []
                cossims = []
                for embedding in submap_embeddings:
                    ###### lower score -> higher similarity 
                    # score = torch.linalg.norm(embedding - query_vector)
                    # scores.append(score.item())
                    ###### cosine similarity
                    cossim = F.cosine_similarity(embedding, query_vector, dim=0)
                    cossims.append(cossim.item())
                
                # best_score_id = np.argmin(scores)
                # best_score = scores[best_score_id]
                best_sim_id = np.argmax(cossims)
                best_sim = cossims[best_sim_id]

                # if best_score < overall_best_score:
                if best_sim > overall_best_sim:
                    # overall_best_score = best_score
                    overall_best_sim = best_sim
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_sim_id  #best_score_id

        return (
            overall_best_sim,  #overall_best_score, 
            overall_best_submap_id, 
            overall_best_frame_index
        )


    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(
                self.submaps[detected_loop.detected_submap_id].get_frame_at_index(
                    detected_loop.detected_submap_frame
                )
            )
        return frames
    
    
    def update_submap_homographies(self, graph):
        for submap_id in self.submaps.keys():
            submap = self.submaps[submap_id]
            
            submap.set_reference_homography(
                graph.get_homography(submap_id).matrix()
            )
    
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]


    def write_poses_to_file(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
        with open(file_name, "w") as f:
            for submap in self.ordered_submaps_by_key():
                poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
                frame_ids = submap.get_frame_ids()
                assert len(poses) == len(frame_ids), "Number of provided poses and number of frame ids do not match"
                
                for frame_id, pose in zip(frame_ids, poses):
                    x, y, z = pose[0:3, 3]
                    rotation_matrix = pose[0:3, 0:3]
                    quaternion = R.from_matrix(rotation_matrix).as_quat()  # x, y, z, w
                    output = np.array([
                        float(frame_id), 
                        x, y, z, 
                        *quaternion
                    ])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def write_poses_to_file2(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        written_ids = []
        
        with open(file_name, "w") as f:
            for submap in self.ordered_submaps_by_key():
                poses = submap.get_all_poses_world2(ignore_loop_closure_frames=True)
                frame_ids = submap.get_frame_ids()
                assert len(poses) == len(frame_ids), "Number of provided poses and number of frame ids do not match"
                
                for frame_id, pose in zip(frame_ids, poses):
                    if frame_id in written_ids:
                        continue
                    
                    x, y, z = pose[0:3, 3]
                    rotation_matrix = pose[0:3, 0:3]
                    quaternion = R.from_matrix(rotation_matrix).as_quat()  # x, y, z, w
                    output = np.array([
                        float(frame_id), 
                        x, y, z, 
                        *quaternion
                    ])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")
                    
                    written_ids.append(frame_id)


    def save_framewise_pointclouds(self, file_name):
        os.makedirs(file_name, exist_ok=True)
        
        for submap in self.ordered_submaps_by_key():
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(ignore_loop_closure_frames=True)
            for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
                # save pcd as numpy array
                np.savez(f"{file_name}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)

    def write_points_to_file(self, file_name):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame()
            pcd = pcd.reshape(-1, 3)
            pcd_all.append(pcd)
            colors_all.append(submap.get_points_colors())
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)
        