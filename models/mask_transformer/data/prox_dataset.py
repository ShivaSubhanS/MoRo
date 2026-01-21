import torch
from torch.utils import data
from tqdm import tqdm
import glob
import smplx
import pickle as pkl
import json
import os
import pandas as pd
import numpy as np
import cv2

from lightning import LightningDataModule

from ..utils import (
    get_bbox_valid,
    crop,
    GlobalTrajHelper,
    get_bm_params,
    get_contact_label, 
)
from ..utils.body_model import BodyModel

# PROX scene floor heights
prox_floor_height = {'N0Sofa': -0.9843093165454873,
                     'MPH1Library': -0.34579620031341207,
                     'N3Library': -0.6736229583361132,
                     'N3Office': -0.7772727989022952,
                     'BasementSittingBooth': -0.767080139846674,
                     'MPH8': -0.41432886722717904,
                     'MPH11': -0.7169139211234009,
                     'MPH16': -0.8408992040141058,
                     'MPH112': -0.6419028605753081,
                     'N0SittingBooth': -0.6677103008966809,
                     'N3OpenArea': -1.0754909672969915,
                     'Werkraum': -0.6777057869851316}
                     
test_recording_name_list = [
    "MPH1Library_00034_01",
    "N0Sofa_00034_01",
    "N0Sofa_00034_02",
    "N0Sofa_00141_01",
    "N0Sofa_00145_01",
    "N3Library_00157_01",
    "N3Library_00157_02",
    "N3Library_03301_01",
    "N3Library_03301_02",
    "N3Library_03375_01",
    "N3Library_03375_02",
    "N3Library_03403_01",
    "N3Library_03403_02",
    "N3Office_00034_01",
    "N3Office_00139_01",
    "N3Office_00150_01",
    "N3Office_00153_01",
    "N3Office_00159_01",
    "N3Office_03301_01",
]

class DataloaderProx(data.Dataset):
    def __init__(self, cfg, recording):
        """
        Args:
            cfg: configuration object
            recording: single recording name (not list)
        """
        self.cfg = cfg
        self.joints_num = 22
        self.dataset_root = cfg.dataset_root
        self.recording = recording
        self.seq_name = recording

        female_subjects_ids = [162, 3452, 159, 3403]
        subject_id = int(recording.split('_')[1])
        if subject_id in female_subjects_ids:
            gender = 'female'
        else:
            gender = 'male' 

        self.clip_len = cfg.clip_len
        self.clip_overlap_len = cfg.overlap_len

        self.device = "cpu"  # cannot use cuda in multi-process dataloader
        self.model_type = "smplx"
        self.bm_model = BodyModel(
            bm_path=cfg.bm_path,
            model_type=self.model_type,
            # gender="neutral",
            gender=gender,
        )

        self.normalize = cfg.normalize
        self.canonical = cfg.canonical
        self.global_traj_helper = GlobalTrajHelper()

        # Get scene name and load camera info
        self.scene_name = recording.split("_")[0]
        
        # Load camera information
        with open(os.path.join(self.dataset_root, 'cam2world', self.scene_name + '.json'), 'r') as f:
            cam2world = np.array(json.load(f))
        self.cam2world = torch.from_numpy(cam2world).float().to(self.device)
        
        with open(os.path.join(self.dataset_root, 'calibration', 'Color.json'), 'r') as f:
            self.color_cam = json.load(f)
        
        # Initialize data structures
        self.frame_name_list = []
        self.param_gt_list = []
        self.n_samples = 0

        # Load data for the recording
        self.read_data()

        # Coordinate transformation, identity, z-axis up
        self.world2cano = torch.tensor(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1]]
        ).unsqueeze(0).float()

        print(f"[INFO] PROX recording {recording}: get {self.n_samples} sub clips in total.")

    def read_data(self):
        """Read PROX data for the recording"""
        # Load SMPL fitting results
        fitting_gt_root = os.path.join(
            self.dataset_root,
            f"PROXD_filled",
            self.recording,
            "results"
        )
        
        frame_list = os.listdir(fitting_gt_root)
        frame_list.sort()
        
        # Load parameters for all frames
        frame_name_list = []
        param_gt_list = []
        for frame_name in frame_list:
            frame_name_list.append(frame_name)
            with open(os.path.join(fitting_gt_root, frame_name, "000.pkl"), "rb") as f:
                param_gt = pkl.load(f)
            param_gt_list.append(param_gt)

        # Divide sequence into clips with overlapping window
        seq_idx = 0
        while True:
            start = seq_idx * (self.clip_len - self.clip_overlap_len)
            end = start + self.clip_len
            self.frame_name_list.append(frame_name_list[start:end])
            self.param_gt_list.append(param_gt_list[start:end])
            seq_idx += 1
            if end >= len(frame_name_list):
                break
            
        self.n_samples = seq_idx

    def create_mesh(self, bm_params_dict, bm_model):
        translation = bm_params_dict["transl"]
        rotation = bm_params_dict["global_orient"]

        bm_params_dict["transl"] = None
        if self.canonical:
            bm_params_dict["global_orient"] = None

        body_mesh = bm_model(**bm_params_dict)
        vertices = body_mesh.vertices.clone().detach()
        joints = body_mesh.joints.clone().detach()
        full_joints = body_mesh.full_joints.clone().detach()
        root_pos = body_mesh.joints[:, :1].clone().detach()

        vertices = vertices - root_pos
        joints = joints - root_pos
        full_joints = full_joints - root_pos

        if self.normalize:
            vertices_offset = torch.mean(vertices, dim=1, keepdim=True)
            vertices = vertices - vertices_offset
        else:
            vertices_offset = torch.zeros_like(vertices)

        translation = translation.unsqueeze(1)
        translation = translation + root_pos

        mesh_dict = {
            "mesh": vertices,
            "local_joints": joints,
            "offset": vertices_offset,
            "root_pos": root_pos,
            "rotation": rotation,
            "translation": translation,
        }

        return mesh_dict, full_joints

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Get clip data
        frame_names = self.frame_name_list[index]
        param_gt_clip = self.param_gt_list[index]
        
        num_frames = len(frame_names)
        
        K = np.asarray(self.color_cam["camera_mtx"])
        dist_coeffs = np.asarray(self.color_cam["k"])
        
        # Get SMPL parameters for clip
        param_gt = {}
        for key in ["transl", "global_orient", "betas", "body_pose"]:
            param_gt[key] = np.concatenate([param[key] for param in param_gt_clip])
            if key == "body_pose":
                param_gt[key] = param_gt[key][:, :63]  # discard hand pose
                
        bm_params_cam = get_bm_params(param_gt)

        # Create mesh
        mesh_dict, full_joints_gt_local = self.create_mesh(bm_params_cam.copy(), self.bm_model)
        joints_gt_cam = mesh_dict["local_joints"] @ mesh_dict["rotation"].permute(0, 2, 1) + mesh_dict["translation"]

        # Project to 2D and distort
        F, J = joints_gt_cam.shape[:2] 
        joints_gt_2d = cv2.projectPoints(
            joints_gt_cam.detach().cpu().numpy().reshape(-1, 3),
            rvec=np.zeros((1, 3), dtype=np.float32),
            tvec=np.zeros((3), dtype=np.float32),
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
        )[0].reshape(F, J, 2)
        K_tensor = torch.from_numpy(K).float().to(self.device)

        # Compute bounding boxes from undistorted SMPL joints
        img_path_list = []
        center_list = []
        scale_list = []
        crop_img_list = []
        h, w = 1080, 1920  # PROX image dimensions

        # load the keypoint derived bounding boxes
        bbox_center, bbox_scale = None, None
        if not self.cfg.get("gt_bbox", True):
            # load bbox from preprocessed npz file
            bbox_path = os.path.join(self.dataset_root, 
                "keypoints_openpose", self.recording, f"bbox.npz")
            bbox_data = np.load(bbox_path)
            frame_names_bbox = bbox_data["frame_names"]
            centers_bbox = bbox_data["centers"]
            scales_bbox = bbox_data["scales"]
            
            # create mapping from frame name to index
            bbox_center = dict(zip(frame_names_bbox, centers_bbox))
            bbox_scale = dict(zip(frame_names_bbox, scales_bbox))

        for i in range(num_frames):
            frame_name = frame_names[i]
            joints_2d = joints_gt_2d[i]

            img_path = os.path.join(
                self.dataset_root,
                "recordings",
                self.recording,
                "Color",
                frame_name + ".jpg",
            )

            try:
                if not os.path.exists(img_path):
                    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                else:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Undistort the image before cropping
            img = cv2.undistort(
                img,
                cameraMatrix=K,
                distCoeffs=dist_coeffs,
            )
            img = cv2.flip(img, 1)  # flip horizontally

            scale_factor_bbox = 1.2
            res = self.cfg.get("crop_res", (224, 224))
            
            if self.cfg.get("gt_bbox", True):
                # Compute bounding box from SMPL joints
                center, scale, _, _ = get_bbox_valid(
                    joints_2d, rescale=scale_factor_bbox
                )
            else:
                center = bbox_center[frame_name]
                scale = bbox_scale[frame_name]
            img, crop_img = crop(img, center, scale, res=res)

            img_path_list.append(img_path)
            center_list.append(center)
            scale_list.append(scale)
            crop_img_list.append(crop_img)

        center_list = np.stack(center_list).astype(np.float32)
        scale_list = np.stack(scale_list).astype(np.float32)
        center_list = torch.from_numpy(center_list).float()
        scale_list = torch.from_numpy(scale_list).float()
        
        # Process cropped images
        crop_img_list = np.stack(crop_img_list)
        crop_img_list = crop_img_list.transpose((0, 3, 1, 2))
        crop_img_list = crop_img_list.astype(np.float32)
        crop_img_list /= 255.0

        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        crop_img_list = (crop_img_list - mean) / std
        crop_img_list = torch.from_numpy(crop_img_list).float()

        # Bounding box information
        focal = 0.5 * (K[0, 0] + K[1, 1])
        cam_center = torch.zeros_like(center_list)
        cam_center[:, 0] = K[0, 2]
        cam_center[:, 1] = K[1, 2]
        bbox = (
            torch.cat(
                [center_list - cam_center, scale_list.unsqueeze(-1) * 200.0], dim=1
            )
            / focal
        )

        # Get trajectory representation
        rotation = mesh_dict["rotation"]
        translation = mesh_dict["translation"]
        _, cano_traj_clean = self.global_traj_helper.get_cano_traj_repr(
            rotation, translation
        )

        # Prepare batch dictionary
        batch = {
            # Basic info
            "body_idx": "0",  # PROX doesn't have body_idx concept
            "seq_name": self.seq_name,
            "center": center_list,
            "scale": scale_list,
            "crop_imgs": crop_img_list,
            "bbox": bbox,
            "has_transl": True,
            "true_params": False,
            
            # Camera parameters
            "K": K_tensor.float().unsqueeze(0).repeat(num_frames, 1, 1),
            "dist_coeffs": dist_coeffs,
            "cam2world": self.cam2world.float().unsqueeze(0).repeat(num_frames, 1, 1),
            "world2cano": self.world2cano,
            
            # Images
            "img_paths": img_path_list,
            
            # Trajectory
            "cano_traj_clean": cano_traj_clean.squeeze(0).float(),
        }
        
        # Add mesh data
        batch.update(mesh_dict)
        
        # Foot contact labels
        if rotation.shape[0] > 1:
            joints_gt_world = joints_gt_cam @ self.cam2world[:3, :3].T + self.cam2world[:3, 3]
            contact_label = get_contact_label(joints_gt_world, vel_thres=self.cfg.contact_vel_thres, contact_hand=self.cfg.contact_hand)
            batch["contact_label"] = contact_label.float()

        return batch


class DataModuleProx(LightningDataModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg
        self.debug = debug

        if self.debug:
            self.test_recordings = [
                "N0Sofa_00145_01",
                # "N3Library_03301_01",
                # "N3Library_03301_02",
                # "MPH1Library_00034_01",
                # "MPH112_00034_01",
                # "N0SittingBooth_03403_01",
                ]
        else:
            recordings = self.cfg.get("recording", None) 
            if recordings is None:
                self.test_recordings = test_recording_name_list
            elif isinstance(recordings, str):
                self.test_recordings = [recordings]
            else:
                self.test_recordings = recordings

    def setup(self, stage=None):
        if stage in [None, "test"]:
            if not hasattr(self, "test_datasets"):
                test_datasets = [
                    DataloaderProx(
                        self.cfg,
                        recording=recording,
                    )
                    for recording in self.test_recordings
                ]

                self.test_dataset = data.ConcatDataset(test_datasets)

    def test_dataloader(self):
        return data.DataLoader(
                self.test_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers_val,
                drop_last=False,
                pin_memory=True,
            )
            