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
    GlobalTrajHelper,
    crop,
    get_bm_params,
    get_keypoint_mapping,
    crop_bbox_seq,
    get_contact_label
)
from ..utils.body_model import BodyModel


class DataloaderBEDLAM(data.Dataset):
    def __init__(
        self,
        cfg,
        scenes,
        split="train",
    ):
        self.cfg = cfg

        self.joints_num = 22

        self.dataset_root = "datasets/mask_transformer/BEDLAM"
        self.scenes = scenes
        self.split = split

        self.seq_name = scenes[0]

        self.clip_len = cfg.clip_len
        self.clip_overlap_len = cfg.overlap_len

        self.skip = cfg.skip

        self.device = "cpu"  # cannot use cuda in multi-process dataloader
        self.bm_neutral = BodyModel(
            bm_path=cfg.bm_path,
            model_type="smplx",
            gender="neutral",
        )

        self.normalize = cfg.normalize
        self.canonical = cfg.canonical

        self.global_traj_helper = GlobalTrajHelper()

        self.scene_dict = self.get_scene_dict()
        if self.split == "test":
            self.init_data()
        else:
            self.load_data()

        self.world2cano = torch.tensor(
            [[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1]]
        ).unsqueeze(0).float()

        print(
            "[INFO] {} set: get {} sub clips in total.".format(
                self.split, self.n_samples
            )
        )

    def get_scene_dict(self):
        df = pd.read_csv(os.path.join(self.dataset_root, "bedlam_scene_names.csv"))
        scene_dict = dict(zip(df["Scene name"], df["Folder name"]))
        return scene_dict

    def init_data(self):
        self.clip_data_list = []
        for scene in tqdm(self.scenes):
            folder = self.scene_dict[scene]
            with open(
                os.path.join(self.dataset_root, "processed_labels", f"{folder}.pkl"),
                "rb",
            ) as f:
                data = pkl.load(f)

            num_frames = [len(x) for x in data["imgname"]]

            for seq_idx, seq_frames in enumerate(tqdm(num_frames)):
                clip_idx = 0
                while 1:
                    start = clip_idx * (self.clip_len - self.clip_overlap_len)
                    end = start + self.clip_len
                    if end > seq_frames:
                        break

                    clip_data = {
                        k: np.stack(v[seq_idx][start:end]) for k, v in data.items()
                    }
                    clip_data["scene"] = scene
                    clip_data["folder"] = folder
                    if self.filter_data(clip_data):
                        self.clip_data_list.append(clip_data)
                    clip_idx += self.skip

        self.n_samples = len(self.clip_data_list)

    def filter_data(self, clip_data):
        # filter the clip data according to visibility
        world2cam = clip_data["cam_ext"]  # could be moving camera
        K = clip_data["cam_int"]  # could be zoomed in

        param_gt_cam = {}
        param_gt_cam["transl"] = clip_data["trans_cam"] + world2cam[:, :3, 3]
        param_gt_cam["global_orient"] = clip_data["pose_world"][:, :3]
        param_gt_cam["betas"] = clip_data["shape"]
        param_gt_cam["body_pose"] = clip_data["pose_world"][:, 3:66]

        bm_params_gt_cam = get_bm_params(param_gt_cam)

        smpl_output_gt_cam = self.bm_neutral(**bm_params_gt_cam)
        joints_gt_cam = smpl_output_gt_cam.joints.detach().cpu().numpy()
        joints_gt_2d = joints_gt_cam @ K.transpose(0, 2, 1)
        joints_gt_2d = joints_gt_2d[:, :, :2] / joints_gt_2d[:, :, 2:]

        clip_data["joints_cam"] = joints_gt_cam
        clip_data["joints_2d"] = joints_gt_2d

        if "closeup" in clip_data["scene"]:
            # filter out the frames with invalid bbox
            IMG_W = 720
            IMG_H = 1280
        else:
            IMG_W = 1280
            IMG_H = 720

        is_valid = np.logical_and.reduce(
            [
                joints_gt_2d[..., 0] < IMG_W,
                joints_gt_2d[..., 1] < IMG_H,
                joints_gt_2d[..., 0] >= 0,
                joints_gt_2d[..., 1] >= 0,
            ]
        )
        is_valid = np.sum(is_valid, axis=1)
        is_valid = is_valid >= 16
        valid_ratio = np.sum(is_valid) / len(is_valid)
        if valid_ratio < 0.8:
            return False
        return True

    def save_data(self):
        save_path = os.path.join(
            os.path.dirname(self.dataset_root),
            "bedlam_preprocess",
            f"{self.split}_{self.clip_len}_{self.clip_overlap_len}_{self.skip}.pkl",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pkl.dump(self.clip_data_list, f)

    def load_data(self):
        save_path = os.path.join(
            os.path.dirname(self.dataset_root),
            "bedlam_preprocess",
            f"{self.split}_{self.clip_len}_{self.clip_overlap_len}_{self.skip}.pkl",
        )
        if not os.path.exists(save_path):
            self.init_data()
            self.save_data()
            return
        with open(save_path, "rb") as f:
            self.clip_data_list = pkl.load(f)

        self.n_samples = len(self.clip_data_list)

    def create_mesh(self, bm_params_dict):
        translation = bm_params_dict["transl"]
        rotation = bm_params_dict["global_orient"]

        bm_params_dict["transl"] = None
        if self.canonical:
            bm_params_dict["global_orient"] = None

        body_mesh = self.bm_neutral(**bm_params_dict)
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

        # compute the simplified translation and rotation
        # so that the transformation from local to global is Rx+t
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

    def get_scene_clip(self, index):
        clip_idx = 0
        for scene, scene_clips in self.scene_num_clips.items():
            if clip_idx + scene_clips > index:
                return scene, index - clip_idx
            clip_idx += scene_clips
        raise ValueError("Index out of range")
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        clip_data = self.clip_data_list[index]

        scene = clip_data["scene"]
        folder = clip_data["folder"]

        gender = "neutral"  # we use SMPL-X neutral gt from BEDLAM
        sub = clip_data["sub"][0]
        world2cam = clip_data["cam_ext"]  # could be moving camera
        cam2world = np.linalg.inv(world2cam)
        K = clip_data["cam_int"]  # could be zoomed in
        K = torch.from_numpy(K).float()


        # real camera coord trans should add world2cam[:3,3]
        param_gt_cam = {}
        param_gt_cam["transl"] = clip_data["trans_cam"] + world2cam[:, :3, 3]
        param_gt_cam["global_orient"] = clip_data["pose_cam"][:, :3]
        param_gt_cam["betas"] = clip_data["shape"]
        param_gt_cam["body_pose"] = clip_data["pose_cam"][:, 3:66]

        bm_params_cam = get_bm_params(param_gt_cam)

        # get mesh
        mesh_dict, full_joints_gt_local = self.create_mesh(bm_params_cam.copy())
        full_joints_gt_cam = full_joints_gt_local @ mesh_dict["rotation"].permute(0, 2, 1) + mesh_dict["translation"]

        full_joints_gt_2d = full_joints_gt_cam @ K.permute(0, 2, 1)
        full_joints_gt_2d = full_joints_gt_2d[:, :, :2] / (full_joints_gt_2d[:, :, 2:] + 1e-6)
        full_joints_gt_2d = full_joints_gt_2d.detach().cpu().numpy()
        full_joints_gt_2d = np.concatenate([full_joints_gt_2d, np.ones((full_joints_gt_2d.shape[0], full_joints_gt_2d.shape[1], 1))], axis=-1)

        # get images
        imgnames = clip_data["imgname"]
        centers = clip_data["center"]
        scales = clip_data["scale"]
        num_frames = len(imgnames)

        img_path_list = []
        crop_img_list = []

        # do cropping if needed
        if self.split == "train" and self.cfg.extreme_crop_aug and np.random.rand() < self.cfg.extreme_crop_aug_prob:
            # use random cropping for training
            mapping, valid = get_keypoint_mapping(model_type="smplx")
            kps = np.zeros((num_frames, 25 + 19, 3), dtype=full_joints_gt_2d.dtype)
            kps[:, valid] = full_joints_gt_2d[:, mapping[valid]]

            centers, scales = crop_bbox_seq(
                centers, scales, kps, extreme_crop_lvl=self.cfg.extreme_crop_lvl,
                ratio=self.cfg.extreme_crop_seq_ratio
            )

        shift_centers = centers.copy()

        for i in range(num_frames):
            imgname = imgnames[i]
            center = centers[i]
            scale = scales[i]

            img_path = os.path.join(
                self.dataset_root,
                "images",
                folder,
                "png",
                imgname,
            )
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if "closeup" in scene:
                # rotate the image
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w, c = img.shape

            if self.split == "train" and self.cfg.extreme_crop_aug:
                tx = np.clip(np.random.randn(), -1.0, 1.0) * self.cfg.trans_factor
                ty = np.clip(np.random.randn(), -1.0, 1.0) * self.cfg.trans_factor
                center[0] += tx * scale * 200
                center[1] += ty * scale * 200
                # center[0] += tx * w
                # center[1] += ty * h
                center[0] = np.clip(center[0], 0, w - 1)
                center[1] = np.clip(center[1], 0, h - 1)

                centers[i] = center

            shift_centers[i, 0] = centers[i, 0] - w / 2
            shift_centers[i, 1] = centers[i, 1] - h / 2

            res = self.cfg.get("crop_res", (224, 224))
            img, crop_img = crop(img, center, scale, res=res)


            img_path_list.append(img_path)
            crop_img_list.append(crop_img)

        # stack the image patches and preprocess to be input to resnet
        crop_img_list = np.stack(crop_img_list)  # stack the image patches
        crop_img_list = crop_img_list.transpose((0, 3, 1, 2))  # permute dimensions
        crop_img_list = crop_img_list.astype(np.float32)  # convert to float32
        crop_img_list /= 255.0  # normalize pixel values to range [0, 1]

        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        crop_img_list = (crop_img_list - mean) / std

        crop_img_list = torch.from_numpy(crop_img_list).float()

        # bounding box information
        centers = torch.from_numpy(centers).float()
        scales = torch.from_numpy(scales).float()
        shift_centers = torch.from_numpy(shift_centers).float()
        focal = 0.5 * (K[:, 0, 0] + K[:, 1, 1])
        bbox = torch.stack(
            [shift_centers[:, 0], shift_centers[:, 1], scales * 200.0], dim=-1
        ) / focal.unsqueeze(-1)

        batch = {}

        # camera + coordinate transformations
        batch.update(
            {
                # intrinsics
                "K": K,
                "dist_coeffs": torch.zeros((8)).float(), # no distortion in BEDLAM
                # camera extrinsics
                "cam2world": torch.from_numpy(cam2world).float(),
                "world2cano": self.world2cano,
            }
        )

        # mesh
        batch.update(mesh_dict)

        rotation = batch["rotation"]
        translation = batch["translation"]
        _, cano_traj_clean = self.global_traj_helper.get_cano_traj_repr(
            rotation, translation
        )
        batch["cano_traj_clean"] = cano_traj_clean.squeeze(0).float()

        # images
        batch.update(
            {
                "body_idx": str(sub),
                "img_paths": img_path_list,
                "center": centers,
                "scale": scales,
                "crop_imgs": crop_img_list,
            }
        )

        batch["seq_name"] = self.seq_name

        batch["bbox"] = bbox
        batch["has_transl"] = True  # for compatibility with other datasets
        batch["true_params"] = True # neutral annotation 


        # foot contact labels
        if rotation.shape[0] > 1:
            joints_gt_cam = clip_data["joints_cam"]
            joints_gt_world = joints_gt_cam @ cam2world[:, :3, :3].transpose(0, 2, 1) + cam2world[:, :3, 3][:, None]
            joints_gt_world = torch.from_numpy(joints_gt_world).float()
            contact_label = get_contact_label(joints_gt_world, vel_thres=self.cfg.contact_vel_thres, contact_hand=self.cfg.contact_hand)
            batch["contact_label"] = contact_label.float()

        return batch


class DataModuleBEDLAM(LightningDataModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg
        self.debug = debug

        # train-test split from BEDLAM-CLIFF
        # ref: https://github.com/pixelite1201/BEDLAM/tree/master/configs/bedlam_cliff_x.yaml
        self.train_scenes = [
            "static-hdri",
            # "agora-bfh", # no agora training here
            "zoom-suburbd",
            "closeup-suburba",
            "closeup-suburbb",
            "closeup-suburbc",
            "closeup-suburbd",
            "closeup-gym",
            "zoom-gym",
            "static-gym",
            "static-office",
            "orbit-office",
            "static-hdri-zoomed",
            "pitchup-stadium",
            "pitchdown-stadium",
            "static-hdri-bmi",
            "closeup-suburbb-bmi",
            "closeup-suburbc-bmi",
            "static-suburbd-bmi",
            "zoom-gym-bmi",
            "static-office-hair",
            "zoom-suburbd-hair",
            # "static-gym-hair",
            "orbit-archviz-15",
            "orbit-archviz-19",
            "orbit-archviz-12",
            "orbit-archviz-10",
        ]
        if self.cfg.clip_len > 1:
            # multiple frames, need a consistent camera space
            self.train_scenes = list(
                filter(
                    lambda x: not x.startswith("orbit") and not x.startswith("zoom"),
                    self.train_scenes,
                )
            )

        self.val_scenes = self.test_scenes = ["static-gym-hair"]

        if self.debug:
            self.train_scenes = self.val_scenes = self.test_scenes = [
                "closeup-suburba",
            ]

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            if not hasattr(self, "train_dataset"):
                self.train_dataset = DataloaderBEDLAM(
                    self.cfg,
                    scenes=self.train_scenes,
                    split="train",
                )
        if stage in [None, "fit", "validate"]:
            if not hasattr(self, "val_dataset"):
                self.val_dataset = DataloaderBEDLAM(
                    self.cfg,
                    scenes=self.val_scenes,
                    split="val",
                )
        if stage in [None, "test"]:
            if not hasattr(self, "test_dataset"):
                self.test_dataset = DataloaderBEDLAM(
                    self.cfg,
                    scenes=self.test_scenes,
                    split="test",
                )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count(), self.cfg.num_workers),
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), self.cfg.num_workers),
            # num_workers=1,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )
