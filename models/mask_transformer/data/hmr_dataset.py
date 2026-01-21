"""
Adapted from VQ-HPS, for training on single-frame HMR dataset
https://github.com/g-fiche/VQ-HPS/blob/master/vq_hps/data/dataset_hmr.py
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import cv2
import numpy as np

from lightning import LightningDataModule

from ..utils.augmentations import (
    crop,
    rot_aa,
    flip_img,
    flip_pose,
    transform,
    flip_kp,
    flip_17,
)
from ..utils.motion_utils import get_bm_params
from ..utils.crop_utils import get_keypoint_mapping, crop_bbox
from ..utils.body_model import BodyModel
import albumentations as A
import torch
import random

J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]


class DatasetHMR(Dataset):
    def __init__(
        self,
        cfg,
        name,
        split="train",
        proportion=1.0,
    ):
        """Initializes the dataset.

        Args:
            dataset_file (str): Path to the dataset file (should be a .npz file)
            augment (bool, optional): If augment is True, augmentations will be performed during training. Defaults to True.
            flip (bool, optional): If True, images will be randomly flipped during training. Defaults to True.
            proportion (float, optional): Proportion of the dataset used (to reduce the validation for instance). Defaults to 1.0.
        """
        super().__init__()

        self.cfg = cfg

        self.split = split
        self.normalize = cfg.normalize
        self.canonical = cfg.canonical
        self.device = "cpu"

        self.name = name
        dataset_file_dict = {
            "coco": "datasets/mask_transformer/coco/coco_smplx.npz",
            "h36m": "datasets/mask_transformer/h36m_train/h36m_train_smplx.npz",
            "mpi_inf_3dhp": "datasets/mask_transformer/mpi-inf-3dhp/mpi_inf_3dhp_train_smplx.npz",
            "mpii": "datasets/mask_transformer/mpii/mpii_smplx.npz",
        }
        self.dataset_file = dataset_file_dict[name]

        self.augment = cfg.augment
        self.flip = cfg.flip
        self.proportion = proportion

        self.data = np.load(self.dataset_file, allow_pickle=True)

        self.imgname = self.data["imgname"]

        full_len = len(self.imgname)

        new_len = full_len
        sampled_indices = [x for x in range(full_len)]
        if self.proportion != 1:
            new_len = int(self.proportion * full_len)
            sampled_indices = random.sample(range(full_len), new_len)
            self.imgname = self.data["imgname"][sampled_indices]

        self.scale = self.data["scale"][sampled_indices]
        self.center = self.data["center"][sampled_indices]

        if "pose_cam" in self.data:
            self.full_pose = self.data["pose_cam"][sampled_indices]
        elif "pose" in self.data:
            self.full_pose = self.data["pose"][sampled_indices]
        else:
            root_orient = self.data["root_orient"][sampled_indices]
            pose_body = self.data["pose_body"][sampled_indices]
            self.full_pose = np.concatenate([root_orient, pose_body], axis=-1)

        self.gender = "neutral"

        if "j2d" in self.data:
            self.j2d = self.data["j2d"][sampled_indices]
        else:
            self.j2d = self.data["part"][sampled_indices]

        if "betas" in self.data:
            self.betas = self.data["betas"][sampled_indices]
        else:
            self.betas = self.data["shape"][sampled_indices]

        # also parse global translation and focal length if available
        # only coco and mpii
        self.has_transl = "global_t" in self.data
        if "global_t" in self.data:
            self.transl = self.data["global_t"][sampled_indices]
        if "focal_l" in self.data:
            self.focal = self.data["focal_l"][sampled_indices]

        self.is_train = self.split == "train"

        print(f"{len(self.imgname)} training samples found")

        self.rot_factor = 30
        self.scale_factor = 0.25

        self.normalize_vqvae = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalize_resnet = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.bm_neutral = BodyModel(
            bm_path=cfg.bm_path,
            model_type="smplx",
            gender="neutral",
        )

    def __len__(self):
        return len(self.imgname)

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0
        rot = 0.0
        sc = 1.0
        tx = 0.0
        ty = 0.0
        extreme_crop_lvl = 0
        if self.is_train and self.augment:
            if np.random.uniform() <= self.cfg.flip_prob and self.flip:
                flip = 1
            else:
                flip = 0

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = (
                np.clip(np.random.randn(), -2.0, 2.0) * self.cfg.rot_factor
                if np.random.uniform() <= self.cfg.rot_aug_prob
                else 0.0
            )

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = (
                np.clip(
                    np.random.randn(),
                    -1.0,
                    1.0,
                )
                * self.cfg.scale_factor
                + 1.0
            )

            tx = np.clip(np.random.randn(), -1.0, 1.0) * self.cfg.trans_factor
            ty = np.clip(np.random.randn(), -1.0, 1.0) * self.cfg.trans_factor

            do_extreme_crop = np.random.uniform() <= self.cfg.extreme_crop_aug_prob
            extreme_crop_lvl = self.cfg.extreme_crop_lvl if do_extreme_crop else 0

        return flip, rot, sc, tx, ty, extreme_crop_lvl

    def rgb_processing(self, rgb_img, center, scale, rot, flip):
        """Process the image"""
        if self.is_train and self.augment:
            aug_comp = [
                A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                A.ImageCompression(20, 100, p=0.1),
                A.RandomRain(blur_value=4, p=0.1),
                A.MotionBlur(blur_limit=(3, 15), p=0.2),
                A.Blur(blur_limit=(3, 10), p=0.1),
                A.RandomSnow(
                    brightness_coeff=1.5, snow_point_lower=0.2, snow_point_upper=0.4
                ),
            ]
            aug_mod = [
                A.CLAHE((1, 11), (10, 10), p=0.2),
                A.ToGray(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.MultiplicativeNoise(
                    multiplier=[0.5, 1.5], elementwise=False, per_channel=True, p=0.2
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    always_apply=False,
                    p=0.2,
                ),
                A.Posterize(p=0.1),
                A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                A.Equalize(mode="cv", p=0.1),
            ]
            albumentation_aug = A.Compose(
                [A.OneOf(aug_comp, p=0.3), A.OneOf(aug_mod, p=0.3)]
            )
            rgb_img = albumentation_aug(image=rgb_img)["image"]

        res = self.cfg.get("crop_res", (224, 224))
        rgb_img = crop(rgb_img, center, scale, res, rot=rot)

        if flip:
            rgb_img = flip_img(rgb_img)

        rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
        return rgb_img

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype("float32")
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype("float32")
        return pose

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        res = self.cfg.get("crop_res", (224, 224))
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale, res, rot=r)
        # convert to normalized coordinates
        kp[:, 0] = 2.0 * kp[:, 0] / res[0] - 1.0
        kp[:, 1] = 2.0 * kp[:, 1] / res[1] - 1.0
        # kp[:, :-1] = 2.0 * kp[:, :-1] / 224 - 1.0
        # flip the x coordinates
        if f:
            if kp.shape[0] == 24:
                kp = flip_kp(kp)
            elif kp.shape[0] == 17:
                kp = flip_17(kp)
            else:
                print(
                    "Error, the skeleton to be flipped must be in SMPL or COCO25 format"
                )
        kp = kp.astype("float32")
        return kp

    def create_mesh(self, bm_params_dict, canonical=None):
        translation = bm_params_dict["transl"]
        rotation = bm_params_dict["global_orient"]

        bm_params_dict["transl"] = None
        canonical = canonical or self.canonical
        if canonical:
            bm_params_dict["global_orient"] = None

        body_mesh = self.bm_neutral(**bm_params_dict)
        vertices = body_mesh.vertices.clone().detach()
        joints = body_mesh.joints.clone().detach()
        root_pos = body_mesh.joints[:, :1].clone().detach()

        vertices = vertices - root_pos
        joints = joints - root_pos

        if self.normalize:
            vertices_offset = torch.mean(vertices, dim=1, keepdim=True)
            vertices = vertices - vertices_offset
            # vertices_offset = vertices_offset + root_pos
        else:
            vertices_offset = torch.zeros_like(vertices)

        # compute the simplified translation and rotation
        # so that the transformation from local to global is Rx+t
        translation = translation.unsqueeze(1)
        translation = translation + root_pos

        mesh_dict = {
            "mesh": vertices,
            "offset": vertices_offset,
            "local_joints": joints,
            
            "root_pos": root_pos,
            "rotation": rotation,
            "translation": translation,
        }

        return mesh_dict

    def __getitem__(self, index):
        imgname = self.imgname[index]
        img_path = f"{os.path.dirname(self.dataset_file)}/{imgname}"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if "closeup" in img_path:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
        ori_img = img.copy()

        center = self.center[index]
        scale = self.scale[index]

        flip, rot, sc, tx, ty, extreme_crop_lvl = self.augm_params()

        # apply random cropping
        j2d = self.j2d[index]
        # try: 
        #     assert j2d.shape[0] == 24, "Expected 24 keypoints in j2d"
        # except AssertionError as e:
        #     print(f"AssertionError: {e}")
        #     print(f"j2d shape: {j2d.shape}, index: {index}, imgname: {imgname}")
        #     j2d = j2d[:24]
        mapping, valid = get_keypoint_mapping(model_type="smpl24")
        kps = np.zeros((25 + 19, 3), dtype=j2d.dtype)
        kps[valid] = j2d[mapping[valid]]
        center, scale = crop_bbox(center, scale, kps, extreme_crop_lvl)

        center[0] += tx * scale * 200
        center[1] += ty * scale * 200
        # center[0] += tx * w
        # center[1] += ty * h
        center[0] = np.clip(center[0], 0, w - 1)
        center[1] = np.clip(center[1], 0, h - 1)

        img = self.rgb_processing(img, center, sc * scale, rot, flip)

        img = torch.from_numpy(img).float()
        resnet_img = self.normalize_resnet(img)

        j2d = j2d[J24_TO_J17]
        j2d = self.j2d_processing(j2d, center, sc * scale, rot, flip)

        betas = self.betas[index][:10]
        gender = self.gender
        full_pose = self.full_pose[index][:66]
        pose = self.pose_processing(full_pose, rot, flip)

        root_orient = pose[:3]
        pose_body = pose[3:]

        betas = np.expand_dims(betas, 0)
        root_orient = np.expand_dims(root_orient, 0)
        pose_body = np.expand_dims(pose_body, 0)

        transl = (
            self.transl[index][None].astype(np.float32)
            if self.has_transl
            else np.zeros((1, 3)).astype(np.float32)
        )

        smpl_params_dict = {
            "betas": betas,
            "body_pose": pose_body,
            "global_orient": root_orient,
            "transl": transl,
        }
        bm_params = get_bm_params(smpl_params_dict)
        mesh_dict = self.create_mesh(bm_params.copy())

        focal = (
            float(self.focal[index]) if self.has_transl else (w ** 2 + h ** 2) ** 0.5
        )
        # note this is full image focal length, not the cropped one,
        # it's hard to estimate this so we mostly assume we have the ground truth
        cam_center = w / 2, h / 2
        K = torch.eye(3).float()
        K[0, 0] = K[1, 1] = focal
        K[0, 2], K[1, 2] = cam_center

        # bounding box information
        # not correct if rotation is applied
        bbox = np.array([[center[0] - w / 2, center[1] - h / 2, scale * 200.0]]) / focal
        if flip:
            bbox[0, 0] = -bbox[0, 0]

        # compose batch
        batch = {}
        batch.update(mesh_dict)

        batch["j2d"] = torch.from_numpy(j2d).float().unsqueeze(0)

        # dummy camera
        batch.update(
            {
                # intrinsics
                "K": K.unsqueeze(0),
                # camera extrinsics
                "cam2world": torch.eye(4).unsqueeze(0).float(),
                "world2cano": torch.eye(4).float(),
            }
        )

        # image
        batch.update(
            {
                "img_paths": [img_path],
                "center": torch.from_numpy(np.array([center])).float(),
                "scale": torch.from_numpy(np.array([scale])).float(),
                "crop_imgs": resnet_img.unsqueeze(0).float(),
            }
        )

        batch["bbox"] = torch.from_numpy(bbox).float()
        # translation only valid when no augmentation
        batch["has_transl"] = self.has_transl and not self.augment

        return batch


class DataModuleHMR(LightningDataModule):
    def __init__(self, cfg, name):
        super().__init__()
        self.cfg = cfg
        self.name = name

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            if not hasattr(self, "train_dataset"):
                self.train_dataset = DatasetHMR(
                    self.cfg,
                    name=self.name,
                    split="train",
                    proportion=1.0,
                )
        if stage in [None, "fit", "validate"] and self.name in ["coco", "mpii"]:
            if not hasattr(self, "val_dataset"):
                self.val_dataset = DatasetHMR(
                    self.cfg,
                    name=self.name,
                    split="val",
                )
        if stage in [None, "test"] and self.name in ["coco", "mpii"]:
            if not hasattr(self, "test_dataset"):
                self.test_dataset = DatasetHMR(
                    self.cfg,
                    name=self.name,
                    split="test",
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count(), self.cfg.num_workers),
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers_val,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers_val,
            drop_last=True,
            pin_memory=True,
        )
