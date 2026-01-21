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
    update_globalRT_for_smplx,
    get_bbox_valid,
    crop,
    GlobalTrajHelper,
    get_bm_params,
    get_keypoint_mapping,
    crop_bbox_seq,
    get_contact_label
)
from ..utils.body_model import BodyModel


class DataloaderEgoBody(data.Dataset):
    def __init__(
        self,
        cfg,
        recordings,
        views=None,
        body_idxs=None,
        split="train",
    ):
        self.cfg = cfg

        self.joints_num = 22

        self.dataset_root = "datasets/mask_transformer/EgoBody" 
        self.recordings = recordings
        if not isinstance(self.recordings, list):
            self.recordings = [self.recordings]
        self.views = views
        if views is not None and not isinstance(self.views, list):
            self.views = [self.views]
        self.body_idxs = body_idxs
        if body_idxs is not None and not isinstance(self.body_idxs, list):
            self.body_idxs = [self.body_idxs]
        self.seq_name = os.path.join(recordings, views, f"body_idx_{body_idxs}") if split == "test" else "train"

        self.split = split
        self.clip_len = cfg.clip_len
        self.clip_overlap_len = cfg.overlap_len

        self.device = "cpu"  # cannot use cuda in multi-process dataloader
        self.bm_male = BodyModel(
            bm_path=cfg.bm_path,
            model_type=cfg.model_type,
            gender="male",
        )
        self.bm_female = BodyModel(
            bm_path=cfg.bm_path,
            model_type=cfg.model_type,
            gender="female",
        )
        self.model_type = "smpl" if cfg.model_type == "smplh" else cfg.model_type

        self.normalize = cfg.normalize
        self.canonical = cfg.canonical

        self.global_traj_helper = GlobalTrajHelper()

        self.init_data()
        if self.split == "test":
            # for testing, read data from scratch
            for recording in tqdm(self.recordings):
                self.read_data(recording)
        else:
            # for training and validation, load preprocessed pickle file
            self.load_data()

        # scene name and floor height
        self.world2cano = torch.tensor(
            [[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1]]
        ).unsqueeze(0).float()

        df = pd.read_csv(os.path.join(self.dataset_root, "data_info_release.csv"))
        recording_name_list = list(df["recording_name"])
        scene_name_list = list(df["scene_name"])
        self.scene_name_dict = dict(zip(recording_name_list, scene_name_list))
        self.floor_height_dict = {
            'seminar_g110': -1.660,
            'seminar_d78': -0.810,
            'seminar_j716': -0.8960,
            'seminar_g110_0315': -0.73,
            'seminar_d78_0318': -1.03,
            'seminar_g110_0415': -0.77
        }

        print(
            "[INFO] {} set: get {} sub clips in total.".format(
                self.split, self.n_samples
            )
        )

    def save_data(self):
        # save data for faster loading...
        save_path = os.path.join(
            os.path.dirname(self.dataset_root),
            "egobody_preprocess",
            f"egobody_{self.model_type}_{self.clip_len}_{self.split}.pkl",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = dict(
            cam2world_dict=self.cam2world_dict,
            master2world_dict=self.master2world_dict,
            cam_intrinsic_dict=self.cam_intrinsic_dict,
            gender_dict=self.gender_dict,
            frame_name_list=self.frame_name_list,
            param_gt_list=self.param_gt_list,
            n_samples=self.n_samples,
            rb_num_clips=self.rb_num_clips,
            rb_list=self.rb_list,
        )
        with open(save_path, "wb") as f:
            pkl.dump(save_dict, f)

    def load_data(self):
        # load data for faster loading...
        print(f"[INFO] Loading preprocessed data for split {self.split}...", flush=True)
        save_path = os.path.join(
            os.path.dirname(self.dataset_root),
            "egobody_preprocess",
            f"egobody_{self.model_type}_{self.clip_len}_{self.split}.pkl",
        )

        if not os.path.exists(save_path):
            for recording in tqdm(self.recordings):
                self.read_data(recording)
            self.save_data()

        with open(save_path, "rb") as f:
            save_dict = pkl.load(f)

        self.cam2world_dict = save_dict["cam2world_dict"]
        self.master2world_dict = save_dict["master2world_dict"]
        self.cam_intrinsic_dict = save_dict["cam_intrinsic_dict"]
        self.gender_dict = save_dict["gender_dict"]

        self.frame_name_list = save_dict["frame_name_list"]
        self.param_gt_list = save_dict["param_gt_list"]

        self.n_samples = save_dict["n_samples"]
        self.rb_num_clips = save_dict["rb_num_clips"]
        self.rb_list = save_dict["rb_list"]

    def init_data(self):
        # initialization for data loading
        self.cam2world_dict = {}  # key: recording + view
        self.master2world_dict = {}  # key: recording
        self.cam_intrinsic_dict = {}  # key: recording + view
        self.gender_dict = {}  # key: recording + body_idx

        self.frame_name_list = []
        self.param_gt_list = []

        self.n_samples = 0
        # rvb: recording view body_idx
        self.rb_num_clips = []
        self.rb_list = []

    def read_data(self, recording):
        df = pd.read_csv(os.path.join(self.dataset_root, "data_info_release.csv"))
        recording_df = df[df["recording_name"] == recording]
        scene_name = recording_df["scene_name"].values[0]
        body_idx_0 = recording_df["body_idx_0"].values[0]
        body_idx_1 = recording_df["body_idx_1"].values[0]
        body_idx_fpv = recording_df["body_idx_fpv"].values[0]

        view_list = self.views or os.listdir(
            os.path.join(self.dataset_root, "kinect_color", recording)
        )
        body_idxs = (body_idx_0, body_idx_1)
        if self.body_idxs is not None:
            body_idxs = [
                body_idx_1 if body_idx else body_idx_0 for body_idx in self.body_idxs
            ]

        for view in view_list:
            recording_view = recording + " " + view

            calib_trans_dir = os.path.join(
                self.dataset_root, "calibrations", recording
            )  # extrinsics
            with open(
                os.path.join(
                    calib_trans_dir,
                    "cal_trans",
                    "kinect12_to_world",
                    scene_name + ".json",
                ),
                "r",
            ) as f:
                master2world = np.asarray(json.load(f)["trans"])

            ######################### load calibration from sub kinect to main kinect (between color cameras)
            # master: kinect 12, sub_1: kinect 11, sub_2: kinect 13, sub_3, kinect 14, sub_4: kinect 15
            if view == "sub_1":
                trans_subtomain_path = os.path.join(
                    calib_trans_dir, "cal_trans", "kinect_11to12_color.json"
                )
            elif view == "sub_2":
                trans_subtomain_path = os.path.join(
                    calib_trans_dir, "cal_trans", "kinect_13to12_color.json"
                )
            elif view == "sub_3":
                trans_subtomain_path = os.path.join(
                    calib_trans_dir, "cal_trans", "kinect_14to12_color.json"
                )
            elif view == "sub_4":
                trans_subtomain_path = os.path.join(
                    calib_trans_dir, "cal_trans", "kinect_15to12_color.json"
                )
            if view != "master":
                with open(os.path.join(trans_subtomain_path), "r") as f:
                    trans_subtomain = np.asarray(json.load(f)["trans"])
                cam2world = np.matmul(master2world, trans_subtomain)  # subcamera2world
            else:
                cam2world = master2world
            cam2world = torch.from_numpy(cam2world).float().to(self.device)
            self.cam2world_dict[recording_view] = cam2world
            ##### egobody gt body is in the master kinect camera coord
            master2world = torch.from_numpy(master2world).float().to(self.device)
            self.master2world_dict[recording] = master2world

            with open(
                os.path.join(
                    self.dataset_root,
                    "kinect_cam_params",
                    "kinect_{}".format(view),
                    "Color.json",
                ),
                "r",
            ) as f:
                color_cam = json.load(f)
            self.cam_intrinsic_dict[recording_view] = color_cam

        for body_idx_gender in body_idxs:
            body_idx = int(body_idx_gender.split(" ")[0])
            gender_gt = body_idx_gender.split(" ")[1]

            recording_body_idx = recording + " " + str(body_idx)
            self.gender_dict[recording_body_idx] = gender_gt

            ######################### see if the target is camera_wearer or interactee to get fitting gt root
            interactee_idx = int(body_idx_fpv.split(" ")[0])
            if body_idx == interactee_idx:
                fitting_gt_root = os.path.join(
                    self.dataset_root,
                    # "smplx_interactee_{}".format(self.split),
                    f"{self.model_type}_interactee",
                    recording,
                    "body_idx_{}".format(body_idx),
                    "results",
                )
            else:
                fitting_gt_root = os.path.join(
                    self.dataset_root,
                    # "smplx_camera_wearer_{}".format(self.split),
                    f"{self.model_type}_camera_wearer",
                    recording,
                    "body_idx_{}".format(body_idx),
                    "results",
                )

            frame_list = os.listdir(fitting_gt_root)
            frame_list.sort()

            frame_total = len(frame_list)
            # print("[INFO] total frames of current sequence: ", frame_total)
            frame_name_list = []
            param_gt_list = []
            for cur_frame_name in frame_list:
                frame_name_list.append(cur_frame_name)
                with open(
                    os.path.join(fitting_gt_root, cur_frame_name, "000.pkl"),
                    "rb",
                ) as f:
                    param_gt = pkl.load(f)

                param_gt_list.append(param_gt)

            ############################### divide sequence into short clips with overlapping window
            seq_idx = 0
            while 1:
                start = seq_idx * (self.clip_len - self.clip_overlap_len)
                end = start + self.clip_len
                if end > len(frame_name_list):
                    if self.split == "test":
                        # for testing, we can use the last part of the sequence 
                        self.frame_name_list.append(frame_name_list[start:])
                        self.param_gt_list.append(param_gt_list[start:])
                        seq_idx += 1
                    break
                self.frame_name_list.append(frame_name_list[start:end])
                self.param_gt_list.append(param_gt_list[start:end])
                seq_idx += 1
            self.n_samples += seq_idx
            self.rb_num_clips.append(seq_idx)
            self.rb_list.append(recording_body_idx)

        print(
            "[INFO] EgoBody sequence {}: get {}/{} sub clips in total.".format(
                recording, seq_idx * len(body_idxs), self.n_samples
            )
        )

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

    def get_rb(self, index):
        clip_idx = 0
        for rb_idx, rb_clips in enumerate(self.rb_num_clips):
            if clip_idx + rb_clips > index:
                rb = self.rb_list[rb_idx]
                return rb.split(" ")
            clip_idx += rb_clips
        raise ValueError("Index out of range")

    def __len__(self):
        if self.views is not None:
            return self.n_samples * len(self.views)
        return self.n_samples

    def __getitem__(self, index):
        if self.views is None:
            # training, sample view from possible views
            recording, body_idx = self.get_rb(index)
            view_list = os.listdir(
                os.path.join(self.dataset_root, "kinect_color", recording)
            )
            view = np.random.choice(view_list)
        else:
            view = self.views[index // self.n_samples]
            index = index % self.n_samples
            recording, body_idx = self.get_rb(index)

        # clip-based
        frame_names = self.frame_name_list[index]
        param_gt_clip = self.param_gt_list[index]

        # recording-based
        recording_view = recording + " " + view
        recording_body_idx = recording + " " + body_idx
        gender = self.gender_dict[recording_body_idx]

        cam2world = self.cam2world_dict[recording_view]

        master2world = self.master2world_dict[recording]

        cam_intrinsic = self.cam_intrinsic_dict[recording_view]
        K = np.array(cam_intrinsic["camera_mtx"]).astype(np.float32)  # K includes f and c
        dist_coeffs = np.array(cam_intrinsic["k"]).astype(np.float32)

        ###### get smplx params for clip
        ###### gt is originally in the master kinect camera coord
        param_gt_master = {}
        for key in ["transl", "global_orient", "betas", "body_pose"]:
            param_gt_master[key] = np.concatenate([param[key] for param in param_gt_clip])
            if key == "body_pose":
                param_gt_master[key] = param_gt_master[key][:, :63]  # discard hand pose

        bm_model = self.bm_male if gender == "male" else self.bm_female
        pelvis = bm_model(betas=torch.from_numpy(param_gt_master["betas"]).float()).joints[:, 0]

        master2cam = torch.linalg.solve(cam2world, master2world)
        param_gt_cam = update_globalRT_for_smplx(
            param_gt_master,
            master2cam.detach().cpu().numpy(),
            delta_T=pelvis.detach().cpu().numpy(),
        )
        bm_params_cam = get_bm_params(param_gt_cam)

        mesh_dict, full_joints_gt_local = self.create_mesh(bm_params_cam.copy(), bm_model)
        joints_gt_cam = mesh_dict["local_joints"] @ mesh_dict["rotation"].permute(0, 2, 1) + mesh_dict["translation"]
        full_joints_gt_cam = full_joints_gt_local @ mesh_dict["rotation"].permute(0, 2, 1) + mesh_dict["translation"]

        # 22 joints, for computing gt bbox
        K_tensor = torch.tensor(K).to(self.device)
        joints_gt_2d = joints_gt_cam @ K_tensor.T
        joints_gt_2d = joints_gt_2d[:, :, :2] / (joints_gt_2d[:, :, 2:] + 1e-6)
        joints_gt_2d = joints_gt_2d.detach().cpu().numpy()

        # full smplx joints, for cropping augmentation
        full_joints_gt_2d = full_joints_gt_cam @ K_tensor.T
        full_joints_gt_2d = full_joints_gt_2d[:, :, :2] / (full_joints_gt_2d[:, :, 2:] + 1e-6)
        full_joints_gt_2d = full_joints_gt_2d.detach().cpu().numpy()
        full_joints_gt_2d = np.concatenate([full_joints_gt_2d, np.ones((full_joints_gt_2d.shape[0], full_joints_gt_2d.shape[1], 1))], axis=-1)

        # get images
        num_frames = len(frame_names)
        
        img_path_list = []
        center_list = []
        scale_list = []
        crop_img_list = []
        h, w = 1080, 1920

        # load the keypoint derived bounding boxes
        bbox_center, bbox_scale = None, None
        if self.split != "train" and not self.cfg.get("gt_bbox", True):
            # load bbox from preprocessed npz file
            bbox_path = os.path.join(self.dataset_root, 
                "keypoints_cleaned", recording, view, f"bbox_idx{body_idx}.npz")
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
                "kinect_color",
                recording,
                view,
                frame_name + ".jpg",
            )

            scale_factor_bbox = 1.2
            res = self.cfg.get("crop_res", (224, 224))
            
            if self.split == "train" or self.cfg.get("gt_bbox", True):
                center, scale, _, _ = get_bbox_valid(
                    joints_2d, rescale=scale_factor_bbox
                )
            else:
                center = bbox_center[frame_name]
                scale = bbox_scale[frame_name]


            img_path_list.append(img_path)
            center_list.append(center)
            scale_list.append(scale)

        center_list = np.stack(center_list).astype(np.float32)
        scale_list = np.stack(scale_list).astype(np.float32)

        if self.split == "test":
            keypoints_list = []
            for i in range(num_frames):
                frame_name = frame_names[i]
                kps_path = os.path.join(self.dataset_root, "keypoints_cleaned", recording, view, frame_name + "_keypoints.json")
                with open(kps_path, 'r') as f:
                    kps_data = json.load(f)
            
                body_kps = np.array(kps_data["people"][int(body_idx)]["pose_keypoints_2d"]).astype(np.float32).reshape((-1, 3))
                valid = body_kps[:, 2] > 0.2
                body_kps[~valid] = 0 # remove unreliable keypoints

                if valid.any():
                    body_kps[valid, :2] = cv2.undistortImagePoints(
                        body_kps[valid, :2].reshape(-1, 1, 2), K, dist_coeffs).reshape(-1, 2)

                keypoints_list.append(body_kps)
            keypoints_list = np.stack(keypoints_list).astype(np.float32)
        
        # do cropping if needed
        if self.split == "train" and self.cfg.extreme_crop_aug:
            if np.random.rand() < self.cfg.extreme_crop_aug_prob:
                # use random cropping for training
                mapping, valid = get_keypoint_mapping(model_type="smplx")
                kps = np.zeros((num_frames, 25 + 19, 3), dtype=full_joints_gt_2d.dtype)
                kps[:, valid] = full_joints_gt_2d[:, mapping[valid]]
                center_list, scale_list = crop_bbox_seq(
                    center_list,
                    scale_list,
                    kps,
                    extreme_crop_lvl=self.cfg.extreme_crop_lvl,
                    ratio=self.cfg.extreme_crop_seq_ratio,
                )
                
            tx = np.clip(np.random.randn(num_frames), -1.0, 1.0) * self.cfg.trans_factor
            ty = np.clip(np.random.randn(num_frames), -1.0, 1.0) * self.cfg.trans_factor
            center_list[:, 0] += tx * scale_list * 200
            center_list[:, 1] += ty * scale_list * 200
            center_list[:, 0] = np.clip(center_list[:, 0], 0, w - 1)
            center_list[:, 1] = np.clip(center_list[:, 1], 0, h - 1)

        for i in range(num_frames):
            img_path = img_path_list[i]
            
            try:
                if not os.path.exists(img_path):
                    # if the image is not found, use black image
                    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    print(f"[WARNING] image not found: {img_path}, use black image instead.")
                else:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.undistort(img.copy(), K, dist_coeffs) 
            except:
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                print(f"[WARNING] image error: {img_path}, use black image instead.")

            center = center_list[i]
            scale = scale_list[i]
            img, crop_img = crop(img, center, scale, res=res)


            crop_img_list.append(crop_img)

        center_list = torch.from_numpy(center_list).float()
        scale_list = torch.from_numpy(scale_list).float()

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
        focal = 0.5 * (K[0, 0] + K[1, 1]) # the focal length is roughly isometric
        cam_center = torch.zeros_like(center_list)
        # the camera center is not the image center on EgoBody!!!
        cam_center[:, 0] = K_tensor[0, 2]  # cx
        cam_center[:, 1] = K_tensor[1, 2]  # cy
        bbox = (
            torch.cat(
                [center_list - cam_center, scale_list.unsqueeze(-1) * 200.0], dim=1
            )
            / focal
        )

        batch = {}

        # camera + coordinate transformations
        batch.update(
            {
                # intrinsics
                "K": K_tensor.float().unsqueeze(0).repeat(num_frames, 1, 1),
                "dist_coeffs": torch.from_numpy(dist_coeffs).float(),
                # camera extrinsics
                "cam2world": cam2world.float().unsqueeze(0).repeat(num_frames, 1, 1),
                # "world2cano": torch.from_numpy(world2cano).float(),
                # "cam2world": torch.eye(4).float().unsqueeze(0).repeat(num_frames, 1, 1),
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
                "body_idx": body_idx,
                "img_paths": img_path_list,
                "center": center_list,
                "scale": scale_list,
                "crop_imgs": crop_img_list,
            }
        )
        batch["seq_name"] = self.seq_name

        batch["bbox"] = bbox
        batch["has_transl"] = True  # for compatibility with other datasets
        batch["true_params"] = False # gendered annotation 

        # cleaned OpenPose detection
        if self.split == "test": 
            batch["keypoints"] = keypoints_list

        # foot contact labels
        if rotation.shape[0] > 1:
            joints_gt_world = joints_gt_cam @ cam2world[:3, :3].T + cam2world[:3, 3]
            contact_label = get_contact_label(joints_gt_world, vel_thres=self.cfg.contact_vel_thres, contact_hand=self.cfg.contact_hand)
            batch["contact_label"] = contact_label.float()

        return batch


class DataModuleEgoBody(LightningDataModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg
        self.debug = debug
        self.dataset_root = "datasets/mask_transformer/EgoBody"

        if self.debug:
            self.train_recs = [
                "recording_20211004_S19_S06_01",
            ]
            self.val_recs = [
                "recording_20220315_S21_S30_03",
            ]
            self.test_data = ["recording_20220318_S33_S34_01"], ["sub_3"], [0]
        else:
            self.train_recs, self.val_recs = self.read_splits()
            self.test_data = self.read_test_data()

    def read_splits(self):
        df = pd.read_csv(os.path.join(self.dataset_root, "data_splits.csv"))
        train_recs = list(df["train"])
        val_recs = list(df["val"])

        # filter out nans
        train_recs = [
            train_rec for train_rec in train_recs if isinstance(train_rec, str)
        ]
        val_recs = [val_rec for val_rec in val_recs if isinstance(val_rec, str)]
        return train_recs, val_recs

    def read_test_data(self):
        test_idx = self.cfg.get("test_idx", 14)
        if test_idx == -1:
            return [self.cfg.recording], [self.cfg.view], [self.cfg.body_idx]

        df = pd.read_csv(os.path.join(self.dataset_root, "egobody_occ_info.csv"))

        if test_idx is None:
            index = slice(None)
        elif isinstance(test_idx, int):
            index = [test_idx]
        else:
            index = test_idx 

        recordings = df["recording_name"].values[index]
        views = df["view"].values[index]
        body_idxs = df["target_idx"].values[index]
        return recordings, views, body_idxs

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            if not hasattr(self, "train_dataset"):
                self.train_dataset = DataloaderEgoBody(
                    self.cfg,
                    recordings=self.train_recs,
                    split="train",
                )
        if stage in [None, "fit", "validate"]:
            if not hasattr(self, "val_dataset"):
                self.val_dataset = DataloaderEgoBody(
                    self.cfg,
                    recordings=self.val_recs,
                    views=["master"],
                    body_idxs=[0],
                    split="val",
                )
        if stage in [None, "test"]:
            if not hasattr(self, "test_dataset"):
                recordings, views, body_idxs = self.test_data
                test_datasets = [
                    DataloaderEgoBody(
                        self.cfg,
                        recordings=recording,
                        views=view,
                        body_idxs=body_idx,
                        split="test",
                    )
                    for recording, view, body_idx in zip(
                        recordings, views, body_idxs
                    )
                ]

                self.test_dataset = data.ConcatDataset(test_datasets)

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
            num_workers=min(os.cpu_count(), self.cfg.num_workers_val),
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers_val,
            drop_last=False,
        )
