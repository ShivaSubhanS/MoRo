from pathlib import Path
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
import cv2
import os
from lightning import LightningDataModule

from ..utils.rich_utils import (
    get_cam2params,
    get_w2az_sahmr,
    parse_seqname_info,
    get_cam_key_wham_vid,
)
from ..utils import (
    crop,
    GlobalTrajHelper,
    get_bm_params,
    get_contact_label,
    update_globalRT_for_smplx,
)
from ..utils.body_model import BodyModel

VID_PRESETS = {
    "easytohard": [
        "test/Gym_013_burpee4/cam_06",
        "test/Gym_011_pushup1/cam_02",
        "test/LectureHall_019_wipingchairs1/cam_03",
        "test/ParkingLot2_009_overfence1/cam_04",
        "test/LectureHall_021_sidebalancerun1/cam_00",
        "test/Gym_010_dips2/cam_05",
    ],
    "debug": [
        # "test/Gym_013_burpee4/cam_06",
        # "test/Gym_012_lunge1/cam_05",

        "test/Gym_013_burpee4/cam_06",
        # "test/Gym_011_pushup1/cam_02",
        # "test/LectureHall_019_wipingchairs1/cam_03",
        # "test/ParkingLot2_009_overfence1/cam_04",
        # "test/LectureHall_021_sidebalancerun1/cam_00",
        # "test/Gym_010_dips2/cam_05",
    ]
}

class DataloaderRich(data.Dataset):
    def __init__(self, cfg, vid, label=None, preproc_data=None):
        """
        Args:
            cfg: configuration object
            vid_presets: key in VID_PRESETS for subset selection
        """
        super().__init__()
        self.cfg = cfg
        self.vid = vid
        self.seq_name = os.path.join(*vid.split("/")[-2:])  # e.g., "Gym_013_burpee4/cam_06"
        
        self.clip_len = cfg.clip_len
        self.clip_overlap_len = cfg.overlap_len
        
        # Load evaluation protocol from WHAM labels
        self.rich_dir = Path(cfg.dataset_root) 
        self.label = label 
        self.preproc_data = preproc_data

        # Device setup
        self.device = "cpu"
        self.bm_male = BodyModel(
            bm_path=cfg.bm_path,
            model_type="smplx",
            gender="male",
        )
        self.bm_female = BodyModel(
            bm_path=cfg.bm_path,
            model_type="smplx",
            gender="female",
        )
        self.model_type = "smplx"
        
        self.normalize = cfg.normalize
        self.canonical = cfg.canonical
        self.global_traj_helper = GlobalTrajHelper()

        # Setup dataset index for clips
        self.frame_ids_list = []
        self.param_gt_list = []
        self.bbx_xys_list = []
        self.ids_list = []
        
        seq_length = len(self.label["frame_id"])
        frame_ids = self.label["frame_id"]
        gt_params = self.label["gt_smplx_params"]
        bbx_xys = self.preproc_data["bbx_xys"]

        # Create clips with multi-frame parameters
        clip_idx = 0
        while True:
            start = clip_idx * (self.clip_len - self.clip_overlap_len)
            end = start + self.clip_len
            end = min(end, seq_length)  
            self.frame_ids_list.append(frame_ids[start:end].numpy())
            # Extract clip parameters directly
            param_gt_clip = {}
            for key in gt_params:
                param_gt_clip[key] = gt_params[key][start:end].numpy()
            self.param_gt_list.append(param_gt_clip)
            self.bbx_xys_list.append(bbx_xys[start:end])
            self.ids_list.append(list(range(start, end)))
            clip_idx += 1
            # with RoPE, uncomment here to allow incomplete last clip
            if end >= seq_length:
                break
        self.n_samples = clip_idx

        # Prepare ground truth motion in ay-coordinate
        self.w2az = get_w2az_sahmr(self.rich_dir)
        self.cam2params = get_cam2params(self.rich_dir)
        seqname_info = parse_seqname_info(self.rich_dir, skip_multi_persons=True)
        self.seqname_to_scanname = {k: v[0] for k, v in seqname_info.items()}

    def __len__(self):
        return self.n_samples

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

    def __getitem__(self, idx):
        # Get clip data
        vid = self.vid
        frame_ids = self.frame_ids_list[idx]
        param_gt_clip = self.param_gt_list[idx]
        bbx_xys_clip = self.bbx_xys_list[idx] 
        idxs = self.ids_list[idx]
        
        label = self.label
        preproc_data = self.preproc_data
        
        num_frames = len(frame_ids)
        gender = label["gender"]
        
        # Camera and transformation setup
        cam_key = get_cam_key_wham_vid(vid)
        scan_name = self.seqname_to_scanname[vid.split("/")[1]]
        T_w2c, K = self.cam2params[cam_key]
        try:
            assert T_w2c.shape == (4, 4), f"Moving camera for {vid}"
        except AssertionError as e:
            breakpoint()
        T_w2az = self.w2az[scan_name]
        

        # Transform SMPLX parameters from world to camera coordinates
        # First create body model to get joints for transformation
        bm_params_world = get_bm_params(param_gt_clip)
        bm_model = self.bm_male if gender == "male" else self.bm_female
        smpl_output_world = bm_model(**bm_params_world)
        joints_gt_world = smpl_output_world.joints.clone().detach()  # [F, 22, 3]
        
        # Transform parameters to camera coordinate
        param_gt_cam = update_globalRT_for_smplx(
            param_gt_clip,
            T_w2c.detach().cpu().numpy(),
            delta_T=joints_gt_world[:, 0].detach().cpu().numpy() - param_gt_clip["transl"],
        )
        
        bm_params_cam = get_bm_params(param_gt_cam)
        
        # Create mesh
        mesh_dict, full_joints_gt_local = self.create_mesh(bm_params_cam.copy(), bm_model)
        
        # Get trajectory representation
        rotation = mesh_dict["rotation"]
        translation = mesh_dict["translation"]
        _, cano_traj_clean = self.global_traj_helper.get_cano_traj_repr(
            rotation, translation
        )

        # Get image paths and crop images
        seq_name = vid.split("/")[1]
        cam_name = vid.split("/")[2]
        img_path_list = []
        crop_img_list = []
        center_list = []
        scale_list = []
        
        # Image preprocessing from bbx_xys
        img_dir = self.rich_dir / "images" / vid 
        img_paths = sorted(img_dir.glob("*.jpeg")) 
        w, h = preproc_data["img_wh"]  # image width, height
        
        for i in range(num_frames):
            frame_id = frame_ids[i]
            # Construct image path - adjust based on actual RICH dataset structure
            img_path = img_paths[frame_id] 
            
            bbx_xys = bbx_xys_clip[i]  # (3,) - x, y, size
            idx = idxs[i]
            

            center = bbx_xys[:2].numpy()  # x, y
            scale = bbx_xys[2].numpy() / 200.  # normalize scale

            if self.cfg.get("crop_bbox", False) and idx % 200 >= 100:

                center[1] -= scale * 50  # Adjust center y for crop
                scale /= 2.
                
            
            # Read and crop image
            try:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                img = np.zeros((h, w, 3), dtype=np.uint8)  # Placeholder if image not found
                breakpoint()
            
            res = self.cfg.get("crop_res", (224, 224))
            img, crop_img = crop(img, center, scale, res=res)
            
            img_path_list.append(str(img_path))
            center_list.append(center)
            scale_list.append(scale)
            crop_img_list.append(crop_img)
        
        center_list = np.stack(center_list).astype(np.float32)
        scale_list = np.stack(scale_list).astype(np.float32)
        
        # Preprocess cropped images
        crop_img_list = np.stack(crop_img_list)
        crop_img_list = crop_img_list.transpose((0, 3, 1, 2))
        crop_img_list = crop_img_list.astype(np.float32)
        crop_img_list /= 255.0
        
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        crop_img_list = (crop_img_list - mean) / std
        
        crop_img_list = torch.from_numpy(crop_img_list).float()
        center_list = torch.from_numpy(center_list).float()
        scale_list = torch.from_numpy(scale_list).float()
        
        # Compute 2D joints for detection
        joints_gt_cam = mesh_dict["local_joints"] @ mesh_dict["rotation"].permute(0, 2, 1) + mesh_dict["translation"]
        joints_gt_2d = joints_gt_cam @ K.T
        joints_gt_2d = joints_gt_2d[:, :, :2] / (joints_gt_2d[:, :, 2:] + 1e-6)
        joints_gt_2d = joints_gt_2d.detach().cpu().numpy()
        
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

        # Prepare batch dictionary
        batch = {
            # Basic info
            "body_idx": seq_name,
            "img_paths": img_path_list,
            "seq_name": self.seq_name,
            "center": center_list,
            "scale": scale_list,
            "crop_imgs": crop_img_list,
            "bbox": bbox,
            "has_transl": True,
            "true_params": False,
            
            # Camera parameters
            "K": K.float().unsqueeze(0).repeat(num_frames, 1, 1),
            "cam2world": T_w2c.float().unsqueeze(0).repeat(num_frames, 1, 1),
            "world2cano": T_w2az.unsqueeze(0).float(),  # This is the w2az transformation
            
            # Trajectory
            "cano_traj_clean": cano_traj_clean.squeeze(0).float(),
        }

        if rotation.shape[0] > 1:
            contact_label = get_contact_label(joints_gt_world, vel_thres=self.cfg.contact_vel_thres, contact_hand=self.cfg.contact_hand)
            batch["contact_label"] = contact_label.float()

        
        # Add mesh data
        batch.update(mesh_dict)
        
        return batch


class DataModuleRich(LightningDataModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg

        self.rich_dir = Path(cfg.dataset_root) 
        self.labels = torch.load(self.rich_dir / "hmr4d_support/rich_test_labels.pt") 
        self.preproc_data = torch.load(self.rich_dir / "hmr4d_support/rich_test_preproc.pt")
        self.vids = select_subset(self.labels, "debug") if debug else select_subset(self.labels)

    def setup(self, stage=None):
        if stage in [None, "test"]:
            if not hasattr(self, "test_dataset"):
                test_datasets = [
                    DataloaderRich(
                        self.cfg,
                        vid=vid,
                        label=self.labels[vid],
                        preproc_data=self.preproc_data[vid],
                    ) for vid in tqdm(self.vids)
                ]

                self.test_dataset = data.ConcatDataset(test_datasets)

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers_val,
            drop_last=False,
            persistent_workers=True,
        )


def select_subset(labels, vid_presets=None):
    vids = list(labels.keys())
    if vid_presets is not None:
        vids = VID_PRESETS[vid_presets]
    return vids
