import torch
from torch.utils import data
from tqdm import tqdm, trange
import glob
import pickle as pkl
import os
import numpy as np

from lightning import LightningDataModule

from ..utils import GlobalTrajHelper, cano_seq_smplx, get_bm_params, get_contact_label
from ..utils.body_model import BodyModel


class DataloaderAMASS(data.Dataset):
    def __init__(
        self,
        cfg,
        debug=False,
        split="train",
        spacing=1,
    ):
        self.cfg = cfg

        # self.preprocessed_amass_root = cfg.dataset_root
        self.preprocessed_amass_root = "datasets/mask_transformer/AMASS"
        self.split = split
        self.clip_len = cfg.clip_len
        self.spacing = spacing

        self.device = "cpu"
        self.bm_neutral = BodyModel(
            bm_path=cfg.bm_path,
            model_type=cfg.model_type,
            gender="neutral",
        )

        self.normalize = cfg.normalize
        self.canonical = cfg.canonical

        ######################################## read data and compute the motion reprentations
        amass_train_datasets = [
            "HumanEva",
            "HDM05",
            "MoSh",
            "Transitions",
            "ACCAD",
            "BMLhandball",
            "BMLmovi",
            "BMLrub",
            "CMU",
            "DFaust",
            "Eyes_Japan_Dataset",
            "PosePrior",
            "SSM",
            "GRAB",
            "SOMA",
        ]
        amass_test_datasets = ["TCDHands", "TotalCapture", "SFU"]
        if split == "train":
            amass_datasets = amass_train_datasets if not debug else ["HumanEva"]
        else:
            amass_datasets = amass_test_datasets if not debug else ["TotalCapture"]
            # amass_datasets = ["HumanEva"] 

        self.global_traj_helper = GlobalTrajHelper()
        self.read_data(amass_datasets)

    def divide_clip(self, dataset_name="HumanEva"):
        preprocessed_amass_joints_dir = os.path.join(
            self.preprocessed_amass_root, "pose_data_fps_30"
        )
        preprocessed_amass_smpl_dir = os.path.join(
            self.preprocessed_amass_root, "smpl_data_fps_30"
        )
        seqs_path = glob.glob(
            os.path.join(preprocessed_amass_smpl_dir, dataset_name, "*/*.npy")
        )  # name list of all npz sequence files in current dataset
        seqs_path = sorted(seqs_path)
        for path in seqs_path:
            seq_name = path.split("/")[-2]
            npy_name = path.split("/")[-1]
            path_joints = os.path.join(
                preprocessed_amass_joints_dir, dataset_name, seq_name, npy_name
            )
            path_smplx = path
            seq_joints = np.load(path_joints)  # [seq_len, 25, 3]
            seq_smplx = np.load(path_smplx)  # [seq_len, 178]
            
            N = len(seq_smplx)  # total frame number of the current sequence
            # divide long sequences into sub clips
            if N >= self.clip_len:
                num_valid_clip = N // self.clip_len
                for i in range(num_valid_clip):
                    joints_clip = seq_joints[
                        (self.clip_len * i) : self.clip_len * (i + 1)
                    ]  # [clip_len, 25, 3]
                    smplx_clip = seq_smplx[
                        (self.clip_len * i) : self.clip_len * (i + 1)
                    ]  # [clip_len, 178]
                    self.joints_clip_list.append(joints_clip)
                    self.smplx_clip_list.append(smplx_clip)
            else:
                continue

    def canonicalize_motion(self):
        self.cano_smplx_clip_list = []
        for i in trange(0, len(self.smplx_clip_list), self.spacing):
            source_data_joints = self.joints_clip_list[i][:, 0:22, :]
            source_data_smplx = self.smplx_clip_list[i]
            smplx_params_dict = {
                "global_orient": source_data_smplx[:, 0:3],
                "transl": source_data_smplx[:, 3:6],
                "betas": source_data_smplx[:, 6:16],
                "body_pose": source_data_smplx[:, 16 : (16 + 63)],
            }  # for the clip

            if self.split == "test":
                cano_positions, cano_smplx_params_dict = cano_seq_smplx(
                    positions=source_data_joints,
                    smplx_params_dict=smplx_params_dict,
                )
                self.cano_smplx_clip_list.append(cano_smplx_params_dict)
            else:
                self.cano_smplx_clip_list.append(smplx_params_dict)

            # this canonical space is for better visualization 

    def read_data(self, amass_datasets):
        preprocess_dir = os.path.join(
            os.path.dirname(self.preprocessed_amass_root), "amass_preprocess"
        )
        os.makedirs(preprocess_dir, exist_ok=True)
        mean_std_path = os.path.join(preprocess_dir, f"amass_cum_traj.pkl")
        self.joints_clip_list = []
        self.smplx_clip_list = []
        for dataset_name in tqdm(amass_datasets):
            self.divide_clip(dataset_name)
        self.canonicalize_motion()
        self.n_samples = len(self.cano_smplx_clip_list)

        print(
            "[INFO] {} set: get {} sub clips in total.".format(
                self.split, self.n_samples
            )
        )

        if self.split == "train":
            if os.path.exists(mean_std_path):
                with open(mean_std_path, "rb") as f:
                    mean_std_dict = pkl.load(f)
                self.Mean = mean_std_dict["Mean"]
                self.Std = mean_std_dict["Std"]
            else:
                # compute mean and std, save
                cano_traj_clean_list = []
                for idx in trange(len(self.cano_smplx_clip_list)):
                    smplx_params_dict = self.cano_smplx_clip_list[idx]
                    bm_params = get_bm_params(smplx_params_dict)
                    mesh_dict = self.create_mesh(bm_params.copy()) 
                    rotation = mesh_dict["rotation"]
                    translation = mesh_dict["translation"]
                    _, cano_traj_clean = self.global_traj_helper.get_cano_traj_repr(
                        rotation,
                        translation,
                        normalize=False,
                    )
                    cano_traj_clean_list.append(cano_traj_clean)

                global_repr_list = np.concatenate(cano_traj_clean_list, axis=0)
                self.Mean = np.mean(global_repr_list, axis=(0, 1), keepdims=True)
                self.Std = np.std(global_repr_list, axis=(0, 1), keepdims=True)

                mean_std_dict = {"Mean": self.Mean, "Std": self.Std}
                with open(mean_std_path, "wb") as f:
                    pkl.dump(mean_std_dict, f)
                self.global_traj_helper = GlobalTrajHelper() 
        else:
            assert os.path.exists(mean_std_path), "mean and std file not found"
            with open(mean_std_path, "rb") as f:
                mean_std_dict = pkl.load(f)
            self.Mean = mean_std_dict["Mean"]
            self.Std = mean_std_dict["Std"]

    def create_mesh(self, bm_params_dict):
        translation = bm_params_dict["transl"]
        rotation = bm_params_dict["global_orient"]

        bm_params_dict["transl"] = None
        if self.canonical:
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

        return mesh_dict

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        smplx_params_dict = self.cano_smplx_clip_list[index]
        bm_params = get_bm_params(smplx_params_dict)
        mesh_dict = self.create_mesh(bm_params.copy())

        batch = {}
        batch.update(mesh_dict)

        rotation = batch["rotation"]
        translation = batch["translation"]
        _, cano_traj_clean = self.global_traj_helper.get_cano_traj_repr(
            rotation, translation
        )

        cano_traj_clean = cano_traj_clean.squeeze(0).float()
        batch["cano_traj_clean"] = cano_traj_clean

        # compute foot contact labels
        if rotation.shape[0] > 1:
            local_joints = mesh_dict["local_joints"]
            global_joints = local_joints @ rotation.transpose(-1, -2) + translation
            contact_label = get_contact_label(global_joints, vel_thres=self.cfg.contact_vel_thres, contact_hand=self.cfg.contact_hand)
            batch["contact_label"] = contact_label.float()

        return batch


class DataModuleAMASS(LightningDataModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg
        self.debug = debug

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            if not hasattr(self, "train_dataset"):
                self.train_dataset = DataloaderAMASS(
                    self.cfg,
                    debug=self.debug,
                    split="train",
                )
        if stage in [None, "fit", "validate"]:
            if not hasattr(self, "val_dataset"):
                self.val_dataset = DataloaderAMASS(
                    self.cfg,
                    debug=self.debug,
                    split="val",
                    spacing=2,
                )
        if stage in [None, "test"]:
            if not hasattr(self, "test_dataset"):
                self.test_dataset = DataloaderAMASS(
                    self.cfg,
                    debug=self.debug,
                    split="test",
                    spacing=self.cfg.spacing,
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
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), self.cfg.num_workers),
            drop_last=True,
            pin_memory=True,
        )
