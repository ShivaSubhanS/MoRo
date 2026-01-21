"""
This dataset is used for training Mesh-VQ-VAE. It selects a random mesh.
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import os
from ..utils.body_model import BodyModel, SMPLH_PATH, NUM_BETAS
import torch

from lightning import LightningDataModule

class DatasetMeshFromSmplx(Dataset):
    def __init__(
        self,
        cfg,
        split="train",
        smplx_male=None,
        smplx_female=None,
        smplx_neutral=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.normalize = self.cfg.normalize
        self.canonical = self.cfg.canonical
        self.pose_hand = self.cfg.pose_hand

        self.read_data(structure=self.cfg.structure)
        self.device = "cpu"

        self.smplx_male = smplx_male.to(self.device)
        self.smplx_female = smplx_female.to(self.device)
        self.smplx_neutral = smplx_neutral.to(self.device)

    def read_data(self, structure="**"):
        """Read the data from the npz files."""
        if self.split == "train":
            train_root = self.cfg.train_path

            if isinstance(train_root, str):
                train_root = [train_root]
            
            # Load individual datasets and track their sizes
            self.individual_data = []
            self.dataset_sizes = []
            
            for path in train_root:
                npz_data = np.load(path, allow_pickle=True)
                data = {k: npz_data[k] for k in npz_data.keys()}
                self.individual_data.append(data)
                self.dataset_sizes.append(len(data["gender"]))

            # Set up weights (default to proportional to dataset size)
            if hasattr(self.cfg, 'weights') and self.cfg.weights is not None:
                if self.cfg.weights == "balanced":
                    self.dataset_weights = np.array([1.0 for _ in self.dataset_sizes])
                else:
                    self.dataset_weights = np.array(self.cfg.weights)
            else:
                # Default weights proportional to dataset sizes
                self.dataset_weights = np.array(self.dataset_sizes, dtype=float)
            self.dataset_weights = self.dataset_weights / self.dataset_weights.sum()

            # Create sample weights for WeightedRandomSampler
            self.sample_weights = []
            self.dataset_indices = []  # Track which dataset each sample comes from
            
            total_size = sum(self.dataset_sizes)
            for i, (data, size) in enumerate(zip(self.individual_data, self.dataset_sizes)):
                weight_per_sample = self.dataset_weights[i] * total_size / size
                self.sample_weights.extend([weight_per_sample] * size)
                self.dataset_indices.extend([i] * size)
            
            self.sample_weights = np.array(self.sample_weights)
            self.dataset_indices = np.array(self.dataset_indices)

            # Concatenate all data
            self.data = {
                k: np.concatenate([d[k] for d in self.individual_data]) for k in self.individual_data[0].keys()
            }

        elif self.split == "val":
            val_path = self.cfg.val_path
            # copy to dict, npz has problem with multiprocess
            npz_data = np.load(val_path, allow_pickle=True)
            data = {k: npz_data[k] for k in npz_data.keys()}
            self.data = data
        elif self.split == "test":
            test_path = self.cfg.test_path
            # copy to dict, npz has problem with multiprocess
            npz_data = np.load(test_path, allow_pickle=True)
            data = {k: npz_data[k] for k in npz_data.keys()}
            self.data = data
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def get_mesh(
        self, gender, body_pose, betas, hand_pose=None, global_orient=None, transl=None
    ):
        """Get the mesh given the SMPL parameters.

        Args:
            gender (str): "m" if male, "f" if female.
            pose_body (np.array): The body pose in SMPL format.
            betas (np.array): The SMPL body shape parameter.
            pose_hand (np.array, optional): The SMPL hand pose parameter. Defaults to None.
            root_orient (np.array, optional): Global orientation. Defaults to None.
            trans (np.array, optional): Global translation. Defaults to None.

        Returns:
            np.array: The 3D coordinates for the 6890 vertices of the SMPL mesh.
        """
        gender = str(gender)
        if gender == "male":
            bm_model = self.smplx_male
        elif gender == "female":
            bm_model = self.smplx_female
        elif gender == "neutral":
            bm_model = self.smplx_neutral

        body_pose = torch.Tensor(body_pose).to(self.device)
        betas = torch.Tensor(betas[:NUM_BETAS][np.newaxis]).to(self.device)
        if hand_pose is not None:
            hand_pose = torch.Tensor(hand_pose).to(self.device)
        if global_orient is not None:
            global_orient = torch.Tensor(global_orient).to(self.device)
        if transl is not None:
            transl = torch.Tensor(transl).to(self.device)

        # pay ATTENTION to the keyword names here
        body = bm_model(
            global_orient=global_orient,
            body_pose=body_pose,
            hand_pose=hand_pose,
            betas=betas,
            transl=transl,
        )
        vertices = body.vertices.squeeze(0).clone().detach().cpu().numpy()
        root_pos = body.joints[0, [0]].clone().detach().cpu().numpy()
        vertices = vertices - root_pos 
        return vertices

    def __len__(self):
        return len(self.data["gender"])

    def __getitem__(self, index):
        """Get a human mesh.

        Args:
            index (_type_): The index of the npz path.

        Returns:
            np.array: A 3D mesh with 6890 vertices in 3 dimensions.
        """
        sampled_data = self.data
        idx = index

        global_orient = sampled_data["root_orient"][idx : idx + 1]
        body_pose = sampled_data["pose_body"][idx : idx + 1]
        betas = sampled_data["betas"][idx]
        gender = sampled_data["gender"][idx]
        gender = str(gender, "utf-8") if isinstance(gender, bytes) else str(gender)
        
        is_neutral = gender == "neutral"
        if self.canonical:
            global_orient = None

        mesh = self.get_mesh(
            gender, body_pose, betas, global_orient=global_orient
        )
        if self.normalize:
            mesh = mesh - np.mean(mesh, axis=0, keepdims=True)

        batch = {
            "mesh": mesh,
            "pose_body": body_pose.reshape(-1, 3),
            "betas": betas,
            "gender": gender,
            "is_neutral": is_neutral
        }
        return batch

    def get_sampler(self):
        """Get a weighted random sampler for training data."""
        if self.split == "train":
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return None


class DataModuleMeshFromSMPLX(LightningDataModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg

        self.build_body_model()

    def build_body_model(self):
        self.smplx_male = BodyModel(self.cfg.bm_path, model_type="smplx", gender="male")
        self.smplx_female = BodyModel(self.cfg.bm_path, model_type="smplx", gender="female")
        self.smplx_neutral = BodyModel(self.cfg.bm_path, model_type="smplx", gender="neutral")

        self.faces = self.smplx_neutral.faces

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = DatasetMeshFromSmplx(
                self.cfg,
                split="train",
                smplx_male=self.smplx_male,
                smplx_female=self.smplx_female,
                smplx_neutral=self.smplx_neutral,
            )
        if stage in [None, "fit", "validate"]:
            self.val_path = self.cfg.val_path
            if isinstance(self.val_path, str):
                self.val_path = [self.val_path]
            self.val_dataset = []
            for path in self.val_path:
                self.cfg.val_path = path
                self.val_dataset.append(
                    DatasetMeshFromSmplx(
                        self.cfg,
                        split="val",
                        smplx_male=self.smplx_male,
                        smplx_female=self.smplx_female,
                        smplx_neutral=self.smplx_neutral,
                    )
                )
        if stage in [None, "test"]:
            self.test_path = self.cfg.test_path
            if isinstance(self.test_path, str):
                self.test_path = [self.test_path]
            self.test_dataset = []
            for path in self.test_path:
                self.cfg.test_path = path
                self.test_dataset.append(
                    DatasetMeshFromSmplx(
                        self.cfg,
                        split="test",
                        smplx_male=self.smplx_male,
                        smplx_female=self.smplx_female,
                        smplx_neutral=self.smplx_neutral,
                    )
                )

    def train_dataloader(self):
        sampler = self.train_dataset.get_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(sampler is None),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=min(os.cpu_count(), self.cfg.num_workers),
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return [
                DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=min(os.cpu_count(), self.cfg.num_workers),
                drop_last=False,
                pin_memory=True,
            )
            for dataset in self.val_dataset
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=min(os.cpu_count(), self.cfg.num_workers),
                drop_last=False,
                pin_memory=True,
            )
            for dataset in self.test_dataset
        ]
