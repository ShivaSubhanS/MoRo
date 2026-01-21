import os
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

from . import DataModuleAMASS, DataModuleEgoBody, DataModuleBEDLAM, DataModuleHMR, DataModuleRich, DataModuleProx


def create_datamodule(cfg, debug, name=None):
    name = name or cfg.name
    if name == "amass":
        return DataModuleAMASS(cfg, debug)
    elif name == "egobody":
        return DataModuleEgoBody(cfg, debug)
    elif name == "bedlam":
        return DataModuleBEDLAM(cfg, debug)
    elif name in ["coco", "h36m", "mpi_inf_3dhp", "mpii"]:
        return DataModuleHMR(cfg, name=name)
    elif name == "mixed":
        return DataModuleMixed(cfg, debug)
    elif name == "rich":
        return DataModuleRich(cfg, debug)
    elif name == "prox":
        return DataModuleProx(cfg, debug)
    else:
        raise ValueError(f"Unknown dataset: {name}")

class MixedDataset(Dataset):
    def __init__(self, datasets, probs=None):
        """
        Args:
            datasets: List of datasets
            probs: Optional list of sampling probabilities (default: size-weighted)
        """
        self.datasets = datasets
        if probs is None:
            # probs = [len(ds) for ds in datasets] # proportional to dataset size
            probs = [1.0 for ds in datasets] # uniform sampling
        probs = np.array(probs, dtype=np.float64)
        probs /= probs.sum()

        self.probs = probs
        self.total_size = sum(len(ds) for ds in datasets)
        self.sample_plan = self._generate_sample_plan()

    def _generate_sample_plan(self):
        plan = []
        for i, (ds, p) in enumerate(zip(self.datasets, self.probs)):
            num_samples = int(round(p * self.total_size))
            replace = num_samples > len(ds)
            indices = np.random.choice(len(ds), num_samples, replace=replace)
            plan.extend([(i, idx) for idx in indices])
        np.random.shuffle(plan)
        return plan

    def on_epoch_start(self):
        self.sample_plan = self._generate_sample_plan()

    def __len__(self):
        return len(self.sample_plan)

    def __getitem__(self, index):
        ds_id, idx = self.sample_plan[index]
        data = self.datasets[ds_id][idx]
        # data.update({"sample_idx": idx, "dataset_idx": ds_id})
        return data

class DataModuleMixed(LightningDataModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg
        self.debug = debug

        self.sub_datasets = cfg.sub_datasets
        self.datamodules = []
        for name in self.sub_datasets:
            datamodule = create_datamodule(cfg, debug, name=name)
            self.datamodules.append(datamodule)

        self.probs = cfg.get("probs", None)  # Optional, allows weighted sampling

    def setup(self, stage=None):
        for datamodule in self.datamodules:
            datamodule.setup(stage)

        if stage in (None, "fit"):
            if not hasattr(self, "train_dataset"):
                self.train_dataset = MixedDataset(
                    [datamodule.train_dataset for datamodule in self.datamodules],
                    probs=self.probs
                )

        if stage in (None, "fit", "validate"):
            if not hasattr(self, "val_dataset"):
                self.val_datasets = [
                    datamodule.val_dataset for datamodule in self.datamodules if hasattr(datamodule, "val_dataset")
                ]
        if stage in (None, "test"):
            if not hasattr(self, "test_dataset"):
                self.test_datasets = [
                    datamodule.test_dataset for datamodule in self.datamodules if hasattr(datamodule, "test_dataset")
                ]

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
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers_val,
                drop_last=True,
                pin_memory=True,
            )
            for dataset in self.val_datasets
        ] if self.val_datasets else None

    def test_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers_val,
                drop_last=True,
                pin_memory=True,
            )
            for dataset in self.test_datasets
        ] if self.test_datasets else None