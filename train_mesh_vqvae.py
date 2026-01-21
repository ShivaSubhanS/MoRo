"""
Code for training the Mesh-VQ-VAE.
"""

from models.mesh_vq_vae import (
    DataModuleMeshFromSMPLX, 
    MeshVQVAEModule,
)
import hydra
import os
import numpy as np
import torch

import time
from omegaconf import OmegaConf, DictConfig
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb


@hydra.main(
    config_path="configs/mesh_vq_vae",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    if "seed" not in cfg:
        cfg.seed = int(time.time() * 1000) % 1000
    L.seed_everything(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.exp_dir = os.path.join("./exp/mesh_vqvae", cfg.name)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    save_dir = os.path.join(cfg.exp_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    print("Creating data module...")
    datamodule = DataModuleMeshFromSMPLX(cfg.dataset)

    print("Creating lightning module...")
    mesh_vqvae_module = MeshVQVAEModule(
        cfg=cfg,
        faces=datamodule.faces,
    )
    # mesh_vqvae_module.strict_loading = False

    callbacks = []
    ckpt_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_last=True,
        every_n_train_steps=5_000,
    )
    callbacks.append(ckpt_callback)

    loggers = []
    loggers += [
        WandbLogger(
            mode="online" if cfg.wandb else "disabled",
            name=cfg.name, 
            config=OmegaConf.to_object(cfg),
            project="Mesh-VQ-VAE",
            dir=cfg.exp_dir,
            settings=wandb.Settings(start_method="fork"),
        )
    ]

    print("Number of GPUs ", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    print("Training...")
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        **cfg.trainer,
    )

    if cfg.mode == "train":
        ckpt_path = os.path.join(save_dir, "last.ckpt") if cfg.resume else None
        trainer.fit(mesh_vqvae_module, datamodule=datamodule, ckpt_path=ckpt_path)
        trainer.test(mesh_vqvae_module, datamodule=datamodule)
    elif cfg.mode == "test":
        ckpt_path = os.path.join(save_dir, "last.ckpt") 
        trainer.test(mesh_vqvae_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
