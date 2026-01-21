"""
Code for training the Mesh-VQ-VAE.
"""

from models.mask_transformer import (
    create_datamodule,
    MaskTransformerModule,
)
import hydra
import os
import numpy as np
import torch

# import cv2
# cv2.setNumThreads(1)

import time
import datetime
from omegaconf import OmegaConf, DictConfig
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.register_new_resolver("eval", eval)

    if "seed" not in cfg:
        cfg.seed = int(time.time() * 1000) % 1000
    L.seed_everything(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.exp_dir = os.path.join("./exp/mask_transformer", cfg.name)
    os.makedirs(cfg.exp_dir, exist_ok=True)

    if cfg.benchmark is not None:
        if cfg.model.motion_ckpt is not None and cfg.model.motion_subtag is not None:
            cfg.model.motion_ckpt = f"{cfg.model.motion_ckpt}_{cfg.model.motion_subtag}"
        cfg.model.motion_ckpt = (
            os.path.join(cfg.exp_dir, cfg.model.motion_ckpt, "checkpoints", "last.ckpt")
            if cfg.model.motion_ckpt is not None and not cfg.resume and not cfg.resume_weights_only
            else None
        )

        if cfg.model.image_ckpt is not None and cfg.model.image_subtag is not None:
            cfg.model.image_ckpt = f"{cfg.model.image_ckpt}_{cfg.model.image_subtag}"
        cfg.model.image_ckpt = (
            os.path.join(cfg.exp_dir, cfg.model.image_ckpt, "checkpoints", "last.ckpt")
            if cfg.model.image_ckpt is not None and not cfg.resume and not cfg.resume_weights_only
            else None
        )
        if cfg.subtag is not None:
            cfg.benchmark = f"{cfg.benchmark}_{cfg.subtag}"
        cfg.exp_dir = os.path.join(cfg.exp_dir, cfg.benchmark)
        save_dir = os.path.join(cfg.exp_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        wandb_name = cfg.name + "-" + cfg.benchmark
    else:
        save_dir = os.path.join(cfg.exp_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        wandb_name = cfg.name

    print("Creating data module...")
    datamodule = create_datamodule(cfg.dataset, cfg.debug)

    print("Creating lightning module...")
    if cfg.mode == "test":
        cfg.model.motion_ckpt = cfg.model.image_ckpt = None
        ckpt_path = os.path.join(save_dir, "last.ckpt")

        mask_transformer_module = MaskTransformerModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            strict=True,
            cfg=cfg,
        )
    elif cfg.resume_weights_only:
        ckpt_path = cfg.get("resume_ckpt", None) or os.path.join(save_dir, "last.ckpt")

        mask_transformer_module = MaskTransformerModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            strict=False,
            cfg=cfg,
        )
    else:
        mask_transformer_module = MaskTransformerModule(
            cfg=cfg,
        )

    callbacks = []
    ckpt_callback = ModelCheckpoint(
        dirpath=save_dir,
        every_n_train_steps=1_000,
        save_last=True,
    )
    callbacks.append(ckpt_callback)

    loggers = []
    loggers += [
        WandbLogger(
            mode="online" if cfg.wandb else "disabled",
            name=wandb_name,
            config=OmegaConf.to_object(cfg),
            entity="motion-reconstruction", 
            project="MoRo",
            dir=cfg.exp_dir,
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
        ckpt_path = (
            cfg.get("resume_ckpt", None) or os.path.join(save_dir, "last.ckpt")
            if cfg.resume
            else None
        )
        trainer.fit(mask_transformer_module, datamodule=datamodule, ckpt_path=ckpt_path)
    elif cfg.mode == "val":
        ckpt_path = (
            cfg.get("resume_ckpt", None) or os.path.join(save_dir, "last.ckpt")
            if cfg.resume
            else None
        )
        trainer.validate(mask_transformer_module, datamodule=datamodule, ckpt_path=ckpt_path)
    elif cfg.mode == "test":
        trainer.test(mask_transformer_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
