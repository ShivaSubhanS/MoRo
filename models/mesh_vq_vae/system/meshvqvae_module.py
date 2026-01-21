"""
Code for training Mesh-VQ-VAE
"""

import torch
import torch.optim as optim

from ..model import MeshVQVAE
from ..utils.mesh_render import plot_meshes, get_colors_from_meshes
from ..utils.eval import *

import lightning as L
import wandb

from ..utils.body_model import BodyModel


class MeshVQVAEModule(L.LightningModule):
    def __init__(self, cfg, faces=None, datamodule=None):
        super().__init__()
        self.cfg = cfg
        self.f = faces
        self.datamodule = datamodule

        bm_path = cfg.dataset.bm_path
        self.batch_size = cfg.dataset.batch_size

        self.bm_neutral = BodyModel(
            bm_path,
            model_type=cfg.dataset.model_type,
            gender="neutral",
        )
        self.bm_neutral.eval()

        self.mesh_vqvae = MeshVQVAE(cfg.model.vqvae)

    def forward(self, batch):
        mesh = batch["mesh"]
        result = self.mesh_vqvae(mesh)
        commitment_loss = result["commitment_loss"]
        ortho_loss = result["ortho_loss"]
        entropy_loss = result["entropy_loss"]
        mesh_recon = result["x_recon"]
        perplexity = result["perplexity"]
        coverage = result["coverage"]
        out = {}

        if self.cfg.dataset.normalize:
            mesh_recon = mesh_recon - torch.mean(mesh_recon, dim=1, keepdim=True)
        out["mesh_recon"] = mesh_recon

        # compute loss
        if mesh.shape[1] != mesh_recon.shape[1]:
            mesh = mesh[:, :mesh_recon.shape[1], :]
        v2v_error = v2v(mesh, mesh_recon)
        j2j_error = v2v(self.bm_neutral.joint_regressor @ mesh, self.bm_neutral.joint_regressor @ mesh_recon) 
        loss_dict = {
            "commitment_loss": commitment_loss,
            "ortho_loss": ortho_loss,
            "entropy_loss": entropy_loss,
            "v2v_error": v2v_error * 1000,
            "j2j_error": j2j_error * 1000,
            "perplexity": perplexity,
            "coverage": coverage,
        }
        loss = (
            self.cfg.loss.commitment * commitment_loss
            + self.cfg.loss.ortho * ortho_loss
            + self.cfg.loss.entropy * entropy_loss
            + self.cfg.loss.v2v * v2v_error
            + self.cfg.loss.j2j * j2j_error
        )
        loss_dict["loss"] = loss

        return out, loss_dict

    def on_save_checkpoint(self, checkpoint):
        keys = list(checkpoint["state_dict"].keys())
        for key in keys:
            if key.startswith("bm"):
                del checkpoint["state_dict"][key]

    def on_train_start(self):
        for param in self.bm_neutral.parameters():
            param.requires_grad = False
        self.bm_neutral.eval()

    def training_step(self, batch, batch_idx):
        out, loss_dict = self(batch)

        loss = loss_dict["loss"]
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, on_step=True)

        return loss

    def plot_meshes(self, data, data_recon):
        recon_image = plot_meshes(
            data_recon[:4],
            self.f,
            self.device,
            show=False,
            save=None,
        )
        gt_image = plot_meshes(
            data[:4],
            self.f,
            self.device,
            show=False,
            save=None,
        )
        colored_mesh = get_colors_from_meshes(
            data,
            data_recon,
            0,
            3,
        )
        color_image = plot_meshes(
            data_recon[:4],
            self.f,
            self.device,
            show=False,
            save=None,
            colors=torch.from_numpy(colored_mesh)[:4].float(),
        )
        image_dict = {
            "recon": recon_image,
            "gt": gt_image,
            "recon_color": color_image,
        }
        return image_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out, loss_dict = self(batch)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, on_epoch=True)

        # log mesh reconstruction images
        if batch_idx == 0:
            mesh_gt = batch["mesh"]
            mesh_recon = out["mesh_recon"]
            image_dict = self.plot_meshes(mesh_gt, mesh_recon)
            self.logger.experiment.log(
                {
                    "val/samples": [
                        wandb.Image(v, caption=k) for k, v in image_dict.items()
                    ]
                },
                step=self.global_step,
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out, loss_dict = self(batch)
        mesh_gt = batch["mesh"]
        mesh_recon = out["mesh_recon"]
        pa_error = pa_v2v(mesh_gt, mesh_recon)
        loss_dict["pa_error"] = pa_error * 1000
        self.log_dict({f"test/{k}": v for k, v in loss_dict.items()}, on_epoch=True)

        # log mesh reconstruction images
        image_dict = self.plot_meshes(mesh_gt, mesh_recon)
        self.logger.experiment.log(
            {
                "test/samples": [
                    wandb.Image(v, caption=k) for k, v in image_dict.items()
                ]
            },
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.mesh_vqvae.parameters(),
            lr=self.cfg.optim.lr,
            betas=(0.9, 0.99),
            weight_decay=self.cfg.optim.weight_decay,
        )
        lr_scheduler = self.cfg.optim.lr_scheduler
        if lr_scheduler is None:
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.cfg.optim.lr_scheduler.milestones,
                gamma=self.cfg.optim.lr_scheduler.gamma,
            )
            return [optimizer], [scheduler]
