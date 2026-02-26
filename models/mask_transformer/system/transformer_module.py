import joblib
from omegaconf import OmegaConf
import time
import numpy as np
import torch
import torch.nn as nn
import lightning as L
import os
import wandb
from collections import defaultdict

from ..model import MaskTransformer, Backbone, TemporalSmoother
from models.mesh_vq_vae.model.mesh_vq_vae import MeshVQVAE
from ..utils import (
    parse_optimizer,
    parse_scheduler,
    update_cam,
    VIS_COLOR_DICT,
)
from .loss import HMRLoss
from .renderer import BatchRenderer
from ..utils.body_model import BodyModel
from ..model.tools import softmax

import open3d as o3d
import cv2
from PIL import Image


def trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class MaskTransformerModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.bm_neutral = BodyModel(
            bm_path=cfg.dataset.bm_path,
            model_type=cfg.dataset.model_type,
            gender="neutral",
        )
        self.bm_neutral.eval()

        self.vqvae = MeshVQVAE(cfg.model.vqvae)
        self.vqvae.eval()
        self.vqvae.requires_grad_(False)  # Freeze the VQ-VAE model
        self.load_vqvae(cfg.model.vqvae.model_path)

        # image conditioning
        self.task = cfg.model.task
        if self.task != "motion":
            self.backbone = Backbone(cfg.model.backbone)

        cfg.model.dst_former.dim_contact = 6 if cfg.dataset.contact_hand else 4
        self.mask_transformer = MaskTransformer(cfg.model)
        self.mask_transformer.load_and_freeze_token_emb(self.vqvae.get_codebook())

        if self.cfg.model.dst_former.get("freeze", False):
            self.mask_transformer.eval()
            self.mask_transformer.requires_grad_(False)

        self.use_smoother = cfg.model.get("use_smoother", False)
        if self.use_smoother:
            self.smoother = TemporalSmoother(self.cfg.model.smoother)
        else:
            self.smoother = nn.Identity()

        self.global_traj_helper = self.mask_transformer.dst_former.global_traj_helper

        # loss
        self.hmr_loss = HMRLoss(
            cfg, self.bm_neutral.joint_regressor, self.global_traj_helper
        )

    def load_vqvae(self, vqvae_path):
        ckpt = torch.load(vqvae_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        state_dict = {
            k.replace("mesh_vqvae.", "vqvae."): v for k, v in state_dict.items()
        }
        self.load_state_dict(state_dict=state_dict, strict=False)

    def parse_noise(self):
        if self.trainer.training or self.trainer.validating:
            noise_dict = self.cfg.dataset.noise.train
            for idx, step_ratio in enumerate(noise_dict.schedule):
                if self.global_step >= int(step_ratio * self.trainer.max_steps):
                    noise_transl = noise_dict.transl[idx]
                    noise_rot = noise_dict.rot[idx]
        else:
            noise_dict = self.cfg.dataset.noise.test
            noise_transl = noise_dict.transl
            noise_rot = noise_dict.rot
        return noise_transl, noise_rot

    def prepare_batch(self, batch):
        if "mesh" in batch:
            # get tokens
            meshes = batch["mesh"]
            B, F, V, C = meshes.shape
            meshes = meshes.view(B * F, V, C)
            ids = self.vqvae.get_codebook_indices(meshes)
            ids = ids.view(B, F, -1)
            batch["ids"] = ids
        else:
            # when we do not have gt mesh, we assume images are given and set ids as full mask
            B, F = batch["crop_imgs"].shape[:2]
            J = self.cfg.model.vqvae.num_tokens
            batch["ids"] = torch.full((B, F, J), self.mask_transformer.mask_id, dtype=torch.long, device=self.device)


        if self.task == "motion":
            from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
            # compute noisy canonical trajectory
            noise_transl, noise_rot = self.parse_noise()

            rotation = batch["rotation"]
            translation = batch["translation"]

            noise_transl = torch.randn_like(translation) * noise_transl
            translation_noisy = translation + noise_transl

            # matrix_to_euler_angles return nan if the rotation matrix is not in [-1, 1]
            rotation = torch.clamp(rotation, -1., 1.)
            euler_rot = matrix_to_euler_angles(rotation, "ZXY")
            noise_rot = torch.randn_like(euler_rot) * noise_rot / 180 * np.pi
            euler_rot_noisy = euler_rot + noise_rot
            rotation_noisy = euler_angles_to_matrix(euler_rot_noisy, "ZXY")
            _, cano_traj_noisy = self.global_traj_helper.get_cano_traj_repr(
                rotation_noisy, translation_noisy
            )
            batch["rotation_noisy"] = rotation_noisy
            batch["translation_noisy"] = translation_noisy
            batch["cano_traj_noisy"] = cano_traj_noisy

        # get image feature
        cond = None
        if self.task != "motion":
            cond = self.backbone(batch)
        batch["cond"] = cond

    def forward(self, batch, stage="train", mode=None):
        out = self.mask_transformer(
            batch, stage=stage, mode=mode, step=self.global_step
        )
        return out

    def process_output(self, out, batch):
        traj_dict = self.parse_traj(out, batch)
        local_mesh_dict = self.get_meshes(out, batch)
        global_mesh_dict = self.get_global_meshes(local_mesh_dict, traj_dict)

        return traj_dict, global_mesh_dict

    def parse_traj(self, out, batch):
        traj_dict = {}
        if "rotation" in batch and "translation" in batch:
            traj_dict["gt"] = (batch["rotation"], batch["translation"])

        if self.task == "motion":
            R0_noisy = batch["rotation_noisy"][:, :1]
            T0_noisy = batch["translation_noisy"][:, :1]
            rotation_noisy, translation_noisy = (
                self.global_traj_helper.parse_cano_traj_repr(
                    out["cano_traj_noisy"], R0_noisy, T0_noisy
                )
            )
            traj_dict["cano_noisy"] = (rotation_noisy, translation_noisy)

        if self.task in ["image", "video"]:
            traj_dict["cam_coarse"] = out["cam_traj_coarse"]
            traj_dict["cam"] = out["cam_traj"]

        if self.task in ["motion", "video"]:
            if "rotation" in batch and "translation" in batch: 
                R0 = batch["rotation"][:, :1]
                T0 = batch["translation"][:, :1]
            else:
                R0 = torch.eye(3, device=self.device).view(1, 1, 3, 3).repeat(
                    out["cano_traj"].shape[0], 1, 1, 1
                )
                T0 = torch.zeros(1, 1, 1, 3, device=self.device).repeat(
                    out["cano_traj"].shape[0], 1, 1, 1
                )

            rotation_rel, translation_rel = (
                self.global_traj_helper.parse_cano_traj_repr(
                    out["cano_traj"], R0, T0
                )
            )

            traj_dict["cano_rel"] = (rotation_rel, translation_rel)

        if "guided_traj" in out:
            traj_dict["guided"] = out["guided_traj"]

        return traj_dict

    def get_mesh_from_tokens(self, tokens, smooth=False):
        B, F, J, C = tokens.shape

        if smooth:
            tokens = self.smoother(tokens)

        tokens = tokens.view(B * F, J, C)
        mesh = self.vqvae.decode(tokens, quantize=False)
        if self.cfg.dataset.normalize:
            mesh = mesh - mesh.mean(dim=1, keepdim=True)
        mesh = mesh.view(B, F, -1, 3)
        return mesh
        
    def get_mesh_from_ids(self, ids, smooth=False):
        B, F, J = ids.shape
        ids = ids.view(B * F, J)
        latent_code = self.mask_transformer.token_emb(ids)

        mesh = self.get_mesh_from_tokens(latent_code.view(B, F, J, -1), smooth=smooth)
        return mesh

    def diff_mesh(self, logits):
        if self.trainer.training:
            temp_start = self.cfg.model.temp_start
            temp_end = self.cfg.model.temp_end
            temp_end_ratio = self.cfg.model.temp_end_ratio
            ratio = min(
                self.global_step / (self.trainer.max_steps * temp_end_ratio), 1.0
            )
            temp = temp_end + 0.5 * (temp_start - temp_end) * (
                1 + np.cos(ratio * np.pi)
            )
        else:
            temp = self.cfg.model.temp_test

        hard = self.cfg.model.straight_through
        if self.cfg.model.gumbel:
            probs = nn.functional.gumbel_softmax(logits, tau=temp, hard=hard, dim=1)
        else:
            probs = softmax(logits, tau=temp, hard=hard, dim=1)

        weights = self.vqvae.get_codebook()
        tokens = torch.einsum("b n f j, n c -> b f j c", probs, weights)

        mesh = self.get_mesh_from_tokens(tokens, smooth=True)
        return mesh

    def get_meshes(self, out, batch):
        mesh_dict = {}
        if "mesh" in batch:
            gt_mesh = batch["mesh"]
            mesh_dict["gt"] = gt_mesh
            mesh_dict["gt_joints"] = batch["local_joints"]

        if "ids" in batch:
            ids = batch["ids"]
            input_mesh = self.get_mesh_from_ids(ids)
            mesh_dict["input"] = input_mesh

        if "masked_ids" in out:
            masked_ids = out["masked_ids"]
            masked_mesh = self.get_mesh_from_ids(masked_ids)
            mesh_dict["masked"] = masked_mesh

        if "pred_ids" in out:
            pred_ids = out["pred_ids"]
            pred_mesh = self.get_mesh_from_ids(pred_ids, smooth=True)
            mesh_dict["pred"] = pred_mesh

        # differentiable mesh tokens
        if "logits" in out:
            logits = out["logits"]  # [B, N_code, F, J]
            mesh_dict["pred_diff"] = self.diff_mesh(logits)

        return mesh_dict

    def get_global_meshes(self, local_mesh_dict, traj_dict, offset=None):

        def get_global_mesh(mesh_key, traj_key, gt_transl=False):
            if mesh_key not in local_mesh_dict or traj_key not in traj_dict:
                return None
            local_mesh = local_mesh_dict[mesh_key]
            rotation, translation = traj_dict[traj_key]
            if gt_transl:
                translation = traj_dict["gt"][1]
            local_mesh = local_mesh + offset if offset is not None else local_mesh
            global_mesh = local_mesh @ rotation.transpose(-1, -2) + translation
            return global_mesh

        global_mesh_dict = {}
        global_mesh_dict["gt"] = get_global_mesh("gt", "gt")
        global_mesh_dict["gt_joints"] = get_global_mesh("gt_joints", "gt")
        global_mesh_dict["input"] = get_global_mesh("input", "gt")
        global_mesh_dict["pred_diff_local"] = get_global_mesh("pred_diff", "gt")
        global_mesh_dict["pred_local"] = get_global_mesh("pred", "gt")

        if self.task == "motion":
            global_mesh_dict["input_noisy"] = get_global_mesh("input", "cano_noisy")
            global_mesh_dict["masked"] = get_global_mesh("masked", "cano_noisy")

            global_mesh_dict["pred_diff_rel"] = get_global_mesh("pred_diff", "cano_rel")

            global_mesh_dict["pred_rel"] = get_global_mesh("pred", "cano_rel")
        elif self.task == "image":
            gt_transl = not self.cfg.model.get("use_transl", True)
            global_mesh_dict["pred_diff_coarse"] = get_global_mesh(
                "pred_diff", "cam_coarse", gt_transl=gt_transl
            )
            global_mesh_dict["pred_diff"] = get_global_mesh(
                "pred_diff", "cam", gt_transl=gt_transl
            )

            global_mesh_dict["pred_coarse"] = get_global_mesh(
                "pred", "cam_coarse", gt_transl=gt_transl
            )
            global_mesh_dict["pred"] = get_global_mesh(
                "pred", "cam", gt_transl=gt_transl
            )
        elif self.task == "video":
            global_mesh_dict["pred_diff_rel"] = get_global_mesh("pred_diff", "cano_rel")
            global_mesh_dict["pred_diff_coarse"] = get_global_mesh(
                "pred_diff", "cam_coarse"
            )
            global_mesh_dict["pred_diff"] = get_global_mesh("pred_diff", "cam")

            global_mesh_dict["pred_rel"] = get_global_mesh("pred", "cano_rel")
            global_mesh_dict["pred_coarse"] = get_global_mesh("pred", "cam_coarse")
            global_mesh_dict["pred"] = get_global_mesh("pred", "cam")

        return global_mesh_dict

    def on_train_epoch_start(self):
        from ..data.mixed_dataset import MixedDataset
        if isinstance(self.trainer.train_dataloader.dataset, MixedDataset):
            self.trainer.train_dataloader.dataset.on_epoch_start()

    def training_step(self, batch, batch_idx):
        self.prepare_batch(batch)
        out = self(batch, stage="train")
        traj_dict, mesh_dict = self.process_output(out, batch)
        loss_dict = self.hmr_loss(batch, out, traj_dict, mesh_dict)
        metrics_dict = self.hmr_loss.evaluate(batch, out, mesh_dict)

        log_dict = dict(**loss_dict, **metrics_dict)
        
        loss = loss_dict["loss"]

        self.log_dict({f"train/{k}": v for k, v in log_dict.items()}, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.prepare_batch(batch)
        mode_list = self.cfg.model.mode_list

        for mode in mode_list:
            out = self(batch, stage="val", mode=mode)
            traj_dict, mesh_dict = self.process_output(out, batch)
            loss_dict = self.hmr_loss(batch, out, traj_dict, mesh_dict)
            metrics_dict = self.hmr_loss.evaluate(batch, out, mesh_dict)

            val_dict = metrics_dict.copy()
            val_dict.update(
                {
                    "loss_ce": loss_dict["loss_ce"],
                }
            )

            self.log_dict(
                {f"val/{mode}_{k}": v for k, v in val_dict.items()},
                on_epoch=True,
                sync_dist=True,
            )

            total_batches = self.trainer.num_val_batches[dataloader_idx]
            num_logs = 4
            log_img_interval = max(total_batches // num_logs, 1)

            if self.task != "motion" and batch_idx % log_img_interval == 0:
                # for idx in range(batch["K"].shape[0]):
                for idx in range(1):
                    gt_img = self.render_img(
                        batch["img_paths"][0][idx],
                        mesh_dict["gt"][idx, 0],
                        batch["K"][idx, 0],
                        dist_coeffs=batch["dist_coeffs"][idx] if "dist_coeffs" in batch else None,
                    )
                    pred_img = self.render_img(
                        batch["img_paths"][0][idx],
                        mesh_dict["pred"][idx, 0],
                        batch["K"][idx, 0],
                        dist_coeffs=batch["dist_coeffs"][idx] if "dist_coeffs" in batch else None,
                    )
                    self.logger.experiment.log(
                        {
                            "gt_img": wandb.Image(gt_img, caption=f"GT_{mode}_{batch_idx}"),
                            "pred_img": wandb.Image(pred_img, caption=f"Pred_{mode}_{batch_idx}"),
                        }
                    )

    def on_test_start(self):
        self.save_dir = None 

        ################# VISUALIZATION #################
        if self.cfg.model.test.visualize:
            assert self.cfg.dataset.name == "amass"
            print("Visualizing...")
            print("from left to right - input, masked, pred, gt")
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
            )
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.add_geometry(mesh_frame)

            self.cam_trans = np.array(
                [[0, 0, -1, 5], [-1, 0, 0, 2], [0, -1, 0, 2], [0, 0, 0, 1]]
            )

            self.vis_dir = None
            if self.cfg.model.test.save_fig:
                vis_dir = f"vis_{self.cfg.model.mode}_{self.cfg.model.mask_ratio}_{self.cfg.model.test.generate.timesteps}_{self.cfg.dataset.noise.test.transl}_{self.cfg.dataset.noise.test.rot}"
                self.vis_dir = os.path.join(self.cfg.exp_dir, vis_dir)
                os.makedirs(self.vis_dir, exist_ok=True)

        ################# RENDERING #################
        if self.cfg.model.test.render:
            if self.cfg.model.test.save_fig:
                render_dir = (
                    f"render_"
                    f"{self.cfg.dataset.name}_"
                    f"{self.cfg.model.mode}_"
                    f"{self.cfg.model.test.generate.timesteps}_"
                    f"{self.cfg.model.test.generate.topk}"
                )
                self.render_dir = os.path.join(self.cfg.exp_dir, render_dir)
                os.makedirs(self.render_dir, exist_ok=True)

        ################# SAVE RESULT #################
        if self.cfg.model.test.save_result and self.task != "motion":
            result_dir = (
                f"result_{self.cfg.dataset.name}/"
                f"{self.cfg.model.mode}_"
                f"{self.cfg.model.test.generate.timesteps}_"
                f"{self.cfg.model.test.generate.topk}"
            )

            if self.cfg.model.test.generate.topk > 1:
                timestamp = int(time.time())
                result_dir += f"_{timestamp}"

            self.result_dir = os.path.join(self.cfg.exp_dir, result_dir)
            os.makedirs(self.result_dir, exist_ok=True)
            self.save_result_dict = defaultdict(list)

            # Save configuration to result directory
            config_save_path = os.path.join(self.result_dir, "config.yaml")
            OmegaConf.save(self.cfg, config_save_path)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Track sequence changes and save results when sequence ends
        if self.cfg.model.test.save_result and self.task != "motion":
            current_seq_name = batch["seq_name"][0]
            
            # Check if this is a new sequence or if we need to save previous results
            if not hasattr(self, '_current_seq_name'):
                self._current_seq_name = current_seq_name
            elif self._current_seq_name != current_seq_name:
                # Save results for the previous sequence
                self.save_result(self._current_seq_name)
                self._current_seq_name = current_seq_name

        self.prepare_batch(batch)

        out = self(batch, stage="test")

        traj_dict, mesh_dict = self.process_output(out, batch)

        if self.cfg.model.test.save_result:
            metrics_dict = self.hmr_loss.evaluate(batch, out, mesh_dict)
        else:
            metrics_dict = {}

        self.log_dict(
            {f"test/{k}": v for k, v in metrics_dict.items()},
            on_epoch=True,
            sync_dist=True,
        )

        if self.cfg.model.test.visualize:
            if self.task == "motion":
                visualize_keys = ["input", "masked", "pred_rel", "gt"]
            else:
                raise NotImplementedError("Visualization only implemented for motion task.")

            self.visualize(mesh_dict, batch, out, visualize_keys, batch_idx)
            
        if self.cfg.model.test.save_result and self.task != "motion":
            # save the predicted smplx params
            self.add_result(batch, mesh_dict, out)

    def on_test_epoch_end(self):
        if (
            self.task == "motion"
            and self.cfg.model.test.visualize
            and self.cfg.model.test.save_result
        ):
            # save metrics for test epoch
            metrics_dict = self.trainer.callback_metrics
            metrics_dict = {k: v.item() for k, v in metrics_dict.items()}

            save_path = f"metric.npz"
            if self.vis_dir is not None:
                save_path = os.path.join(self.vis_dir, save_path)
                np.savez(save_path, **metrics_dict)

        if self.cfg.model.test.save_result and self.task != "motion":
            # Save results for the last sequence
            if hasattr(self, '_current_seq_name'):
                self.save_result(self._current_seq_name)

    def visualize(self, mesh_dict, batch=None, out=None, visualize_keys=tuple(), batch_idx=0):
        # only visualize first batch
        b = 0
        
        # Get foot contact labels if available
        gt_foot_contact = None
        if batch is not None and "contact_label" in batch:
            gt_foot_contact = batch["contact_label"][b,:,:4].cpu().numpy()  # [F-1, 4]

        pred_foot_contact = None
        if out is not None and "contact" in out:
            pred_foot_contact = out["contact"][b,:,:4].cpu().numpy() > 0  # [F-1, 4]
        
        
        for f in range(self.cfg.dataset.clip_len):
            mesh_o3d_list = []
            for idx, k in enumerate(visualize_keys):
                mesh = mesh_dict[k]
                vertices = mesh[b, f].detach().cpu().numpy()
                # body mesh
                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)

                mesh_o3d.triangles = o3d.utility.Vector3iVector(self.bm_neutral.faces.cpu().numpy())
                mesh_o3d.compute_vertex_normals()
                mesh_o3d.paint_uniform_color(VIS_COLOR_DICT[k])
                transformation = np.identity(4)
                transformation[1, 3] = idx - (len(visualize_keys) - 1) / 2
                mesh_o3d.transform(transformation)

                mesh_o3d_list.append(mesh_o3d)
                self.vis.add_geometry(mesh_o3d)
                
                # Add foot contact visualization only for ground truth mesh
                # foot_contact has shape [F-1, 4] where last frame (f=F-1) has no contact label
                if k == "gt" and gt_foot_contact is not None and f < gt_foot_contact.shape[0]:  
                    foot_contact_frame = gt_foot_contact[f]  # [4] - [left_ankle, left_toe, right_ankle, right_toe]
                    
                    # Get joint positions for the ground truth mesh
                    gt_joints = self.bm_neutral.joint_regressor @ mesh_dict["gt"][b, f]
                    gt_joints = gt_joints.detach().cpu().numpy()
                    
                    # Foot joint indices: [7, 10, 8, 11] corresponds to [left_ankle, left_toe, right_ankle, right_toe]
                    foot_joint_indices = [7, 10, 8, 11]
                    
                    for foot_idx, (joint_idx, is_contact) in enumerate(zip(foot_joint_indices, foot_contact_frame)):
                        # Create a small sphere at the foot location for all joints
                        foot_pos = gt_joints[joint_idx]
                        contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                        contact_sphere.translate(foot_pos)
                        
                        # Apply the same transformation as the GT mesh
                        contact_sphere.transform(transformation)
                        
                        # Color coding: green if in contact, red if not in contact
                        if is_contact > 0.5:  # Foot is in contact with ground
                            contact_sphere.paint_uniform_color([0, 1, 0])  # Green
                        else:  # Foot is not in contact with ground
                            contact_sphere.paint_uniform_color([1, 0, 0])  # Red
                        
                        mesh_o3d_list.append(contact_sphere)
                        self.vis.add_geometry(contact_sphere)
                        
                # Add predicted foot contact visualization on predicted relative mesh
                if k == "pred_rel" and pred_foot_contact is not None and f < pred_foot_contact.shape[0]:
                    pred_foot_contact_frame = pred_foot_contact[f]  # [4]
                    pred_joints = self.bm_neutral.joint_regressor @ mesh_dict["pred_rel"][b, f]
                    pred_joints = pred_joints.detach().cpu().numpy()

                    foot_joint_indices = [7, 10, 8, 11]  # [left_ankle, left_toe, right_ankle, right_toe]
                    for joint_idx, is_contact in zip(foot_joint_indices, pred_foot_contact_frame):
                        foot_pos = pred_joints[joint_idx]
                        contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                        contact_sphere.translate(foot_pos)
                        contact_sphere.transform(transformation)

                        # Green if predicted contact, red otherwise (same scheme as GT)
                        if is_contact:
                            contact_sphere.paint_uniform_color([0, 1, 0])
                        else:
                            contact_sphere.paint_uniform_color([1, 0, 0])

                        mesh_o3d_list.append(contact_sphere)
                        self.vis.add_geometry(contact_sphere)
            
            ctr = self.vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param = update_cam(cam_param, self.cam_trans)
            ctr.convert_from_pinhole_camera_parameters(cam_param)
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.cfg.model.test.save_fig:
                save_path = os.path.join(
                    self.vis_dir,
                    f"clip_{batch_idx:03d}",
                    f"vis_{f:02d}.png",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.vis.capture_screen_image(save_path)
            for mesh_o3d in mesh_o3d_list:
                self.vis.remove_geometry(mesh_o3d)

    def render_img(self, img_path, mesh, K, dist_coeffs=None):
        bg_img = cv2.imread(img_path)
        # only for BEDLAM closeup, rotate by 90 degree
        if "closeup" in img_path:
            bg_img = cv2.rotate(bg_img, cv2.ROTATE_90_CLOCKWISE)
        if dist_coeffs is not None:
            bg_img = cv2.undistort(
                bg_img,
                K.cpu().numpy(),
                dist_coeffs.cpu().numpy(),
            )
        if self.cfg.dataset.name == "prox":
            bg_img = cv2.flip(bg_img, 1)

        h, w, c = bg_img.shape

        renderer = BatchRenderer(K=K.unsqueeze(0), img_w=w, img_h=h, faces=self.bm_neutral.faces).to(self.device)
        joints = self.bm_neutral.joint_regressor @ mesh
        fg_img = renderer(
            mesh.unsqueeze(0).detach(),
            joints.unsqueeze(0).detach(),
        )[0]

        bg_img = Image.fromarray(bg_img[..., ::-1])
        fg_img = Image.fromarray(fg_img)

        render_img = bg_img.copy()
        # Separate RGB and alpha channels for proper compositing
        fg_img_rgb = fg_img.convert("RGB")
        fg_img_alpha = fg_img.split()[-1]  # Extract alpha channel
        render_img.paste(fg_img_rgb, (0, 0), mask=fg_img_alpha)
        render_img = np.array(render_img)

        return render_img 

    def render(self, batch, mesh_dict, render_keys):
        K = batch["K"].clone()
        img_paths = batch["img_paths"]

        B, F = K.shape[:2]
        total_frames = B * F
        K = K.view(total_frames, 3, 3)

        # we assume the images size are the same across batches
        img = cv2.imread(img_paths[0][0])
        # only for BEDLAM closeup, rotate by 90 degree
        if "closeup" in img_paths[0][0]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w, c = img.shape
        downsample = self.cfg.model.test.get("downsample", 1)
        if downsample != 1:
            h, w = round(h / downsample), round(w / downsample)
            K = K / downsample

        # Get bounding box drawing flag
        draw_bbox = self.cfg.model.test.get("draw_bbox", False)
        if draw_bbox:
            centers = batch["center"]  # [B, F, 2]
            scales = batch["scale"]   # [B, F]
            # Apply downsample to bbox coordinates
            if downsample != 1:
                centers = centers / downsample
                scales = scales / downsample

        render_batch = 150
        
        # Create K matrix for maximum batch size once
        K_full = K[:1].expand(render_batch, -1, -1)
        
        # Initialize renderer once with full batch size
        renderer = BatchRenderer(K=K_full, img_w=w, img_h=h, faces=self.bm_neutral.faces).to(self.device)
        
        for key in render_keys:
            verts = mesh_dict[key]
            verts = verts.view(total_frames, -1, 3)
            joints = self.bm_neutral.joint_regressor @ verts

            # Process in batches - BatchRenderer handles variable sizes internally
            all_fg_imgs = []
            for i in range(0, total_frames, render_batch):
                end_idx = min(total_frames, i + render_batch)
                
                verts_batch = verts[i:end_idx]
                joints_batch = joints[i:end_idx]
                
                # BatchRenderer handles variable batch sizes automatically
                fg_imgs_batch = renderer(verts_batch, joints_batch)
                all_fg_imgs.append(fg_imgs_batch)
            
            # Concatenate all batches
            fg_imgs = np.concatenate(all_fg_imgs, axis=0)
            fg_imgs = fg_imgs.reshape(B, F, h, w, 4)

            # paste the fg_img to the original image
            for b in range(B):
                for f in range(F):
                    img_path = img_paths[f][b]
                    bg_img = cv2.imread(img_path)
                    # only for BEDLAM closeup, rotate by 90 degree
                    if "closeup" in img_path:
                        bg_img = cv2.rotate(bg_img, cv2.ROTATE_90_CLOCKWISE)

                    if "dist_coeffs" in batch and batch["dist_coeffs"] is not None:
                        bg_img = cv2.undistort(
                            bg_img,
                            batch["K"][b, f].cpu().numpy(),
                            batch["dist_coeffs"][b].cpu().numpy(),
                        )
                    if self.cfg.dataset.name == "prox":
                        bg_img = cv2.flip(bg_img, 1)  # flip horizontally 

                    if downsample != 1:
                        bg_img = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_LINEAR)
                    fg_img = fg_imgs[b, f]

                    # Draw bounding box if requested
                    if draw_bbox:
                        center = centers[b, f].cpu().numpy()  # [2]
                        scale = scales[b, f].cpu().numpy()    # scalar
                        
                        # Convert center and scale to bounding box coordinates
                        # Scale represents the size of the crop in normalized coordinates
                        # Multiply by 200 to get actual pixel size (based on crop function)
                        bbox_size = scale * 200
                        x1 = int(center[0] - bbox_size / 2)
                        y1 = int(center[1] - bbox_size / 2)
                        x2 = int(center[0] + bbox_size / 2)
                        y2 = int(center[1] + bbox_size / 2)
                        
                        # Clamp to image boundaries
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w - 1))
                        y2 = max(0, min(y2, h - 1))
                        
                        # Draw bounding box rectangle in green
                        cv2.rectangle(bg_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    bg_img = Image.fromarray(bg_img[..., ::-1])
                    fg_img = Image.fromarray(fg_img)

                    render_img = bg_img.copy()
                    # Separate RGB and alpha channels for proper compositing
                    fg_img_rgb = fg_img.convert("RGB")
                    fg_img_alpha = fg_img.split()[-1]  # Extract alpha channel
                    render_img.paste(fg_img_rgb, (0, 0), mask=fg_img_alpha)
                    render_img = np.array(render_img)[..., ::-1]

                    if self.cfg.model.test.save_fig:
                        save_path = os.path.join(
                            self.render_dir, batch["seq_name"][b], img_path.split("/")[-1]
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(
                            save_path,
                            render_img,
                        )

    def add_result(self, batch, mesh_dict, out):
        # save the predicted smplx params
        img_paths = batch["img_paths"]
        img_paths = [[os.path.basename(path) for path in x] for x in img_paths]
        img_paths = np.array(img_paths)

        pred_mesh = mesh_dict["pred"]
        contact_lbl = (out["contact"] > 0).float()

        window_size = self.cfg.dataset.overlap_len // 2
        if len(self.save_result_dict["frame_name"]) == 0 or window_size == 0:
            save_img_paths = img_paths[:, 0]
            save_pred_mesh = pred_mesh[0].detach().cpu().numpy()
            save_pred_contact = contact_lbl[0].detach().cpu().numpy()
        else:
            # remove last frames of the previous sequence
            self.save_result_dict["frame_name"][-1] = self.save_result_dict["frame_name"][-1][:-window_size]
            self.save_result_dict["verts"][-1] = self.save_result_dict["verts"][-1][:-window_size]
            self.save_result_dict["contact"][-1] = self.save_result_dict["contact"][-1][:-window_size]
            save_img_paths = img_paths[window_size:, 0]
            save_pred_mesh = pred_mesh[0, window_size:].detach().cpu().numpy()
            save_pred_contact = contact_lbl[0, window_size:].detach().cpu().numpy()

        self.save_result_dict["frame_name"].append(save_img_paths)
        self.save_result_dict["verts"].append(save_pred_mesh)
        self.save_result_dict["contact"].append(save_pred_contact)

    def save_result(self, seq_name=None):
        if self.cfg.model.test.save_result and self.task != "motion":
            # save the predicted smplx params

            for key in self.save_result_dict.keys():
                self.save_result_dict[key] = np.concatenate(
                    self.save_result_dict[key], axis=0
                )[None]

            save_dir = os.path.join(self.result_dir, seq_name)
            os.makedirs(save_dir, exist_ok=True)
            try:
                joblib.dump(self.save_result_dict, os.path.join(save_dir, "results.pkl"), compress=3)
            except:
                assert os.path.exists(os.path.join(save_dir, "results.pkl"))
                # breakpoint()
            print(f"Saved results to {save_dir}")

            self.save_result_dict = defaultdict(list)

    def on_predict_start(self):
        assert self.task == "video"
        self.seq_name = None

        ################# SAVE RESULT #################
        demo_name = self.cfg.demo.get("name", None)
        name = self.cfg.dataset.name
        if demo_name is not None:
            name = name + "_" + demo_name
        result_dir = (
            f"result_{name}/"
            f"{self.cfg.model.mode}_"
            f"{self.cfg.model.test.generate.timesteps}_"
            f"{self.cfg.model.test.generate.topk}"
        )

        self.result_dir = os.path.join(self.cfg.exp_dir, result_dir)
        os.makedirs(self.result_dir, exist_ok=True)
        self.save_result_dict = defaultdict(list)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        seq_name = batch["seq_name"][0]
        if self.seq_name is None:
            self.seq_name = seq_name
        elif self.seq_name != seq_name:
            # save results for the previous sequence
            self.save_result(self.seq_name)
            self.seq_name = seq_name

        self.prepare_batch(batch)

        out = self(batch, stage="test")

        _, mesh_dict = self.process_output(out, batch)

        self.add_result(batch, mesh_dict, out)

    def on_predict_epoch_end(self):
        # Save results for the last sequence
        self.save_result(self.seq_name)

    def configure_optimizers(self):
        optim = parse_optimizer(self.cfg.optim, self)
        ret = {
            "optimizer": optim,
        }
        if "scheduler" in self.cfg.optim:
            ret.update(
                {"lr_scheduler": parse_scheduler(self.cfg.optim.scheduler, optim)}
            )
        return ret
