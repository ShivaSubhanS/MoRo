import numpy as np
import torch
import torch.nn as nn
import einops
import os
import cv2

from ..utils import batch_compute_similarity_transform_torch

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


class HMRLoss(nn.Module):
    def __init__(self, cfg, joint_regressor, global_traj_helper):
        super(HMRLoss, self).__init__()
        self.cfg = cfg
        self.task = cfg.model.task

        self.register_buffer("joint_regressor", joint_regressor, persistent=False)

        joint_regressor_h36m = torch.from_numpy(
            np.load("body_models/J_regressor_h36m.npy")
        ).float()
        smplx2smpl = torch.load("body_models/smplx2smpl_sparse.pt", weights_only=True).to_dense().float() 
        joint_regressor_h36m = joint_regressor_h36m @ smplx2smpl 
        # this ensures that the 17 joints are not dependent on the eyeball meshes of smplx
        joint_regressor_h36m = joint_regressor_h36m[:, :9383]
        self.register_buffer(
            "joint_regressor_h36m", joint_regressor_h36m, persistent=False
        )

        self.global_traj_helper = global_traj_helper

        # read the loss weights from the config
        # local pose
        self.lambda_ce = self.cfg.model.loss.get("lambda_ce", 1.0)

        # canonical trajectory
        self.lambda_cano_traj = self.cfg.model.loss.get("lambda_cano_traj", 0.0)

        # camera space trajectory
        self.lambda_rot = self.cfg.model.loss.get("lambda_rot", 1.0)
        self.lambda_transl = self.cfg.model.loss.get("lambda_transl", 1.0)

        traj_loss_type = self.cfg.model.loss.get("traj_loss_type", "l2")
        self.traj_loss_fn = self.get_loss_fn(traj_loss_type)

        # local mesh
        self.lambda_v3d_local = self.cfg.model.loss.get("lambda_v3d_local", 0.0)

        # global mesh
        self.lambda_j3d_pos = self.cfg.model.loss.get("lambda_j3d_pos", 0.0)
        self.lambda_j3d_vel = self.cfg.model.loss.get("lambda_j3d_vel", 0.0)
        self.lambda_j2d = self.cfg.model.loss.get("lambda_j2d", 0.0)

        # regularization
        self.lambda_accel = self.cfg.model.loss.get("lambda_accel", 0.0)

        # stationary loss
        self.lambda_skating = self.cfg.model.loss.get("lambda_skating", 0.0)
        if self.cfg.dataset.contact_hand:
            self.stationary_joint_index_list = [7, 10, 8, 11, 20, 21] # [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
        else:
            self.stationary_joint_index_list = [7, 10, 8, 11] # [L_ankle, L_foot, R_ankle, R_foot]
        self.skating_vel_thres = 0.15

        # contact
        self.lambda_contact = self.cfg.model.loss.get("lambda_contact", 0.0)

        self.fps = 30

        loss_type = self.cfg.model.loss.get("loss_type", "l2")
        self.loss_fn = self.get_loss_fn(loss_type)

    @staticmethod
    def get_loss_fn(loss_type):
        if loss_type == "l1":
            return torch.nn.L1Loss(reduction="mean")
        elif loss_type == "l2":
            return torch.nn.MSELoss(reduction="mean")
        elif loss_type == "rmse":
            return lambda x, y: torch.norm(x - y, p=2, dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def project(self, joints_cam, batch):
        Ks = batch["K"]
        joints_2d = joints_cam @ Ks.transpose(-1, -2)
        joints_2d = joints_2d[..., :2] / (joints_2d[..., 2:] + 1e-6)
        return joints_2d

    def normalize_2d(self, joints_2d, batch):
        center = batch["center"]
        center = einops.rearrange(center, "b f c -> b f 1 c")
        scale = batch["scale"]
        scale = einops.rearrange(scale, "b f -> b f 1 1")
        joints_2d = (joints_2d - center) / (scale * 100)
        return joints_2d

    def project_weak_perspective(self, joints_3d, cam_traj, no_exp=False):
        cam_t = cam_traj[..., 6:9]
        if no_exp:
            s = cam_t[..., :1].unsqueeze(2)
        else:
            s = torch.exp(cam_t[..., :1]).unsqueeze(2)
        t = cam_t[..., 1:3].unsqueeze(2)

        joints_3d[..., :2] = joints_3d[..., :2] + t
        joints_2d = joints_3d[..., :2] * s  # normalized to [-1, 1]
        return joints_2d

    def forward(self, batch, out, traj_dict, mesh_dict):
        return self.compute_loss(batch, out, traj_dict, mesh_dict)

    def compute_loss(self, batch, out, traj_dict, mesh_dict):
        """
        compute the training loss
        """
        loss_dict = {}
        loss = 0.0

        # local pose loss
        loss_ce = out["ce_loss"]
        loss_dict["loss_ce"] = loss_ce
        loss += self.lambda_ce * loss_ce

        # canonical trajectory loss
        # if self.task in ["motion", "video"]:
        if self.task in ["motion"]:
            gt_cano_traj = batch["cano_traj_clean"]
            pred_cano_traj = out["cano_traj"]
            loss_cano_traj = (
                gt_cano_traj - pred_cano_traj
            ) * self.global_traj_helper.get_valid_mask(gt_cano_traj)
            loss_cano_traj = self.traj_loss_fn(
                loss_cano_traj, torch.zeros_like(loss_cano_traj)
            )
            loss_dict["loss_cano_traj"] = loss_cano_traj
            loss += self.lambda_cano_traj * loss_cano_traj

        # camera space trajectory loss
        if self.task in ["image", "video"]:
            rot_cam_gt, transl_cam_gt = traj_dict["gt"]
            rot_cam_coarse, transl_cam_coarse = traj_dict["cam_coarse"]
            rot_cam, transl_cam = traj_dict["cam"]

            loss_rot_coarse = self.traj_loss_fn(rot_cam_coarse, rot_cam_gt)
            loss_rot = self.traj_loss_fn(rot_cam, rot_cam_gt)
            loss_dict["loss_rot_coarse"] = loss_rot_coarse
            loss_dict["loss_rot"] = loss_rot
            loss += self.lambda_rot * loss_rot_coarse
            loss += self.lambda_rot * loss_rot

            use_transl = self.cfg.model.get("use_transl", True)
            if use_transl:
                loss_transl_coarse = self.traj_loss_fn(transl_cam_coarse, transl_cam_gt)
                loss_transl = self.traj_loss_fn(transl_cam, transl_cam_gt)
                loss_dict["loss_transl_coarse"] = loss_transl_coarse
                loss_dict["loss_transl"] = loss_transl
                loss += self.lambda_transl * loss_transl_coarse
                loss += self.lambda_transl * loss_transl

        # local mesh loss
        gt_v3d = mesh_dict["gt"]
        pred_diff_v3d_local = mesh_dict["pred_diff_local"]
        loss_v3d_local = self.loss_fn(pred_diff_v3d_local, gt_v3d)
        loss_dict["loss_v3d_local"] = loss_v3d_local
        loss += self.lambda_v3d_local * loss_v3d_local

        # global mesh loss
        gt = mesh_dict["gt_joints"]
        F = gt.shape[1]
        if F > 1:
            gt_vel = gt[:, 1:] - gt[:, :-1]
        if self.task in ["motion", "video"]:
            pred_diff_rel = self.joint_regressor @ mesh_dict["pred_diff_rel"]
            loss_j3d_pos_rel = self.loss_fn(pred_diff_rel, gt)
            loss_dict["loss_j3d_pos_rel"] = loss_j3d_pos_rel
            loss += self.lambda_j3d_pos * loss_j3d_pos_rel

            if F > 1:
                pred_diff_vel_rel = pred_diff_rel[:, 1:] - pred_diff_rel[:, :-1]
                loss_j3d_vel_rel = self.loss_fn(pred_diff_vel_rel, gt_vel)
                loss_dict["loss_j3d_vel_rel"] = loss_j3d_vel_rel
                loss += self.lambda_j3d_vel * loss_j3d_vel_rel

        if self.task in ["image", "video"]:
            pred_diff_coarse = self.joint_regressor @ mesh_dict["pred_diff_coarse"]
            pred_diff = self.joint_regressor @ mesh_dict["pred_diff"]
            loss_j3d_pos_coarse = self.loss_fn(pred_diff_coarse, gt)
            loss_j3d_pos = self.loss_fn(pred_diff, gt)
            loss_dict["loss_j3d_pos_coarse"] = loss_j3d_pos_coarse
            loss_dict["loss_j3d_pos"] = loss_j3d_pos
            loss += self.lambda_j3d_pos * loss_j3d_pos_coarse
            loss += self.lambda_j3d_pos * loss_j3d_pos

            if use_transl:
                gt_j2d = self.normalize_2d(self.project(gt, batch), batch)
                pred_j2d_coarse = self.normalize_2d(
                    self.project(pred_diff_coarse, batch), batch
                )
                pred_j2d = self.normalize_2d(self.project(pred_diff, batch), batch)
                loss_j2d_coarse = (pred_j2d_coarse - gt_j2d).abs().mean()
                loss_j2d = (pred_j2d - gt_j2d).abs().mean()
                loss_dict["loss_j2d_coarse"] = loss_j2d_coarse
                loss_dict["loss_j2d"] = loss_j2d
                loss += self.lambda_j2d * loss_j2d_coarse
                loss += self.lambda_j2d * loss_j2d
            elif "j2d" in batch:
                # supervise with 2D joints on image datasets
                # we use weak perspective projection without cliff transformation for computing 2D joints here
                gt_j2d = batch["j2d"]
                gt_j2d, conf = gt_j2d[..., :2], gt_j2d[..., 2:]

                gt_transl = traj_dict["gt"][1]
                pred_mesh_coarse = mesh_dict["pred_diff_coarse"] - gt_transl
                pred_mesh = mesh_dict["pred_diff"] - gt_transl

                pred_j3d_coarse = self.joint_regressor_h36m @ pred_mesh_coarse
                pred_j3d_coarse = pred_j3d_coarse[:, :, H36M_TO_J17]
                pred_j2d_coarse = self.project_weak_perspective(
                    pred_j3d_coarse, out["cam_traj_coarse_weak"]
                )
                loss_j2d_coarse = ((pred_j2d_coarse - gt_j2d).abs() * conf).mean()
                loss_dict["loss_j2d_coarse"] = loss_j2d_coarse
                loss += self.lambda_j2d * loss_j2d_coarse

        if self.task == "video":
            pred_diff_vel_coarse = pred_diff_coarse[:, 1:] - pred_diff_coarse[:, :-1]
            pred_diff_vel = pred_diff[:, 1:] - pred_diff[:, :-1]
            loss_j3d_vel_coarse = self.loss_fn(pred_diff_vel_coarse, gt_vel)
            loss_j3d_vel = self.loss_fn(pred_diff_vel, gt_vel)
            loss_dict["loss_j3d_vel_coarse"] = loss_j3d_vel_coarse
            loss_dict["loss_j3d_vel"] = loss_j3d_vel
            loss += self.lambda_j3d_vel * loss_j3d_vel_coarse
            loss += self.lambda_j3d_vel * loss_j3d_vel

        # accleration regularization
        if self.task == "motion" and F > 2:
            pred_diff_accel_rel = pred_diff_vel_rel[:, 1:] - pred_diff_vel_rel[:, :-1]

            loss_accel_rel = (pred_diff_accel_rel**2).mean()
            loss_dict["loss_accel_rel"] = loss_accel_rel
            loss += self.lambda_accel * loss_accel_rel

        if self.task == "video":
            pred_diff_accel = pred_diff_vel[:, 1:] - pred_diff_vel[:, :-1]
            loss_accel = (pred_diff_accel**2).mean()
            loss_dict["loss_accel"] = loss_accel
            loss += self.lambda_accel * loss_accel

        # skating loss
        if self.task == "motion" and F > 1:
            contact_lbl_gt = batch[
                "contact_label"
            ]  # [B, F-1, 6] 1 - in contact, 0 - not in contact

            stationary_joint_vel_rel = (
                pred_diff_rel[:, 1:, self.stationary_joint_index_list]
                - pred_diff_rel[:, 0:-1, self.stationary_joint_index_list]
            ) * self.fps  # [B, F-1, 6, 3]
            stationary_joint_vel_rel = torch.norm(stationary_joint_vel_rel, dim=-1)  # [B, F-1, 6]

            thresh = self.skating_vel_thres
            stationary_joint_violation_rel = torch.clamp(
                stationary_joint_vel_rel - thresh, min=0.0
            )

            weight_sum_rel = contact_lbl_gt.sum() + 1e-6
            loss_skating_rel = (stationary_joint_violation_rel * contact_lbl_gt).sum() / weight_sum_rel

            loss_dict["loss_skating_rel"] = loss_skating_rel
            loss += self.lambda_skating * loss_skating_rel

        if self.task == "video" and F > 1:
            contact_lbl_gt = batch[
                "contact_label"
            ]  # [B, F-1, 6] 1 - in contact, 0 - not in contact

            stationary_joint_vel = (
                pred_diff[:, 1:, self.stationary_joint_index_list]
                - pred_diff[:, 0:-1, self.stationary_joint_index_list]
            ) * self.fps  # [B, F-1, 6, 3]
            stationary_joint_vel = torch.norm(stationary_joint_vel, dim=-1)  # [B, F-1, 6]

            thresh = self.skating_vel_thres
            stationary_joint_violation = torch.clamp(
                stationary_joint_vel - thresh, min=0.0
            )

            weight_sum = contact_lbl_gt.sum() + 1e-6
            loss_skating = (stationary_joint_violation * contact_lbl_gt).sum() / weight_sum

            loss_dict["loss_skating"] = loss_skating
            loss += self.lambda_skating * loss_skating

        # contact loss
        if self.task in ["motion", "video"] and F > 1:
            contact_logits = out["contact"][:, :-1]
            contact_lbl_gt = batch["contact_label"]

            loss_contact = nn.functional.binary_cross_entropy_with_logits(
                contact_logits, contact_lbl_gt, reduction="mean"
            )
            loss_dict["loss_contact"] = loss_contact
            loss += self.lambda_contact * loss_contact

        loss_dict["loss"] = loss
        return loss_dict

    def compute_foot_sliding(self, gt_joints, pred_joints, thr=0.3):
        """
        Compute foot sliding error using stationary joints.
        The foot ground contact label is computed by the threshold on GT joints.
        
        Args:
            gt_joints: [B, F, J, 3] ground truth joint positions
            pred_joints: [B, F, J, 3] predicted joint positions
            thr: Threshold for contact detection (default 0.3, equivalent to 1cm/frame * fps)
            
        Returns:
            foot_sliding: scalar tensor representing foot sliding in mm/frame
        """
        # Get stationary joints from GT and predictions
        gt_foot_joints = gt_joints[:, :, self.stationary_joint_index_list]  # [B, F, N_foot, 3]
        pred_foot_joints = pred_joints[:, :, self.stationary_joint_index_list]  # [B, F, N_foot, 3]
        
        # Compute displacement between consecutive frames for GT (to determine contact)
        gt_foot_disp = torch.norm(gt_foot_joints[:, 1:] - gt_foot_joints[:, :-1], dim=-1)  # [B, F-1, N_foot]
        gt_foot_vel = gt_foot_disp * self.fps  # velocity in m/s
        
        # Determine contact: when GT velocity is below threshold
        contact = gt_foot_vel < thr  # [B, F-1, N_foot]
        
        # Compute displacement for predictions
        pred_foot_disp = torch.norm(pred_foot_joints[:, 1:] - pred_foot_joints[:, :-1], dim=-1)  # [B, F-1, N_foot]
        
        # Compute foot sliding: predicted displacement during GT contact
        sliding_disp = pred_foot_disp[contact]  # Only frames where GT foot should be in contact
        
        if sliding_disp.numel() > 0:
            # Return mean sliding distance in mm/frame
            return sliding_disp.mean() * 1000
        else:
            # No contact detected, return zero
            return torch.tensor(0.0, device=gt_joints.device)
    
    def evaluate(self, batch, out, mesh_dict):
        """
        compute the evaluation metrics
        """
        metrics_dict = {}
        v3d_gt = mesh_dict["gt"]
        F = v3d_gt.shape[1]
        j3d_gt = mesh_dict["gt_joints"]

        # local metrics
        v3d_pred_local = mesh_dict["pred_local"]
        mve = torch.norm(v3d_gt - v3d_pred_local, dim=-1).mean() * 1e3
        j3d_pred_local = self.joint_regressor @ v3d_pred_local
        mpjpe = torch.norm(j3d_gt - j3d_pred_local, dim=-1).mean() * 1e3

        shape = j3d_gt.shape
        j3d_pred_local_sym = batch_compute_similarity_transform_torch(
            j3d_pred_local.view(-1, *shape[-2:]), j3d_gt.view(-1, *shape[-2:])
        ).view(*shape)
        pa_mpjpe = torch.norm(j3d_gt - j3d_pred_local_sym, dim=-1).mean() * 1e3

        metrics_dict.update(
            {
                "mve": mve,
                "mpjpe": mpjpe,
                "pa_mpjpe": pa_mpjpe,
            }
        )

        if F > 2:
            j3d_pred_local_vel = j3d_pred_local[:, 1:] - j3d_pred_local[:, :-1]
            j3d_pred_local_accel = (
                j3d_pred_local_vel[:, 1:] - j3d_pred_local_vel[:, :-1]
            )
            local_accel = torch.norm(j3d_pred_local_accel, dim=-1).mean() * self.fps**2
            metrics_dict.update(
                {
                    "local_accel": local_accel,
                }
            )

        if self.task in ["motion", "video"]:
            j3d_gt_vel = j3d_gt[:, 1:] - j3d_gt[:, :-1]

        # global metrics
        if self.task in ["image", "video"]:
            v3d_pred_coarse = mesh_dict["pred_coarse"]
            gmve_coarse = torch.norm(v3d_gt - v3d_pred_coarse, dim=-1).mean() * 1e3
            j3d_pred_coarse = self.joint_regressor @ v3d_pred_coarse
            gmpjpe_coarse = torch.norm(j3d_gt - j3d_pred_coarse, dim=-1).mean() * 1e3

            v3d_pred = mesh_dict["pred"]
            gmve = torch.norm(v3d_gt - v3d_pred, dim=-1).mean() * 1e3
            j3d_pred = self.joint_regressor @ v3d_pred
            gmpjpe = torch.norm(j3d_gt - j3d_pred, dim=-1).mean() * 1e3

            metrics_dict.update(
                {
                    "gmve_coarse": gmve_coarse,
                    "gmpjpe_coarse": gmpjpe_coarse,
                    "gmve": gmve,
                    "gmpjpe": gmpjpe,
                }
            )

            if self.task == "video" and F > 2:
                j3d_pred_vel = j3d_pred[:, 1:] - j3d_pred[:, :-1]
                gmpjve = torch.norm(j3d_gt_vel - j3d_pred_vel, dim=-1).mean() * self.fps

                j3d_pred_accel = j3d_pred_vel[:, 1:] - j3d_pred_vel[:, :-1]
                accel = torch.norm(j3d_pred_accel, dim=-1).mean() * self.fps**2
                p_accel = (
                    torch.norm(j3d_pred_accel[:, :, 0], dim=-1).mean() * self.fps**2
                )

                metrics_dict.update(
                    {
                        "gmpjve": gmpjve,
                        "accel": accel,
                        "p_accel": p_accel,
                    }
                )

        else:
            v3d_pred_rel = mesh_dict["pred_rel"]
            gmve_rel = torch.norm(v3d_gt - v3d_pred_rel, dim=-1).mean() * 1e3
            j3d_pred_rel = self.joint_regressor @ v3d_pred_rel
            gmpjpe_rel = torch.norm(j3d_gt - j3d_pred_rel, dim=-1).mean() * 1e3

            metrics_dict.update(
                {
                    "gmve_rel": gmve_rel,
                    "gmpjpe_rel": gmpjpe_rel,
                }
            )

            if F > 2:
                j3d_pred_rel_vel = j3d_pred_rel[:, 1:] - j3d_pred_rel[:, :-1]
                gmpjve_rel = (
                    torch.norm(j3d_gt_vel - j3d_pred_rel_vel, dim=-1).mean() * self.fps
                )

                j3d_pred_rel_accel = j3d_pred_rel_vel[:, 1:] - j3d_pred_rel_vel[:, :-1]
                accel_rel = torch.norm(j3d_pred_rel_accel, dim=-1).mean() * self.fps**2
                metrics_dict.update(
                    {
                        "gmpjve_rel": gmpjve_rel,
                        "accel_rel": accel_rel,
                    }
                )

        # quantization error for reference
        v3d_input = mesh_dict["input"]
        qe = torch.norm(v3d_gt - v3d_input, dim=-1).mean() * 1e3

        metrics_dict.update(
            {
                "qe": qe,
            }
        )

        if "acc" in out:
            metrics_dict["acc"] = out["acc"]

        # foot skating metrics
        if self.task == "motion" and F > 1:
            # skating for rel prediction
            fs_rel =  self.compute_foot_sliding(j3d_gt, j3d_pred_rel) 
            
            metrics_dict.update({
                "fs_rel": fs_rel,
            })

        if self.task == "video" and F > 1:
            fs = self.compute_foot_sliding(j3d_gt, j3d_pred)

            metrics_dict.update({
                "fs": fs,
            })

        # contact accuracy
        if self.task in ["motion", "video"] and F > 1:
            contact_logits = out["contact"][:, :-1]
            contact_lbl_pred = contact_logits.detach() > 0.0
            contact_lbl_gt = batch["contact_label"]

            correct = (contact_lbl_pred == contact_lbl_gt).float()
            contact_acc = correct.mean()

            metrics_dict.update(
                {
                    "contact_acc": contact_acc,
                }
            )

        return metrics_dict