import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import defaultdict

def fit_smplx(vertices, evaluator, ret_error=False, **kwargs):
    from pytorch3d.transforms import axis_angle_to_matrix

    B, F = vertices.shape[:2]
    vertices = vertices.reshape(B * F, -1, 3) 

    cfg = {
        "num_iter": 1,
        "beta_regularizer": 1,
        "share_beta": True
    }
    cfg.update(kwargs)
    fit_result = evaluator.fitter.fit(vertices, joint_weights=evaluator.joint_weights, **cfg)
    pose_rotvecs, betas, trans = fit_result["pose_rotvecs"], fit_result["shape_betas"], fit_result["trans"]
    pose_rotmat = axis_angle_to_matrix(pose_rotvecs.reshape(-1, 3)).reshape(B, F, -1, 3, 3)
    betas = betas.reshape(B, F, -1)
    trans = trans.reshape(B, F, 3) 
    bm_params = {
        "body_pose": pose_rotmat[:, :, 1:22],
        "betas": betas,
        "global_orient": pose_rotmat[:, :, 0],
        "transl": trans,
    }
    bm_output = evaluator.bm_neutral(**bm_params)
    full_vertices = bm_output.full_vertices.clone().detach()

    if ret_error:
        # check fitting error
        recon_vertices = bm_output.vertices.detach().reshape(B * F, -1, 3) 
        recon_joints = bm_output.joints.detach().reshape(B * F, -1, 3)
        joints = evaluator.bm_neutral.joint_regressor @ recon_vertices
        v2v_error = torch.norm(recon_vertices - vertices, dim=-1).mean().item() * 1e3
        j2j_error = torch.norm(joints - recon_joints, dim=-1).mean().item() * 1e3
        return full_vertices, v2v_error, j2j_error
    else:
        return full_vertices

class EvalMetrics:
    """Evaluation metrics calculator for human motion estimation."""
    
    def __init__(self, fps: int = 30, has_visibility_mask: bool = False, gt_free_mode: bool = False, ground_axis: str = 'y'):
        self.fps = fps
        self.has_visibility_mask = has_visibility_mask
        self.gt_free_mode = gt_free_mode
        self.ground_axis = ground_axis  # 'x', 'y', or 'z'
        
        # Set up axis indices based on ground_axis
        if ground_axis == 'y':
            self.height_axis = 1  # y-axis is up
            self.horizontal_axes = [0, 2]  # x, z are horizontal
        elif ground_axis == 'x':
            self.height_axis = 0  # x-axis is up
            self.horizontal_axes = [1, 2]  # y, z are horizontal
        elif ground_axis == 'z':
            self.height_axis = 2  # z-axis is up
            self.horizontal_axes = [0, 1]  # x, y are horizontal
        else:
            raise ValueError(f"Invalid ground_axis: {ground_axis}. Must be 'x', 'y', or 'z'")
        
    def compute_gt_free_metrics(self, pred_joints_world: torch.Tensor, 
                               pred_verts_world: torch.Tensor,
                               ground_height: float) -> Dict[str, np.ndarray]:
        """
        Compute GT-free evaluation metrics.
        
        Args:
            pred_joints_world: [B, F, J, 3] predicted joints in world coordinates (torch.Tensor)
            pred_verts_world: [B, F, V, 3] predicted vertices in world coordinates (torch.Tensor)
            ground_height: Height of the ground floor
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing GT-free metrics
        """
        metrics = {}
        
        # Compute jitter
        metrics['jitter'] = self.compute_jitter(pred_joints_world)
        
        # Compute foot skating ratio (no GT needed)
        metrics['skating_ratio'] = self.compute_foot_skating_ratio(pred_joints_world, ground_height)
        
        # Compute ground penetration
        metrics['penetration_depth'], metrics['penetration_ratio'] = self.compute_ground_penetration_metrics(
            pred_joints_world, ground_height
        )
        
        # Compute global acceleration norm
        metrics['global_accel'] = self.compute_global_acceleration_norm(pred_joints_world)
        
        return metrics
        
    def compute_all_metrics(self, pred_joints_world: torch.Tensor, gt_joints_world: torch.Tensor,
                           pred_verts_world: torch.Tensor, gt_verts_world: torch.Tensor,
                           ground_height: float, visibility_mask: torch.Tensor = None,
                           eval_frame_range: tuple = None) -> Dict[str, np.ndarray]:
        """
        Compute all evaluation metrics in one call.
        
        Args:
            pred_joints_world: [B, F, J, 3] predicted joints in world coordinates (torch.Tensor)
            gt_joints_world: [B, F, J, 3] ground truth joints in world coordinates (torch.Tensor)
            pred_verts_world: [B, F, V, 3] predicted vertices in world coordinates (torch.Tensor)
            gt_verts_world: [B, F, V, 3] ground truth vertices in world coordinates (torch.Tensor)
            ground_height: Height of the ground floor
            visibility_mask: [B, F, J] visibility mask for joints (optional)
            eval_frame_range: (start_idx, end_idx) for evaluation range (optional)
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing all computed metrics
        """
        # Apply frame range if specified
        if eval_frame_range is not None:
            start_idx, end_idx = eval_frame_range
            pred_joints_world = pred_joints_world[:, start_idx:end_idx]
            gt_joints_world = gt_joints_world[:, start_idx:end_idx]
            pred_verts_world = pred_verts_world[:, start_idx:end_idx]
            gt_verts_world = gt_verts_world[:, start_idx:end_idx]
            assert visibility_mask is not None and visibility_mask.shape[1] == end_idx - start_idx, \
                "Visibility mask must match the evaluation frame range."
        metrics = {}
        
        # Compute camera coordinate metrics (using world coordinates with pelvis alignment)
        cam_metrics = self.compute_camcoord_metrics(
            pred_joints_world, gt_joints_world, pred_verts_world, gt_verts_world
        )
        
        # Add camera coordinate metrics to results
        metrics.update(cam_metrics)
        
        # Compute GMPJPE (Global MPJPE without alignment)
        metrics['gmpjpe'] = self.compute_gmpjpe(pred_joints_world, gt_joints_world)
        
        # Compute visibility-based MPJPE if mask is available
        if self.has_visibility_mask and visibility_mask is not None:
            vis_metrics = self.compute_visibility_mpjpe(pred_joints_world, gt_joints_world, visibility_mask)
            metrics.update(vis_metrics)
        
        # Compute global acceleration (without pelvis alignment)
        metrics['g_accel'] = self.compute_error_accel(pred_joints_world, gt_joints_world)
        
        # Compute foot sliding
        metrics['fs'] = self.compute_foot_sliding(gt_verts_world, pred_verts_world)
        
        # Compute jitter
        metrics['jitter'] = self.compute_jitter(pred_joints_world)
        
        # Compute RTE (Root Translation Error)
        metrics['rte'] = self.compute_rte(gt_joints_world[:, :, 0], pred_joints_world[:, :, 0])
        
        return metrics
    
    def compute_gmpjpe(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> np.ndarray:
        """
        Compute Global MPJPE without any alignment.
        
        Args:
            pred_joints: (..., J, 3) predicted joints
            gt_joints: (..., J, 3) ground truth joints
            
        Returns:
            GMPJPE values in mm
        """
        gmpjpe = self.compute_jpe(pred_joints, gt_joints) * 1000  # Convert to mm
        return gmpjpe
    
    def compute_camcoord_metrics(self, pred_j3d: torch.Tensor, target_j3d: torch.Tensor,
                                pred_verts: torch.Tensor, target_verts: torch.Tensor,
                                pelvis_idxs: List[int] = [1, 2]) -> Dict[str, np.ndarray]:
        """
        Compute camera coordinate metrics following the reference implementation.
        
        Args:
            pred_j3d: (..., J, 3) predicted joints
            target_j3d: (..., J, 3) target joints
            pred_verts: (..., V, 3) predicted vertices
            target_verts: (..., V, 3) target vertices
            pelvis_idxs: Indices for pelvis joints
            
        Returns:
            Dictionary with pa_mpjpe, mpjpe, pve, accel metrics
        """
        # Align by pelvis
        pred_j3d, target_j3d, pred_verts, target_verts = self.batch_align_by_pelvis(
            [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
        )

        # Metrics
        m2mm = 1000
        S1_hat = self.batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
        pa_mpjpe = self.compute_jpe(S1_hat, target_j3d) * m2mm
        mpjpe = self.compute_jpe(pred_j3d, target_j3d) * m2mm
        pve = self.compute_jpe(pred_verts, target_verts) * m2mm
        accel = self.compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d)

        return {
            "pa_mpjpe": pa_mpjpe,
            "mpjpe": mpjpe,
            "pve": pve,
            "accel": accel,
        }
    
    def compute_jpe(self, S1: torch.Tensor, S2: torch.Tensor) -> np.ndarray:
        """Compute joint position error."""
        return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    
    def batch_align_by_pelvis(self, data_list: List[torch.Tensor], 
                             pelvis_idxs: List[int] = [1, 2]) -> List[torch.Tensor]:
        """
        Align all data to the corresponding pelvis location.
        
        Args:
            data_list: [pred_j3d, target_j3d, pred_verts, target_verts]
            pelvis_idxs: Indices for pelvis joints
            
        Returns:
            Aligned data list
        """
        pred_j3d, target_j3d, pred_verts, target_verts = data_list

        pred_pelvis = pred_j3d[..., pelvis_idxs, :].mean(dim=-2, keepdims=True).clone()
        target_pelvis = target_j3d[..., pelvis_idxs, :].mean(dim=-2, keepdims=True).clone()

        # Align to the pelvis
        pred_j3d = pred_j3d - pred_pelvis
        target_j3d = target_j3d - target_pelvis
        pred_verts = pred_verts - pred_pelvis
        target_verts = target_verts - target_pelvis

        return [pred_j3d, target_j3d, pred_verts, target_verts]
    
    def batch_compute_similarity_transform_torch(self, S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
        """
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 closest to a set of 3D points S2.
        """
        # Reshape for batch processing
        original_shape = S1.shape
        if len(original_shape) > 3:
            S1 = S1.view(-1, *original_shape[-2:])
            S2 = S2.view(-1, *original_shape[-2:])
        
        transposed = False
        if S1.shape[-1] == 3 and S1.shape[-2] != 3:
            S1 = S1.permute(0, 2, 1)
            S2 = S2.permute(0, 2, 1)
            transposed = True
        
        assert S2.shape[1] == S1.shape[1]

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0, 2, 1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0], 1, 1)
        Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0, 2, 1)

        # Reshape back to original shape
        if len(original_shape) > 3:
            S1_hat = S1_hat.view(original_shape)

        return S1_hat
    
    def compute_error_accel(self, joints_gt: torch.Tensor, joints_pred: torch.Tensor, 
                           valid_mask: torch.Tensor = None) -> np.ndarray:
        """
        Compute acceleration error.
        
        Args:
            joints_gt: (..., F, J, 3) ground truth joints
            joints_pred: (..., F, J, 3) predicted joints
            valid_mask: (..., F) validity mask
            fps: Frame rate
            
        Returns:
            Acceleration error array
        """
        # Flatten batch dimensions
        original_shape = joints_gt.shape
        if len(original_shape) > 3:
            joints_gt = joints_gt.view(-1, *original_shape[-3:])
            joints_pred = joints_pred.view(-1, *original_shape[-3:])
        
        # Compute acceleration using finite differences
        accel_gt = joints_gt[:, :-2] - 2 * joints_gt[:, 1:-1] + joints_gt[:, 2:]
        accel_pred = joints_pred[:, :-2] - 2 * joints_pred[:, 1:-1] + joints_pred[:, 2:]
        
        # Compute error
        normed = torch.norm(accel_pred - accel_gt, dim=-1).mean(dim=-1)
        normed = normed * self.fps**2
        
        # Handle validity mask
        if valid_mask is None:
            new_vis = torch.ones(normed.shape, dtype=torch.bool)
        else:
            if len(original_shape) > 3:
                valid_mask = valid_mask.view(-1, original_shape[-3])
            invis = ~valid_mask
            invis1 = torch.roll(invis, -1, dims=-1)
            invis2 = torch.roll(invis, -2, dims=-1)
            new_invis = (invis | invis1 | invis2)[..., :-2]
            new_vis = ~new_invis
        
        return normed[new_vis].cpu().numpy()
    
    def compute_jitter(self, joints: torch.Tensor) -> np.ndarray:
        """
        Compute jitter of the motion.
        
        Args:
            joints: (..., F, J, 3) joint positions
            fps: Frame rate
            
        Returns:
            Jitter values
        """
        # Flatten batch dimensions for processing
        original_shape = joints.shape
        if len(original_shape) > 3:
            joints = joints.view(-1, *original_shape[-3:])
        
        pred_jitter = torch.norm(
            (joints[:, 3:] - 3 * joints[:, 2:-1] + 3 * joints[:, 1:-2] - joints[:, :-3]) * (self.fps**3),
            dim=-1,
        ).mean(dim=-1)

        return pred_jitter.cpu().numpy() / 10.0
    
    def compute_foot_sliding(self, target_verts: torch.Tensor, pred_verts: torch.Tensor, 
                           thr: float = 0.3) -> np.ndarray:
        """
        Compute foot sliding error.
        The foot ground contact label is computed by the threshold of 1 cm/frame
        
        Args:
            target_verts: (..., F, V, 3) target vertices
            pred_verts: (..., F, V, 3) predicted vertices
            thr: Threshold for contact detection
            Note: originally thr=1e-2, but multiplied by fps=30 here for consistency
            
        Returns:
            Foot sliding error, in mm/frame
            It should be mm/s for fair comparison of different fps, but we keep it for consistency with previous works.
        """
        assert target_verts.shape == pred_verts.shape
        
        # Flatten batch dimensions
        original_shape = target_verts.shape
        if len(original_shape) > 3:
            target_verts = target_verts.view(-1, *original_shape[-3:])
            pred_verts = pred_verts.view(-1, *original_shape[-3:])
        
        # Foot vertices indices (assuming SMPL topology)
        foot_idxs = [3216, 3387, 6617, 6787]
        
        # # Handle case where vertex count is different
        # if target_verts.shape[-2] != 6890:
        #     # Use a subset of vertices as foot proxies
        #     num_verts = target_verts.shape[-2]
        #     foot_idxs = [int(num_verts * 0.47), int(num_verts * 0.49), 
        #                 int(num_verts * 0.96), int(num_verts * 0.98)]
        
        errors = []
        for batch_idx in range(target_verts.shape[0]):
            # Compute contact label
            foot_loc = target_verts[batch_idx, :, foot_idxs]
            foot_disp = torch.norm(foot_loc[1:] - foot_loc[:-1], dim=-1) * self.fps
            contact = foot_disp < thr

            pred_feet_loc = pred_verts[batch_idx, :, foot_idxs]
            pred_disp = torch.norm(pred_feet_loc[1:] - pred_feet_loc[:-1], dim=-1)

            error = pred_disp[contact]
            if len(error) > 0:
                errors.append(error.cpu().numpy())
        
        if len(errors) > 0:
            return np.concatenate(errors) * 1000  # Convert to mm
        else:
            return np.array([])

    def compute_rte(self, target_trans: torch.Tensor, pred_trans: torch.Tensor) -> np.ndarray:
        """
        Compute Root Translation Error (RTE).
        
        Args:
            target_trans: (..., F, 3) target root translations
            pred_trans: (..., F, 3) predicted root translations
            
        Returns:
            RTE values normalized by displacement
        """
        # Flatten batch dimensions
        original_shape = target_trans.shape
        if len(original_shape) > 2:
            target_trans = target_trans.view(-1, *original_shape[-2:])
            pred_trans = pred_trans.view(-1, *original_shape[-2:])
        
        errors = []
        for batch_idx in range(target_trans.shape[0]):
            target_seq = target_trans[batch_idx]
            pred_seq = pred_trans[batch_idx]
            
            # Compute global alignment
            _, rot, trans = self.align_pcl(target_seq[None, :], pred_seq[None, :], fixed_scale=True) 
            pred_trans_hat = (torch.einsum("tij,tnj->tni", rot, pred_seq[None, :]) + trans[None, :])[0]

            # Compute the entire displacement of ground truth trajectory
            disps, disp = [], 0
            for p1, p2 in zip(target_seq, target_seq[1:]):
                delta = torch.norm(p2 - p1, dim=-1)
                disp += delta
                disps.append(disp)

            # Compute absolute root-translation-error (RTE)
            rte = torch.norm(target_seq - pred_trans_hat, dim=-1)

            # Normalize it to the displacement
            if disp > 0:
                normalized_rte = (rte / disp).cpu().numpy()
                errors.append(normalized_rte)
        
        if len(errors) > 0:
            return np.concatenate(errors) * 1e2  # Convert to cm
        else:
            return np.array([])
   
    def align_pcl(self, Y, X, weight=None, fixed_scale=False):
        """align similarity transform to align X with Y using umeyama method
        X' = s * R * X + t is aligned with Y
        :param Y (*, N, 3) first trajectory
        :param X (*, N, 3) second trajectory
        :param weight (*, N, 1) optional weight of valid correspondences
        :returns s (*, 1), R (*, 3, 3), t (*, 3)
        """
        # Ensure all tensors are on the same device
        device = Y.device
        
        *dims, N, _ = Y.shape
        N = torch.ones(*dims, 1, 1, device=device, dtype=Y.dtype) * N

        if weight is not None:
            Y = Y * weight
            X = X * weight
            N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

        # subtract mean
        my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
        mx = X.sum(dim=-2) / N[..., 0]
        y0 = Y - my[..., None, :]  # (*, N, 3)
        x0 = X - mx[..., None, :]

        if weight is not None:
            y0 = y0 * weight
            x0 = x0 * weight

        # correlation
        C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
        U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

        S = torch.eye(3, device=device, dtype=Y.dtype).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
        neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
        S[neg, 2, 2] = -1

        R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

        D = torch.diag_embed(D)  # (*, 3, 3)
        if fixed_scale:
            s = torch.ones(*dims, 1, device=device, dtype=Y.dtype)
        else:
            var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
            s = torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / var[..., 0]  # (*, 1)

        t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

        return s, R, t
 
    def compute_visibility_mpjpe(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor, 
                               visibility_mask: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute MPJPE for visible and occluded joints separately.
        
        Args:
            pred_joints: (..., J, 3) predicted joints
            gt_joints: (..., J, 3) ground truth joints
            visibility_mask: (..., J) boolean mask where True indicates visible joints
            
        Returns:
            Dictionary with visible_mpjpe and occluded_mpjpe
        """
        # Align by pelvis first
        pred_joints_aligned, gt_joints_aligned, _, _ = self.batch_align_by_pelvis(
            [pred_joints, gt_joints, pred_joints, gt_joints], pelvis_idxs=[1, 2]
        )
        
        # Compute joint position errors
        jpe = torch.sqrt(((pred_joints_aligned - gt_joints_aligned) ** 2).sum(dim=-1)) * 1000  # Convert to mm
        
        # Separate visible and occluded joints
        visible_mask = visibility_mask.bool()
        occluded_mask = ~visible_mask
        
        # Extract errors for visible and occluded joints
        visible_jpe = jpe[visible_mask]
        occluded_jpe = jpe[occluded_mask]
        
        return {
            "visible_mpjpe": visible_jpe.cpu().numpy(),
            "occluded_mpjpe": occluded_jpe.cpu().numpy(),
        }

    def compute_foot_skating_ratio(self, pred_joints_world: torch.Tensor, 
                                 ground_height: float, thresh_vel: float = 0.1,
                                 thresh_height: float = 0.1) -> np.ndarray:
        """
        Compute foot skating ratio based on foot velocity and height.
        Following the exact reference implementation.
        
        Args:
            pred_joints_world: (..., F, J, 3) predicted joints in world coordinates
            ground_height: Height of the ground floor
            thresh_vel: Velocity threshold for skating detection (m/s)
            thresh_height: Height threshold for contact detection (m)
            
        Returns:
            Skating ratio array
        """
        # Flatten batch dimensions
        original_shape = pred_joints_world.shape
        if len(original_shape) > 3:
            pred_joints_world = pred_joints_world.view(-1, *original_shape[-3:])
        
        # Foot joint indices based on reference: left ankle, left foot, right ankle, right foot
        foot_joint_idxs = [7, 10, 8, 11]
        
        joints_foot_rec = pred_joints_world[:, :, foot_joint_idxs]  # [F, 4, 3]
        
        horizontal_axes = self.horizontal_axes  # [0, 2] for y-up, [1, 2] for x-up, [0, 1] for z-up
        foot_vel_horizontal = joints_foot_rec[:, 1:, :, horizontal_axes] - joints_foot_rec[:, :-1, :, horizontal_axes]  # [F-1, 4, 2]
        joints_feet_horizon_vel_rec = torch.norm(foot_vel_horizontal, dim=-1) * self.fps  # [F-1, 4]
        
        # Get foot heights following reference
        joints_feet_height_rec = joints_foot_rec[:, :-1, :, self.height_axis]  # [F-1, 4]
        joints_feet_height_rec = joints_feet_height_rec - ground_height
        
        skating_rec_left = (joints_feet_horizon_vel_rec[:, :, 0] > thresh_vel) & \
                            (joints_feet_horizon_vel_rec[:, :, 1] > thresh_vel) & \
                            (joints_feet_height_rec[:, :, 0] < (thresh_height + 0.05)) & \
                            (joints_feet_height_rec[:, :, 1] < thresh_height)
        
        skating_rec_right = (joints_feet_horizon_vel_rec[:, :, 2] > thresh_vel) & \
                            (joints_feet_horizon_vel_rec[:, :, 3] > thresh_vel) & \
                            (joints_feet_height_rec[:, :, 2] < (thresh_height + 0.05)) & \
                            (joints_feet_height_rec[:, :, 3] < thresh_height)
        
        skating_rec = skating_rec_left * skating_rec_right  # [B, F-1]
        
        return skating_rec.cpu().numpy()

    def compute_ground_penetration_metrics(self, pred_joints_world: torch.Tensor, 
                                         ground_height: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ground penetration metrics following the reference implementation.
        
        Args:
            pred_joints_world: (..., F, J, 3) predicted joints in world coordinates
            ground_height: Height of the ground floor
            
        Returns:
            Tuple of (penetration_depth, penetration_ratio)
        """
        # Flatten batch dimensions
        original_shape = pred_joints_world.shape
        if len(original_shape) > 3:
            pred_joints_world = pred_joints_world.view(-1, *original_shape[-3:])
        
        # Focus on foot joints (ankle and foot) - joints [10, 11] as in reference
        foot_joint_idxs = [10, 11]  # Left foot, right foot
        
        # Extract foot joints: [B, F, 2, 3]
        foot_joints = pred_joints_world[:, :, foot_joint_idxs]  # [B, F, 2, 3]
        
        # Following reference implementation exactly:
        pene_dist = foot_joints[:, :, :, self.height_axis] - ground_height  # [B, F, 2]
        pene_freq = pene_dist < -0.05  # [B, F, 2]
        pene_freq = pene_freq.float().mean(dim=-1)  # [B, F] - mean across foot joints
        pene_dist[pene_dist >= 0] = 0
        pene_dist_mean = pene_dist.mean(dim=-1)  # [B, F] - mean across foot joints
        
        # Convert to final metrics
        penetration_depth = (-pene_dist_mean).cpu().numpy() * 1000  # Convert to mm, only negative values
        penetration_ratio = pene_freq.cpu().numpy() * 100  # Convert to percentage
        
        return penetration_depth, penetration_ratio.flatten()

    def compute_global_acceleration_norm(self, pred_joints_world: torch.Tensor) -> np.ndarray:
        """
        Compute global acceleration norm (without GT) - vectorized version.
        
        Args:
            pred_joints_world: (..., F, J, 3) predicted joints in world coordinates
            
        Returns:
            Global acceleration norm values
        """
        # Flatten batch dimensions
        original_shape = pred_joints_world.shape
        if len(original_shape) > 3:
            pred_joints_world = pred_joints_world.view(-1, *original_shape[-3:])
        
        # Filter out sequences with less than 3 frames
        valid_sequences = pred_joints_world.shape[1] >= 3
        if not valid_sequences:
            return np.array([])
        
        # Compute acceleration using finite differences for all sequences at once
        # accel = joints[2:] - 2 * joints[1:-1] + joints[:-2]  # [B, F-2, J, 3]
        accel = (pred_joints_world[:, 2:] - 2 * pred_joints_world[:, 1:-1] + pred_joints_world[:, :-2]) * (self.fps ** 2)
        
        # Compute norm across all joints and dimensions: [B, F-2]
        accel_norm = torch.norm(accel, dim=-1).mean(dim=-1)  # [B, F-2]
        
        return accel_norm.cpu().numpy().flatten()


class EvalResultsManager:
    """Manager for collecting and aggregating evaluation results."""
    
    def __init__(self, gt_free_mode: bool = False):
        self.gt_free_mode = gt_free_mode
        
        if gt_free_mode:
            # GT-free metrics only
            self.jitter_list = defaultdict(list)
            self.penetration_depth_list = defaultdict(list)
            self.penetration_ratio_list = defaultdict(list)
            self.skating_ratio_list = defaultdict(list)
            self.global_accel_list = defaultdict(list)  # Added global acceleration
        else:
            # Updated for new metrics
            self.pa_mpjpe_list = defaultdict(list)
            self.mpjpe_list = defaultdict(list)
            self.gmpjpe_list = defaultdict(list)
            self.pve_list = defaultdict(list)
            self.accel_list = defaultdict(list)
            self.g_accel_list = defaultdict(list)  # Global acceleration
            self.fs_list = defaultdict(list)
            self.jitter_list = defaultdict(list)
            self.rte_list = defaultdict(list)
            # New visibility-based metrics
            self.visible_mpjpe_list = defaultdict(list)
            self.occluded_mpjpe_list = defaultdict(list)
        
    def add_results_from_metrics_dict(self, recording_name: str, metrics: Dict[str, np.ndarray]):
        """Add evaluation results from a metrics dictionary."""
        if self.gt_free_mode:
            self.jitter_list[recording_name].append(metrics['jitter'])
            self.penetration_depth_list[recording_name].append(metrics['penetration_depth'])
            self.penetration_ratio_list[recording_name].append(metrics['penetration_ratio'])
            self.skating_ratio_list[recording_name].append(metrics['skating_ratio'])
            self.global_accel_list[recording_name].append(metrics['global_accel'])  # Added
        else:
            self.pa_mpjpe_list[recording_name].append(metrics['pa_mpjpe'])
            self.mpjpe_list[recording_name].append(metrics['mpjpe'])
            self.gmpjpe_list[recording_name].append(metrics['gmpjpe'])
            self.pve_list[recording_name].append(metrics['pve'])
            self.accel_list[recording_name].append(metrics['accel'])
            self.g_accel_list[recording_name].append(metrics['g_accel'])
            self.fs_list[recording_name].append(metrics['fs'])
            self.jitter_list[recording_name].append(metrics['jitter'])
            self.rte_list[recording_name].append(metrics['rte'])
            
            # Add visibility-based metrics if available
            if 'visible_mpjpe' in metrics:
                self.visible_mpjpe_list[recording_name].append(metrics['visible_mpjpe'])
            if 'occluded_mpjpe' in metrics:
                self.occluded_mpjpe_list[recording_name].append(metrics['occluded_mpjpe'])
        
    def aggregate_results(self, recording_name_list: List[str]) -> Dict:
        """Aggregate results for multiple recordings."""
        if self.gt_free_mode:
            # Concatenate results for each recording
            for recording_name in recording_name_list:
                if len(self.jitter_list[recording_name]) > 0:
                    self.jitter_list[recording_name] = np.concatenate(
                        self.jitter_list[recording_name], axis=0
                    )
                else:
                    self.jitter_list[recording_name] = np.array([])

                if len(self.penetration_depth_list[recording_name]) > 0:
                    self.penetration_depth_list[recording_name] = np.concatenate(
                        self.penetration_depth_list[recording_name], axis=0
                    )
                else:
                    self.penetration_depth_list[recording_name] = np.array([])

                if len(self.penetration_ratio_list[recording_name]) > 0:
                    self.penetration_ratio_list[recording_name] = np.concatenate(
                        self.penetration_ratio_list[recording_name], axis=0
                    )
                else:
                    self.penetration_ratio_list[recording_name] = np.array([])

                if len(self.skating_ratio_list[recording_name]) > 0:
                    self.skating_ratio_list[recording_name] = np.concatenate(
                        self.skating_ratio_list[recording_name], axis=0
                    )
                else:
                    self.skating_ratio_list[recording_name] = np.array([])

                if len(self.global_accel_list[recording_name]) > 0:
                    self.global_accel_list[recording_name] = np.concatenate(
                        self.global_accel_list[recording_name], axis=0
                    )
                else:
                    self.global_accel_list[recording_name] = np.array([])
            
            # Aggregate all recordings
            self.jitter_list["all"] = np.concatenate([
                self.jitter_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.jitter_list[recording_name]) > 0
            ], axis=0)
            
            self.penetration_depth_list["all"] = np.concatenate([
                self.penetration_depth_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.penetration_depth_list[recording_name]) > 0
            ], axis=0)
            
            self.penetration_ratio_list["all"] = np.concatenate([
                self.penetration_ratio_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.penetration_ratio_list[recording_name]) > 0
            ], axis=0)
            
            self.skating_ratio_list["all"] = np.concatenate([
                self.skating_ratio_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.skating_ratio_list[recording_name]) > 0
            ], axis=0)
            
            self.global_accel_list["all"] = np.concatenate([
                self.global_accel_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.global_accel_list[recording_name]) > 0
            ], axis=0)
            
            # Compute final metrics
            result = {
                "jitter": self.jitter_list["all"].mean() if len(self.jitter_list["all"]) > 0 else 0.0,
                "penetration_depth": self.penetration_depth_list["all"].mean() if len(self.penetration_depth_list["all"]) > 0 else 0.0,
                "penetration_ratio": self.penetration_ratio_list["all"].mean() if len(self.penetration_ratio_list["all"]) > 0 else 0.0,
                "skating_ratio": self.skating_ratio_list["all"].mean() if len(self.skating_ratio_list["all"]) > 0 else 0.0,
                "global_accel": self.global_accel_list["all"].mean() if len(self.global_accel_list["all"]) > 0 else 0.0,
            }
        else:
            # Concatenate results for each recording
            for recording_name in recording_name_list:
                self.pa_mpjpe_list[recording_name] = np.concatenate(
                    self.pa_mpjpe_list[recording_name], axis=0
                )
                self.mpjpe_list[recording_name] = np.concatenate(
                    self.mpjpe_list[recording_name], axis=0
                )
                self.gmpjpe_list[recording_name] = np.concatenate(
                    self.gmpjpe_list[recording_name], axis=0
                )
                self.pve_list[recording_name] = np.concatenate(
                    self.pve_list[recording_name], axis=0
                )
                self.accel_list[recording_name] = np.concatenate(
                    self.accel_list[recording_name], axis=0
                )
                self.g_accel_list[recording_name] = np.concatenate(
                    self.g_accel_list[recording_name], axis=0
                )
                if len(self.fs_list[recording_name]) > 0:
                    self.fs_list[recording_name] = np.concatenate(
                        self.fs_list[recording_name], axis=0
                    )
                else:
                    self.fs_list[recording_name] = np.array([])
                
                if len(self.jitter_list[recording_name]) > 0:
                    self.jitter_list[recording_name] = np.concatenate(
                        self.jitter_list[recording_name], axis=0
                    )
                else:
                    self.jitter_list[recording_name] = np.array([])

                if len(self.rte_list[recording_name]) > 0:
                    self.rte_list[recording_name] = np.concatenate(
                        self.rte_list[recording_name], axis=0
                    )
                else:
                    self.rte_list[recording_name] = np.array([])
            
                # Handle visibility-based metrics
                if len(self.visible_mpjpe_list[recording_name]) > 0:
                    self.visible_mpjpe_list[recording_name] = np.concatenate(
                        self.visible_mpjpe_list[recording_name], axis=0
                    )
                else:
                    self.visible_mpjpe_list[recording_name] = np.array([])
                
                if len(self.occluded_mpjpe_list[recording_name]) > 0:
                    self.occluded_mpjpe_list[recording_name] = np.concatenate(
                        self.occluded_mpjpe_list[recording_name], axis=0
                    )
                else:
                    self.occluded_mpjpe_list[recording_name] = np.array([])
            
            # Aggregate all recordings
            self.pa_mpjpe_list["all"] = np.concatenate([
                self.pa_mpjpe_list[recording_name].flatten()
                for recording_name in recording_name_list
            ], axis=0)
            
            self.mpjpe_list["all"] = np.concatenate([
                self.mpjpe_list[recording_name].flatten()
                for recording_name in recording_name_list
            ], axis=0)
            
            self.gmpjpe_list["all"] = np.concatenate([
                self.gmpjpe_list[recording_name].flatten()
                for recording_name in recording_name_list
            ], axis=0)

            self.pve_list["all"] = np.concatenate([
                self.pve_list[recording_name].flatten()
                for recording_name in recording_name_list
            ], axis=0)
            
            self.accel_list["all"] = np.concatenate([
                self.accel_list[recording_name].flatten()
                for recording_name in recording_name_list
            ], axis=0)
            
            self.g_accel_list["all"] = np.concatenate([
                self.g_accel_list[recording_name].flatten()
                for recording_name in recording_name_list
            ], axis=0)
            
            self.fs_list["all"] = np.concatenate([
                self.fs_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.fs_list[recording_name]) > 0
            ], axis=0)
            
            self.jitter_list["all"] = np.concatenate([
                self.jitter_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.jitter_list[recording_name]) > 0
            ], axis=0)
            
            self.rte_list["all"] = np.concatenate([
                self.rte_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.rte_list[recording_name]) > 0
            ], axis=0)
            
            # Aggregate visibility-based metrics
            visible_arrays = [
                self.visible_mpjpe_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.visible_mpjpe_list[recording_name]) > 0
            ]
            self.visible_mpjpe_list["all"] = np.concatenate(visible_arrays, axis=0) if visible_arrays else np.array([])
            
            occluded_arrays = [
                self.occluded_mpjpe_list[recording_name].flatten()
                for recording_name in recording_name_list if len(self.occluded_mpjpe_list[recording_name]) > 0
            ]
            self.occluded_mpjpe_list["all"] = np.concatenate(occluded_arrays, axis=0) if occluded_arrays else np.array([])
            
            # Compute final metrics
            result = {
                "pa_mpjpe": self.pa_mpjpe_list["all"].mean(),
                "mpjpe": self.mpjpe_list["all"].mean(),
                "gmpjpe": self.gmpjpe_list["all"].mean(),
                "pve": self.pve_list["all"].mean(),
                "accel": self.accel_list["all"].mean(),
                "g_accel": self.g_accel_list["all"].mean(),
                "fs": self.fs_list["all"].mean() if len(self.fs_list["all"]) > 0 else 0.0,
                "jitter": self.jitter_list["all"].mean() if len(self.jitter_list["all"]) > 0 else 0.0,
                "rte": self.rte_list["all"].mean() if len(self.rte_list["all"]) > 0 else 0.0,
            }
            
            # Add visibility-based metrics if available
            if len(self.visible_mpjpe_list["all"]) > 0:
                result["visible_mpjpe"] = self.visible_mpjpe_list["all"].mean()
            if len(self.occluded_mpjpe_list["all"]) > 0:
                result["occluded_mpjpe"] = self.occluded_mpjpe_list["all"].mean()
                
        return result
        
    def print_results(self, final_metrics: Dict):
        """Print evaluation results in a formatted way."""
        if self.gt_free_mode:
            print("\n --------------- GT-free evaluation metrics -------------")
            print("Jitter: {:0.3f}".format(final_metrics["jitter"]))
            print("Skating Ratio: {:0.3f}".format(final_metrics["skating_ratio"]))
            print("Penetration Depth (mm): {:0.2f}".format(final_metrics["penetration_depth"]))
            print("Penetration Ratio (%): {:0.2f}".format(final_metrics["penetration_ratio"]))
            print("Global Accel (m/s^2): {:0.2f}".format(final_metrics["global_accel"]))
        else:
            print("\n --------------- evaluation metrics -------------")
            print("PA-MPJPE (mm): {:0.2f}".format(final_metrics["pa_mpjpe"]))
            print("MPJPE (mm): {:0.2f}".format(final_metrics["mpjpe"]))
            # Print visibility-based metrics if available
            if "visible_mpjpe" in final_metrics:
                print("Visible MPJPE (mm): {:0.2f}".format(final_metrics["visible_mpjpe"]))
            if "occluded_mpjpe" in final_metrics:
                print("Occluded MPJPE (mm): {:0.2f}".format(final_metrics["occluded_mpjpe"]))
            print("PVE (mm): {:0.2f}".format(final_metrics["pve"]))
            print("GMPJPE (mm): {:0.2f}".format(final_metrics["gmpjpe"]))
            print("Accel (m/s^2): {:0.2f}".format(final_metrics["accel"]))
            print("Global Accel (m/s^2): {:0.2f}".format(final_metrics["g_accel"]))
            print("RTE (cm): {:0.2f}".format(final_metrics["rte"]))
            print("Jitter: {:0.3f}".format(final_metrics["jitter"]))
            print("Foot Sliding (mm): {:0.2f}".format(final_metrics["fs"]))