import numpy as np
import os
import pickle
import torch
from scipy.spatial.transform import Rotation as R
import copy
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

face_joint_indx = [2, 1, 17, 16]
# Right/Left foot, same for SMPL and SMPLX
fid_r, fid_l = [8, 11], [7, 10]

def get_bm_params(smpl_params, device=None):
    bm_params = {
        k: torch.from_numpy(v).float().to(device=device) for k, v in smpl_params.items()
    }
    bm_params["body_pose"] = axis_angle_to_matrix(bm_params["body_pose"].view(*bm_params["body_pose"].shape[:-1], 21, 3))
    bm_params["global_orient"] = axis_angle_to_matrix(bm_params["global_orient"])

    return bm_params

def get_smpl_batch_params(bm_params):
    smpl_batch_params = {}
    # local pose
    smpl_batch_params["pose_rotmat"] = bm_params["body_pose"]
    smpl_batch_params["pose_6d"] = matrix_to_rotation_6d(smpl_batch_params["pose_rotmat"])

    # shape
    smpl_batch_params["betas"] = bm_params["betas"]

    # trajectory
    smpl_batch_params["rotation"] = bm_params["global_orient"]
    smpl_batch_params["translation"] = bm_params["transl"].unsqueeze(1)
    return smpl_batch_params
    

def update_globalRT_for_smplx(
    body_param_dict, trans_to_target_origin, smplx_model=None, device=None, delta_T=None
):
    """
    input:
        body_param_dict:
        smplx_model: the model to generate smplx mesh, given body_params
        trans_to_target_origin: coordinate transformation [4,4] mat
        delta_T: pelvis location?
    Output:
        body_params with new globalR and globalT, which are corresponding to the new coord system
    """

    ### step (1) compute the shift of pelvis from the origin
    bs = len(body_param_dict["transl"])

    # if delta_T is None:
    #     body_param_dict_torch = {}
    #     for key in body_param_dict.keys():
    #         body_param_dict_torch[key] = torch.FloatTensor(body_param_dict[key]).to(
    #             device
    #         )
    #     body_param_dict_torch["transl"] = torch.zeros([bs, 3], dtype=torch.float32).to(
    #         device
    #     )
    #     body_param_dict_torch["global_orient"] = torch.zeros(
    #         [bs, 3], dtype=torch.float32
    #     ).to(device)

    #     body_param_dict_torch["jaw_pose"] = torch.zeros(bs, 3).to(device)
    #     body_param_dict_torch["leye_pose"] = torch.zeros(bs, 3).to(device)
    #     body_param_dict_torch["reye_pose"] = torch.zeros(bs, 3).to(device)
    #     body_param_dict_torch["left_hand_pose"] = torch.zeros(bs, 45).to(device)
    #     body_param_dict_torch["right_hand_pose"] = torch.zeros(bs, 45).to(device)
    #     body_param_dict_torch["expression"] = torch.zeros(bs, 10).to(device)

    #     smpl_out = smplx_model(**body_param_dict_torch)
    #     delta_T = smpl_out.joints[:, 0, :]  # (bs, 3,)
    #     delta_T = delta_T.detach().cpu().numpy()

    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_param_dict["global_orient"]
    body_R_mat = R.from_rotvec(body_R_angle).as_matrix()  # to a [bs, 3,3] rotation mat
    body_T = body_param_dict["transl"]
    body_mat = np.zeros([bs, 4, 4])
    body_mat[:, :-1, :-1] = body_R_mat
    body_mat[:, :-1, -1] = body_T + delta_T
    body_mat[:, -1, -1] = 1

    ### step (3): perform transformation, and decalib the delta shift
    body_params_dict_new = copy.deepcopy(body_param_dict)
    if trans_to_target_origin.ndim == 2:
        # add batch dimension
        trans_to_target_origin = np.expand_dims(trans_to_target_origin, axis=0)  # [1, 4, 4]
        trans_to_target_origin = np.repeat(trans_to_target_origin, bs, axis=0)  # [bs, 4, 4]

    body_mat_new = np.matmul(trans_to_target_origin, body_mat)  # [bs, 4, 4]
    body_R_new = R.from_matrix(body_mat_new[:, :-1, :-1]).as_rotvec()
    body_T_new = body_mat_new[:, :-1, -1]
    body_params_dict_new["global_orient"] = body_R_new.reshape(-1, 3).astype(np.float32)
    body_params_dict_new["transl"] = (
        (body_T_new - delta_T).reshape(-1, 3).astype(np.float32)
    )
    return body_params_dict_new

def cano_seq_smplx(
    positions,
    smplx_params_dict,
    preset_floor_height=None,
    return_transf_mat=False,
    transf_matrix_0=None,
):
    """
    Perform canonicalization to the original motion sequence, such that:
    - the sueqnce is z+ axis up
    - frame 0 of the output sequence faces y+ axis
    - x/y coordinate of frame 0 is located at origin
    - foot on floor
    Use for AMASS and PROX (coordinate system z axis up)
    input:
        - positions: original joint positions (z axis up)
        - smplx_params_dict: original smplx params
        - preset_floor_height: if not None, the preset floor height
        - return_transf_mat: if True, also return the transf matrix for canonicalization
    Output:
        - cano_positions: canonicalized joint positions
        - cano_smplx_params_dict: canonicalized smplx params
        - transf_matrix (if return_transf_mat): the transf matrix for canonicalization
    """
    if transf_matrix_0 is None:
        transf_matrix_0 = np.eye(4) 
    ##### given a motion sequence, first rotate it to z axis up 
    cano_positions = positions.copy()
    cano_positions = np.matmul(cano_positions, transf_matrix_0[:3, :3].T)

    ##### positions: z axis up
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx

    ######################## Put on Floor
    if preset_floor_height:
        floor_height = preset_floor_height
    else:
        floor_height = cano_positions.min(axis=0).min(axis=0)[2]
    cano_positions[:, :, 2] -= floor_height  # z: up-axis, foot on ground

    ######################## transl such that XY for frame 0 is at origin
    root_pos_init = cano_positions[0]  # [22, 3]
    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])
    cano_positions = cano_positions - root_pose_init_xy

    ######################## transfrom such that frame 0 faces y+ axis
    joints_frame0 = cano_positions[0]  # [N, 3] joints of first frame
    across1 = joints_frame0[r_hip] - joints_frame0[l_hip]
    across2 = joints_frame0[sdr_r] - joints_frame0[sdr_l]
    x_axis = across1 + across2
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0, 0, 1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    transf_rotmat = np.stack([x_axis, y_axis, z_axis], axis=1)  # [3, 3]
    cano_positions = np.matmul(cano_positions, transf_rotmat)  # [T(/bs), 22, 3]

    ######################## canonicalization transf matrix for smpl params
    transf_matrix_1 = np.array(
        [
            [1, 0, 0, -root_pose_init_xy[0]],
            [0, 1, 0, -root_pose_init_xy[1]],
            [0, 0, 1, -floor_height],
            [0, 0, 0, 1],
        ]
    )
    transf_matrix_2 = np.zeros([4, 4])
    transf_matrix_2[0:3, 0:3] = transf_rotmat.T
    transf_matrix_2[-1, -1] = 1
    transf_matrix = np.matmul(transf_matrix_2, transf_matrix_1)
    transf_matrix = np.matmul(transf_matrix, transf_matrix_0)
    cano_smplx_params_dict = update_globalRT_for_smplx(
        smplx_params_dict,
        transf_matrix,
        delta_T=positions[:, 0] - smplx_params_dict["transl"],
    )

    if not return_transf_mat:
        return cano_positions, cano_smplx_params_dict
    else:
        return cano_positions, cano_smplx_params_dict, transf_matrix

def foot_detect(positions, thres=5e-5, up_axis="z"):
    """
    Detect if the feet are on the ground based on the positions of the feet joints.
    input:
        - positions: joint positions [T, J, 3]
        - thres: threshold for velocity
        - up_axis: the up axis, can be "x", "y", or "z"
    output:
        - foot_contact: foot contact label [T-1, 4]
    """
    if up_axis == "y":
        up_axis_dim = 1
    elif up_axis == "z":
        up_axis_dim = 2
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.18, 0.15])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, up_axis_dim]
    feet_l = (
        ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
    ).astype(float)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, up_axis_dim]
    feet_r = (
        ((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)
    ).astype(float)

    foot_contact = np.concatenate([feet_l, feet_r], axis=-1)  # [T-1, 4]
    return foot_contact

def get_contact_label(joints, vel_thres=0.15, fps=30, contact_hand=True):
    """
    get contact label based on joint velocity
    irrelevant to the coordinate system
    """
    if contact_hand:
        stationary_joint_indx =  [7, 10, 8, 11, 20, 21]
    else:
        stationary_joint_indx = [7, 10, 8, 11] # foot only 
    joint_vel = torch.norm(joints[1:, stationary_joint_indx] - joints[:-1, stationary_joint_indx], dim=-1) * fps
    contact_label = joint_vel < vel_thres # 1 as stable, 0 as moving
    return contact_label

##################### get global trajectory representation
class GlobalTrajHelper:
    def __init__(self):
        self.REPR_LIST = [
            # "rot_6d",
            "rot_rel_6d",
            # "trans",
            "trans_vel",
        ]

        self.REPR_DIM_DICT = {
            # "rot_6d": 6,
            "rot_rel_6d": 6,
            # "trans": 3,
            "trans_vel": 3,
        }

        # read Mean and Std
        mean_std_path = os.path.join("datasets", "mask_transformer", "amass_preprocess", f"amass_cum_traj.pkl")
        if not os.path.exists(mean_std_path):
            print(f"Mean and Std file not found at {mean_std_path}")
            print("Will recompute mean and std")
            return
        with open(mean_std_path, "rb") as f:
            mean_std_dict = pickle.load(f)
        self.Mean = mean_std_dict["Mean"]
        self.Std = mean_std_dict["Std"]
        self.Mean = torch.from_numpy(self.Mean)
        self.Std = torch.from_numpy(self.Std)
    
    def repr_tensor_to_dict(self, repr_tensor):
        """
        convert the global repr tensor to dict
        """
        data_dict = {}
        start_idx = 0
        for key in self.REPR_LIST:
            end_idx = start_idx + self.REPR_DIM_DICT[key]
            data_dict[key] = repr_tensor[..., start_idx:end_idx]
            start_idx = end_idx
        return data_dict

    def repr_dict_to_tensor(self, data_dict):
        """
        convert the global repr dict to tensor
        """
        is_tensor = isinstance(data_dict[self.REPR_LIST[0]], torch.Tensor)
        if is_tensor:
            repr_tensor = torch.cat([data_dict[key] for key in self.REPR_LIST], dim=-1)
        else:
            repr_tensor = np.concatenate([data_dict[key] for key in self.REPR_LIST], axis=-1)
        return repr_tensor


    @staticmethod
    def get_valid_mask(traj):
        """
        get the valid mask for the global representation
        """
        valid_mask = torch.ones_like(traj, dtype=torch.bool)
        # valid_mask[:, 0, :6] = False
        # valid_mask[:, -1, 6:12] = False
        # valid_mask[:, 0, 12:15] = False
        # valid_mask[:, -1, 15:] = False
        
        valid_mask[:, -1] = False # last frame, no relative gt
        return valid_mask


    def get_cano_traj_repr(self, rotation, translation, normalize=True):
        """
        calculate the motion representation for input sequence
        input:
            - rotation: global rotation matrix [B, F, 3, 3]
            - translation: global translation after composition [B, F, 1, 3]
        Output:
            - data_dict: global trajectory representation
        """
        if rotation.ndim == 3:
            rotation = rotation.unsqueeze(0)
            translation = translation.unsqueeze(0)
        F = rotation.shape[1]
        if F == 1:
            return None, torch.zeros(*rotation.shape[:2], 9, device=rotation.device)

        ##################### frame-to-frame relative repr #####################
        """relative rotation from frame t to t+1: dR_t = R_t^{-1} @ R_{t+1}"""
        rot_rel = rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:]  # [B, F-1, 3, 3]
        rot_rel_6d = matrix_to_rotation_6d(rot_rel)  # [B, F-1, 6]
        rot_rel_6d = torch.cat(
            [rot_rel_6d, rot_rel_6d[:, -1:]], dim=1
        )  # [B, F, 6], pad last frame

        """relative translation from frame t to t+1: dt_t = R_t^{-1} @ (t_{t+1} - t_t)"""
        trans_diff = translation[:, 1:] - translation[:, :-1]  # [B, F-1, 1, 3]
        trans_vel = (trans_diff @ rotation[:, :-1]).squeeze(2)  # [B, F-1, 3]
        trans_vel = torch.cat(
            [trans_vel, trans_vel[:, -1:]], dim=1
        )  # [B, F, 1, 3], pad last frame

        ################### final full body repr #####################
        data_dict = {
            "rot_rel_6d": rot_rel_6d,
            "trans_vel": trans_vel,
        }
        global_repr = self.repr_dict_to_tensor(data_dict)
        if normalize:
            global_repr = self.normalize(global_repr)

        return data_dict, global_repr


    def parse_cano_traj_repr(self, global_repr, R0, T0):
        """
        global_repr: predicted global representation [B, F, 9]
        R0: initial rotation matrix [B, 1, 3, 3]
        T0: initial translation [B, 1, 1, 3]
        batch: input batch with ground truth rotation and translation
        """
        global_repr = self.unnormalize(global_repr) 
        device = global_repr.device
        global_repr_dict = self.repr_tensor_to_dict(global_repr)
        B, F = global_repr_dict["rot_rel_6d"].shape[:2]
        
        rot_rel_6d = global_repr_dict["rot_rel_6d"]
        rot_rel = rotation_6d_to_matrix(rot_rel_6d)
        
        # Reconstruct absolute rotations: R_{t+1} = R_t @ dR_t
        rotation_list = []
        rotation_list.append(R0.squeeze(1))  # [B, 3, 3]
        for i in range(F - 1):
            rotation_list.append(rotation_list[-1] @ rot_rel[:, i])
        rotation = torch.stack(rotation_list, dim=1)

        trans_vel = global_repr_dict["trans_vel"].unsqueeze(2)
        
        # Reconstruct absolute translations: t_{t+1} = t_t + R_t @ dt_t
        translation_list = []
        translation_list.append(T0.squeeze(1))  # [B, 1, 3]
        for i in range(F - 1):
            trans_increment = trans_vel[:, i] @ rotation_list[i].transpose(-1, -2)
            translation_list.append(translation_list[-1] + trans_increment)
        translation = torch.stack(translation_list, dim=1)

        return rotation, translation

    def normalize(self, global_repr):
        """
        normalize the global representation
        """
        device = global_repr.device
        global_repr = (global_repr - self.Mean.to(device)) / self.Std.to(device)
        return global_repr
    
    def unnormalize(self, global_repr):
        """
        unnormalize the global representation
        """
        device = global_repr.device
        global_repr = global_repr * self.Std.to(device) + self.Mean.to(device)
        return global_repr

    @staticmethod
    def cliff_transform(cam_traj_weak, bbox):
        """
        cam_traj_weak: [B, F, 9]
        bbox: [B, F, 3]
        crop_size: int, optional
        crop_focal: float, optional
        """
        cam_rot_6d = cam_traj_weak[..., :6]
        cam_rot = rotation_6d_to_matrix(cam_rot_6d)

        cam_t = cam_traj_weak[..., 6:]
        s = cam_t[..., :1]
        # make sure the scale is positive 
        s = torch.exp(s)
        t = cam_t[..., 1:]
        # t_crop_z = 2 * crop_focal / (crop_size * s + 1e-9)
        # t_crop = torch.cat([t, t_crop_z], dim=-1)
        # t_full_x = t_crop[..., 0] + 2 * bbox[..., 0] / (bbox[..., 3] * s + 1e-9)
        # t_full_y = t_crop[..., 1] + 2 * bbox[..., 1] / (bbox[..., 3] * s + 1e-9)
        # t_full_z = t_crop[..., 2] * crop_size / (crop_focal * bbox[..., 3] + 1e-9)

        t_full_xy = t + 2 * bbox[..., :2] / (bbox[..., 2:] * s + 1e-9)
        t_full_z = 2 / (s * bbox[..., 2:] + 1e-9)
        cam_trans = torch.cat([t_full_xy, t_full_z], dim=-1)
        cam_trans.unsqueeze_(2) 
        return cam_rot, cam_trans

    def inv_cliff_transform(self, cam_rot, cam_trans, bbox):
        """
        cam_rot: [B, F, 3, 3]
        cam_trans: [B, F, 1, 3]
        bbox: [B, F, 3]
        """
        """
        Inverse the cliff transform to get the weak perspective camera trajectory.
        """
        cam_rot_6d = matrix_to_rotation_6d(cam_rot)

        t_full_xy = cam_trans[..., 0, :2]
        t_full_z = cam_trans[..., 0, 2:]
        s = 2 / (t_full_z * bbox[..., 2:] + 1e-9)
        t = t_full_xy - 2 * bbox[..., :2] / (bbox[..., 2:] * s + 1e-9)
        cam_traj_weak = torch.cat([cam_rot_6d, s, t], dim=-1)  # [B, F, 9]
        return cam_traj_weak
