import numpy as np
import cv2
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import datetime
import os, json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R


# estimated floor height from scene mesh
# z axis up
prox_floor_height = {
    "N0Sofa": -0.9843093165454873,
    "MPH1Library": -0.34579620031341207,
    "N3Library": -0.6736229583361132,
    "N3Office": -0.7772727989022952,
    "BasementSittingBooth": -0.767080139846674,
    "MPH8": -0.41432886722717904,
    "MPH11": -0.7169139211234009,
    "MPH16": -0.8408992040141058,
    "MPH112": -0.6419028605753081,
    "N0SittingBooth": -0.6677103008966809,
    "N3OpenArea": -1.0754909672969915,
    "Werkraum": -0.6777057869851316,
}

# y axis up
egobody_floor_height = {
    "seminar_g110": -1.660,
    "seminar_d78": -0.810,
    "seminar_j716": -0.8960,
    "seminar_g110_0315": -0.73,
    "seminar_d78_0318": -1.03,
    "seminar_g110_0415": -0.77,
}

def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # !!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

def get_logger(logdir):
    logger = logging.getLogger("emotion")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord


def projectPoints(v, cam):
    v = v.reshape((-1, 3)).copy()
    return cv2.projectPoints(
        v,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray(cam["camera_mtx"]),
        np.asarray(cam["k"]),
    )[0].squeeze()


def perspective_projection(
    points,
    # translation,
    focal_length,
    camera_center=None,
    rotation=None,
):
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = (
            torch.eye(3, device=points.device, dtype=points.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
    if camera_center is None:
        camera_center = torch.zeros(
            batch_size, 2, device=points.device, dtype=points.dtype
        )
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    # points = points + translation.unsqueeze(1)
    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)
    return projected_points[:, :, :-1]

def estimate_angular_velocity(rot_seq, dRdt):
    """
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    """
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    # dRdt = self.estimate_linear_velocity(rot_seq, h)

    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], dim=-1)  # [B, T, ..., 3]
    return w


def estimate_angular_velocity_np(rot_seq, dRdt):
    # rot_seq: [T, 3, 3]
    # dRdt: [T, 3, 3]
    R = rot_seq
    RT = np.transpose(R, (0, -1, -2))
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)  # [B, T, ..., 3]
    return w

def batch_compute_similarity_transform_torch(S1, S2):
    """
    Inspired from https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427#file-batch_procrustes_pytorch-py
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
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
    # dtype = K.dtype
    # U, _, V = torch.svd(K.to(torch.float32))
    # U = U.to(dtype)
    # V = V.to(dtype)
    U, _, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    # BF16 compatable...
    # UVT = U.bmm(V.permute(0, 2, 1))
    # UVT = UVT.to(torch.float32)
    # Z[:, -1, -1] *= torch.sign(torch.det(UVT).to(dtype))
    # Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

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

    return S1_hat
