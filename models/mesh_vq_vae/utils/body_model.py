"""
Code borrowed from HuMoR: https://github.com/davrempe/humor/blob/main/humor/body_model/body_model.py
"""

import os
import trimesh

import numpy as np
import torch
from torch import nn

from smplx import SMPL, SMPLH, SMPLX
from smplx import SMPLLayer, SMPLHLayer, SMPLXLayer
from smplx.vertex_ids import vertex_ids
from smplx.utils import Struct

SMPLH_PATH = "body_models/smplh"
NUM_BETAS = 10
NUM_JOINTS = 22
NUM_SMPLX_BODY_VERTICES = 9383


class BodyModel(nn.Module):
    """
    Wrapper around SMPLX body model class.
    """

    def __init__(
        self,
        bm_path,
        num_betas=10,
        batch_size=1,
        num_expressions=10,
        use_vtx_selector=False,
        model_type="smplh",
        gender="neutral",
    ):
        super(BodyModel, self).__init__()
        """
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        """
        self.model_type = model_type
        self.gender = gender

        bm_path = os.path.join(
            bm_path, model_type, f"{model_type.upper()}_{gender.upper()}.npz"
        )

        self.use_vtx_selector = use_vtx_selector
        cur_vertex_ids = None
        if self.use_vtx_selector:
            cur_vertex_ids = vertex_ids[model_type]
        data_struct = None
        if ".npz" in bm_path:
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding="latin1", allow_pickle=True)
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == "smplh":
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate(
                    [data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM - B))],
                    axis=-1,
                )  # super hacky way to let smplh use 16-size beta
        kwargs = {
            "model_type": model_type,
            "data_struct": data_struct,
            "num_betas": num_betas,
            "batch_size": batch_size,
            "num_expression_coeffs": num_expressions,
            "vertex_ids": cur_vertex_ids,
            "use_pca": False,
            "flat_hand_mean": True,
        }
        assert model_type in ["smpl", "smplh", "smplx"]
        if model_type == "smpl":
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
            self.faces = torch.from_numpy(self.bm.faces.astype(np.int32))
            self.register_buffer("joint_regressor", self.bm.J_regressor[:NUM_JOINTS])
        elif model_type == "smplh":
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
            self.faces = torch.from_numpy(self.bm.faces.astype(np.int32))
            self.register_buffer("joint_regressor", self.bm.J_regressor[:NUM_JOINTS])
        elif model_type == "smplx":
            self.bm = SMPLX(
                bm_path,
                num_betas=num_betas,
                batch_size=batch_size,
                use_pca=False,
                flat_hand_mean=True, 
            )
            self.num_joints = SMPLX.NUM_JOINTS
            self.faces = self.get_smplx_faces()
            # self.faces = torch.from_numpy(self.bm.faces.astype(np.int32))
            self.register_buffer("joint_regressor", self.bm.J_regressor[:NUM_JOINTS, :NUM_SMPLX_BODY_VERTICES])

    def get_smplx_faces(self):
        # for SMPLX, only keep body vertices, the eyeball meshes are discarded
        assert self.model_type == "smplx"

        # get faces for the body component of smplx mesh
        output = self.bm()
        vertices = output.vertices[0].detach().cpu().numpy()
        faces = self.bm.faces
        mesh = trimesh.Trimesh(vertices, faces)
        cc = mesh.split(only_watertight=False)
        body_mesh = cc[0]
        return torch.from_numpy(body_mesh.faces).int()

    def forward(
        self,
        global_orient=None,
        body_pose=None,
        hand_pose=None,
        jaw_pose=None,
        eye_pose=None,
        betas=None,
        transl=None,
        dmpls=None,
        expression=None,
        return_dict=False,
        **kwargs,
    ):
        """
        Note dmpls are not supported.
        """
        assert dmpls is None
        # Note that global orient and pose are in rotation matrix format, not axis-angle
        out_obj = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=(
                None
                if hand_pose is None
                else hand_pose[:, : (SMPLH.NUM_HAND_JOINTS * 3)]
            ),
            right_hand_pose=(
                None
                if hand_pose is None
                else hand_pose[:, (SMPLH.NUM_HAND_JOINTS * 3) :]
            ),
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=None if eye_pose is None else eye_pose[:, :3],
            reye_pose=None if eye_pose is None else eye_pose[:, 3:],
            return_full_pose=True,
            **kwargs,
        )

        out = {
            "output": out_obj,
            "vertices": out_obj.vertices if self.model_type != "smplx" else out_obj.vertices[..., :NUM_SMPLX_BODY_VERTICES, :],
            "faces": self.faces,
            "betas": out_obj.betas,
            "joints": out_obj.joints[:, :NUM_JOINTS],
            "pose_body": out_obj.body_pose,
            "full_pose": out_obj.full_pose,
        }
        if self.model_type in ["smplh", "smplx"]:
            out["pose_hand"] = torch.cat(
                [out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1
            )
        if self.model_type == "smplx":
            out["pose_jaw"] = out_obj.jaw_pose
            out["pose_eye"] = eye_pose

        if not return_dict:
            out = Struct(**out)

        return out
