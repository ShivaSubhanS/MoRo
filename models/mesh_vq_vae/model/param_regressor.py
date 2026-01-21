"""
Implements the SMPLX parameter regressor.
"""

import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix

import numpy as np

from .resnet import Resnet1D


class ParamRegressor(nn.Module):
    def __init__(self, cfg):
        """Initialize a SMPLX parameter regressor.

        Args:
            cfg (DictConfig): Config dictionary.
        """
        super(ParamRegressor, self).__init__()

        self.d_in = cfg.d_in
        self.d_out = cfg.d_out
        self.width = cfg.width
        self.depth = cfg.depth
        self.activation = nn.LeakyReLU()

        dims = [self.d_in] + [self.width] * self.depth + [self.d_out]

        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.LayerNorm(dims[i]))
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != self.num_layers - 1:
                self.layers.append(self.activation)

    def forward(self, x):
        """Forward pass of the regressor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.view(-1, self.d_in)
        for layer in self.layers:
            x = layer(x)

        pred_pose = x[:, :-10]
        pred_beta = x[:, -10:]

        pred_pose_6d = pred_pose.reshape(-1, 21, 6)
        pred_pose_rotmat = rotation_6d_to_matrix(pred_pose_6d)

        output = {
            "pred_pose_body_6d": pred_pose_6d,
            "pred_pose_body_rotmat": pred_pose_rotmat,
            "pred_beta": pred_beta,
        }
        return output


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(self.dims)
        return x


class PoseSPDecoderV1(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(PoseSPDecoderV1, self).__init__()
        self.cfg = cfg

        num_tokens = cfg.num_tokens
        code_dim = cfg.code_dim

        down_t = cfg.down_t
        width = cfg.width
        depth = cfg.depth
        token_size_div = cfg.token_size_div
        dilation_rate = cfg.dilation_rate

        num_joints = cfg.num_joints
        output_dim = cfg.output_dim

        decoder_layers = []
        self.num_joints = num_joints

        decoder_layers.append(Permute((0, 2, 1)))
        decoder_layers.append(
            nn.Conv1d(code_dim, width, 3, 1, 1, padding_mode="replicate")
        )
        decoder_layers.append(nn.ReLU())

        decoder_layers.append(Permute((0, 2, 1)))
        decoder_layers.append(nn.Conv1d(num_tokens, num_tokens, 1))
        decoder_layers.append(Permute((0, 2, 1)))
        decoder_layers.append(nn.ReLU())

        print(f"Num of tokens --> {num_tokens}")
        for i in list(
            np.linspace(
                self.num_joints, num_tokens, token_size_div, endpoint=False, dtype=int
            )[::-1]
        ):
            decoder_layers.append(nn.Upsample(i, mode="linear"))
            decoder_layers.append(
                nn.Conv1d(width, width, 3, 1, 1, padding_mode="replicate")
            )
            decoder_layers.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_rate,
                    reverse_dilation=True,
                    activation="relu",
                    norm=False,
                ),
                nn.Conv1d(width, out_dim, 3, 1, 1, padding_mode="replicate"),
            )
            decoder_layers.append(block)

        self.decoder = nn.Sequential(*decoder_layers)

        # self.pose_regressor = nn.Sequential(
        #     Permute((0, 2, 1)),
        #     nn.LayerNorm(width),
        #     nn.Linear(width, output_dim),
        # )
        self.pose_regressor = nn.Sequential(
            nn.Conv1d(width, output_dim, 3, 1, 1, padding_mode="replicate"),
            Permute((0, 2, 1)),
        )
        self.beta_regressor = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 10),
        )
        self.beta_regressor[1].bias.data.fill_(0.0)

    def forward(self, x):
        output = {}

        x = self.decoder(x)

        pred_pose = self.pose_regressor(x)

        pred_beta = self.beta_regressor(x.mean(dim=-1))

        pred_pose_6d = pred_pose
        pred_pose_rotmat = rotation_6d_to_matrix(pred_pose_6d)

        output.update(
            {
                "pred_pose_body_6d": pred_pose_6d,
                "pred_pose_body_rotmat": pred_pose_rotmat,
                "pred_beta": pred_beta,
            }
        )

        return output

class KinematicTreePoseDecoder(nn.Module):
    def __init__(self, num_joints=21, width=512, output_dim=6, multi_reg=True):
        super().__init__()

        self.num_joints = num_joints
        self.ktree_parents = np.array([0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19,], dtype=np.int32)

        if multi_reg:
            layers = []
            for idx in range(num_joints):
                layer = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, output_dim))
                layers.append(layer)
        else:
            layers = [nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, output_dim))] * num_joints

        self.layers = nn.ModuleList(layers)

    def forward(self, joint_feats):
        out = [None] * self.num_joints
        for j_idx in range(self.num_joints):
            parent_feat = joint_feats[..., self.ktree_parents[j_idx]]
            curr_feat = joint_feats[..., j_idx + 1]
            in_feat = torch.cat([curr_feat, parent_feat], dim=-1)
            out[j_idx] = self.layers[j_idx](in_feat)

        out = torch.stack(out, dim=1)
        return out

class PoseSPDecoderKT(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(PoseSPDecoderKT, self).__init__()
        self.cfg = cfg

        num_tokens = cfg.num_tokens
        code_dim = cfg.code_dim

        width = cfg.width
        token_size_div = cfg.token_size_div

        num_joints = cfg.num_joints
        output_dim = cfg.output_dim

        decoder_layers = []
        self.num_joints = num_joints
        self._num_joints = num_joints + 1 

        decoder_layers.append(Permute((0, 2, 1)))
        decoder_layers.append(
            nn.Conv1d(code_dim, width, 3, 1, 1, padding_mode="replicate")
        )
        decoder_layers.append(nn.ReLU())

        decoder_layers.append(Permute((0, 2, 1)))
        decoder_layers.append(nn.Conv1d(num_tokens, num_tokens, 1))
        decoder_layers.append(Permute((0, 2, 1)))
        decoder_layers.append(nn.ReLU())

        print(f"Num of tokens --> {num_tokens}")
        for i in list(
            np.linspace(
                self._num_joints, num_tokens, token_size_div, endpoint=False, dtype=int
            )[::-1]
        ):
            decoder_layers.append(nn.Upsample(i, mode="linear"))
            decoder_layers.append(
                nn.Conv1d(width, width, 3, 1, 1, padding_mode="replicate")
            )
            if i != self._num_joints:
                decoder_layers.append(nn.ReLU())

        # for i in range(down_t):
        #     out_dim = width
        #     block = nn.Sequential(
        #         Resnet1D(
        #             width,
        #             depth,
        #             dilation_rate,
        #             reverse_dilation=True,
        #             activation="relu",
        #             norm=False,
        #         ),
        #         nn.Conv1d(width, out_dim, 3, 1, 1, padding_mode="replicate"),
        #     )
        #     decoder_layers.append(block)

        self.decoder = nn.Sequential(*decoder_layers)
        self.beta_regressor = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 10),
        )
        self.beta_regressor[1].bias.data.fill_(0.0)

        self.pose_regressor = KinematicTreePoseDecoder(num_joints=num_joints, width=width, output_dim=output_dim, multi_reg=cfg.multi_reg)
        # self.pose_regressor = nn.Sequential(
        #     nn.Conv1d(width, output_dim, 3, 1, 1, padding_mode="replicate"),
        #     Permute((0, 2, 1)),
        # )

    def forward(self, x):
        output = {}

        x = self.decoder(x)
        pred_beta = self.beta_regressor(x.mean(dim=-1))

        pred_pose = self.pose_regressor(x)

        pred_pose_6d = pred_pose
        pred_pose_rotmat = rotation_6d_to_matrix(pred_pose_6d)

        output.update(
            {
                "pred_pose_body_6d": pred_pose_6d,
                "pred_pose_body_rotmat": pred_pose_rotmat,
                "pred_beta": pred_beta,
            }
        )

        return output
