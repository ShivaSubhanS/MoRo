import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class TemporalSmoother(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dim_in = self.dim_out = cfg.dim_in
        self.dim_hidden = cfg.dim_hidden

        self.num_tokens = cfg.num_tokens
        self.share_weights = cfg.share_weights
        
        if not self.share_weights:
            self.dim_in = self.dim_out = cfg.dim_in * self.num_tokens

        kernel_size = cfg.kernel_size
        num_layers = cfg.num_layers

        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=self.dim_in if i == 0 else self.dim_hidden,
                    out_channels=self.dim_hidden,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                    padding_mode="replicate",
                )
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Conv1d(self.dim_hidden, self.dim_out, kernel_size=1, bias=False)  # project back
        )
        # make it small for initialization
        # nn.init.normal_(layers[-1].weight, mean=0.0, std=1e-3)
        self.net = nn.Sequential(*layers)

        # gating
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.load_image_pretrain()

    def load_image_pretrain(self):
        motion_ckpt = self.cfg.get("motion_ckpt", None)
        image_ckpt = self.cfg.get("image_ckpt", None)
        if image_ckpt is not None:
            ckpt = torch.load(image_ckpt)
            model_dict = ckpt["state_dict"]
            load_dict = {k.replace("smoother.", ""): v for k, v in model_dict.items() if k.startswith("smoother.")}
            self.load_state_dict(load_dict, strict=False)
        elif motion_ckpt is not None:
            ckpt = torch.load(motion_ckpt)
            model_dict = ckpt["state_dict"]
            load_dict = {k.replace("smoother.", ""): v for k, v in model_dict.items() if k.startswith("smoother.")}
            self.load_state_dict(load_dict, strict=False)

    def forward(self, x):
        # x: [B, F, J, C]
        B, F, J, C = x.shape
        if self.share_weights:
            out = einops.rearrange(x, "b f j c -> (b j) c f")
            out = self.net(out)  # → [B*J, 9, F]
            out = einops.rearrange(out, "(b j) c f -> b f j c", b=B)
        else:
            out = einops.rearrange(x, "b f j c -> b (j c) f")
            out = self.net(out)
            out = einops.rearrange(out, "b (j c) f -> b f j c", j=J)
        return x + self.alpha * out

