import torch
import einops

from .vit import vit

class Backbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = vit(use_checkpoint=self.cfg.get("use_checkpoint", False))

        pretrained_ckpt_path = "ckpt/backbones/vit_pose_hmr2.pth"
        sd = torch.load(pretrained_ckpt_path)
        self.model.load_state_dict(sd)

        self.load_image_pretrain()

        if cfg.get("freeze", False):
            self.eval()
            self.requires_grad_(False)

    def load_image_pretrain(self):
        image_ckpt = self.cfg.get("image_ckpt", None)
        if image_ckpt is not None:
            ckpt = torch.load(image_ckpt)
            model_dict = ckpt["state_dict"]
            load_dict = {k.replace("backbone.", ""): v for k, v in model_dict.items() if k.startswith("backbone.")}
            self.load_state_dict(load_dict, strict=False)

    def forward(self, batch, img_chunk_size=8):
        img = batch["crop_imgs"]
        B, F = img.shape[:2]
        img = einops.rearrange(img, "b f c h w -> (b f) c h w")

        # Process ViT in small chunks to avoid OOM
        chunks = []
        for i in range(0, img.shape[0], img_chunk_size):
            chunk = img[i:i + img_chunk_size]
            chunks.append(self.model(chunk))
        img_feat = torch.cat(chunks, dim=0)

        img_feat = einops.rearrange(img_feat, "(b f) c h w -> b f (h w) c", b=B, f=F)
        return img_feat