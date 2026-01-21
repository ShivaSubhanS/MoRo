import torch
from torch.utils import data
import numpy as np
import cv2
import os

from lightning import LightningDataModule

from ..utils import crop

class DataloaderDemo(data.Dataset):
    def __init__(self, cfg, image_paths, bbx_xys, K):
        """
        Args:
            cfg: configuration object
            image_paths: list of image file paths
            bbx_xys: bounding boxes [N, L, 3] where N is number of tracks, 
                    L is number of frames, 3 is (cx, cy, scale)
            K: camera intrinsics matrix [3, 3]
        """
        self.cfg = cfg
        self.image_paths = image_paths
        self.bbx_xys = bbx_xys  # [N, L, 3]
        self.K = torch.from_numpy(K).float()  # [3, 3] 
        
        self.num_tracks = bbx_xys.shape[0]
        self.num_frames = len(image_paths)
        
        self.clip_len = cfg.clip_len
        self.clip_overlap_len = cfg.overlap_len
        
        # Initialize data structures for each track
        self.track_clips = []  # List of (track_idx, start_frame, end_frame)
        
        # Divide each track into clips with overlapping window
        for track_idx in range(self.num_tracks):
            seq_idx = 0
            while True:
                start = seq_idx * (self.clip_len - self.clip_overlap_len)
                end = start + self.clip_len
                
                actual_end = min(end, self.num_frames)
                self.track_clips.append((track_idx, start, actual_end))
                
                seq_idx += 1
                if end >= self.num_frames:
                    break
        
        self.n_samples = len(self.track_clips)
        print(f"[INFO] Demo dataset: {self.num_tracks} tracks, {self.num_frames} frames, {self.n_samples} clips total")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        track_idx, start_frame, end_frame = self.track_clips[index]
        num_frames = end_frame - start_frame
        
        # Get bounding boxes for this clip
        bbx_clip = self.bbx_xys[track_idx, start_frame:end_frame]  # [F, 3]
        
        # Process images
        crop_img_list = []
        center_list = []
        scale_list = []
        img_path_list = []
        
        res = self.cfg.get("crop_res", (224, 224))
        
        for i in range(num_frames):
            frame_idx = start_frame + i
            img_path = self.image_paths[frame_idx]
            img_path_list.append(img_path)
            
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get bounding box
            cx, cy, scale = bbx_clip[i].cpu().numpy()
            center = np.array([cx, cy], dtype=np.float32)
            
            # Crop image
            img, crop_img = crop(img, center, scale, res=res)
            
            center_list.append(center)
            scale_list.append(scale)
            crop_img_list.append(crop_img)
        
        # Stack and convert
        center_list = np.stack(center_list).astype(np.float32)
        scale_list = np.array(scale_list).astype(np.float32)
        center_list = torch.from_numpy(center_list).float()
        scale_list = torch.from_numpy(scale_list).float()
        
        # Process cropped images
        crop_img_list = np.stack(crop_img_list)
        crop_img_list = crop_img_list.transpose((0, 3, 1, 2))
        crop_img_list = crop_img_list.astype(np.float32)
        crop_img_list /= 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        crop_img_list = (crop_img_list - mean) / std
        crop_img_list = torch.from_numpy(crop_img_list).float()
        
        # Use K from initialization
        focal = self.K[0, 0]  # Extract focal length from K
        cam_center = torch.zeros_like(center_list)
        cam_center[:, 0] = self.K[0, 2]
        cam_center[:, 1] = self.K[1, 2]
        
        bbox = (
            torch.cat(
                [center_list - cam_center, scale_list.unsqueeze(-1) * 200.0], dim=1
            )
            / focal
        )
        
        # Prepare batch dictionary
        batch = {
            # Basic info
            "body_idx": str(track_idx),
            "seq_name": f"id{track_idx}",
            "center": center_list,
            "scale": scale_list,
            "crop_imgs": crop_img_list,
            "bbox": bbox,
            "has_transl": True, 
            "true_params": False,
            
            # Camera parameters
            "K": self.K.unsqueeze(0).repeat(num_frames, 1, 1),
            "dist_coeffs": torch.zeros((8)).float(),

            # Images
            "img_paths": img_path_list,
        }
        
        return batch


class DataModuleDemo(LightningDataModule):
    def __init__(self, cfg, image_paths, bbx_xys, K):
        """
        Args:
            cfg: configuration object
            image_paths: list of image file paths
            bbx_xys: bounding boxes tensor [N, L, 3]
            K: camera intrinsics matrix [3, 3]
        """
        super().__init__()
        self.cfg = cfg
        self.image_paths = image_paths
        self.bbx_xys = bbx_xys
        self.K = K

    def setup(self, stage=None):
        if stage in [None, "predict", "test"]:
            self.predict_dataset = DataloaderDemo(
                self.cfg,
                image_paths=self.image_paths,
                bbx_xys=self.bbx_xys,
                K=self.K,
            )

    def predict_dataloader(self):
        return data.DataLoader(
            self.predict_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.get("num_workers_val", 0),
            drop_last=False,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return self.predict_dataloader()
