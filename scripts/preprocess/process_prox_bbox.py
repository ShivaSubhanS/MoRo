import numpy as np
import os
import json
from glob import glob
from tqdm import tqdm

root = "../../datasets/mask_transformer/PROX/keypoints_openpose"
kps_paths = glob(os.path.join(root, "*"))

def process_egobody_bbox(kps_path):
    scale_factor_bbox = 1.2 
    kps_files = sorted(glob(os.path.join(kps_path, "*keypoints.json")))
    last_bbox = np.array([100., 100., 200., 200.], dtype=np.float32) # dummy bbox
    center_dict = {}
    scale_dict = {}
    bbox_dict = {} 
    for kps_file in kps_files:
        frame_name = os.path.splitext(os.path.basename(kps_file))[0][:-10] # remove "_keypoints"

        with open(kps_file, 'r') as f:
            kps_data = json.load(f)
    
        if len(kps_data["people"]) == 0:
            bbox = last_bbox
        else:
            body_kps = np.array(kps_data["people"][0]["pose_keypoints_2d"]).astype(np.float32).reshape((-1, 3))
            valid = body_kps[:, 2] > 0.2
            valid_kps = body_kps[valid, :-1]

            if valid_kps.shape[0] < 2:
                bbox = last_bbox
            else:
                x0, y0 = np.min(valid_kps, axis=0)
                x1, y1 = np.max(valid_kps, axis=0)
                bbox = np.array([x0, y0, x1, y1])
                
        last_bbox = bbox
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        scale = max(
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
        ) * scale_factor_bbox / 200
        center_dict[frame_name] = center
        scale_dict[frame_name] = scale
        bbox_dict[frame_name] = bbox 

    # save the center and scale as numpy arrays
    frame_names = list(center_dict.keys())
    centers = np.array([center_dict[name] for name in frame_names])
    scales = np.array([scale_dict[name] for name in frame_names])
    bboxes = np.array([bbox_dict[name] for name in frame_names])
    
    save_path = os.path.join(kps_path, f"bbox.npz")
    np.savez(save_path, frame_names=frame_names, centers=centers, scales=scales, bboxes=bboxes)

# for debugging
# process_egobody_bbox(kps_paths[0])

from joblib import Parallel, delayed

Parallel(n_jobs=64)(
    delayed(process_egobody_bbox)(kps_path) 
    for kps_path in tqdm(kps_paths)
)
