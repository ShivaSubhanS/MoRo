import os
from os import path as osp
from tqdm import trange
import numpy as np

import glob

def preprocess(folder, save_path, skip=10):
    split = "train"

    paths = glob.glob(osp.join(folder, "*", "*", "*.npz"))

    trim_rate = 0.2

    os.makedirs(save_path, exist_ok=True)
    dt_npz = osp.join(save_path, f'{split}.npz')
    pose_body = np.empty((0,63))
    root_orient = np.empty((0,3))
    betas = np.empty((0,10))
    gender = np.empty((0), dtype=str)
    name = np.empty((0), dtype=str) 

    for n_seq in trange(len(paths)):
        path = paths[n_seq]
        seq_name = "-".join(path.split("/")[-3:-1])
        seq_data = np.load(path)
        
        N = seq_data['poses'].shape[0]
        keep_idx = np.array(range(int(trim_rate * N), int((1-trim_rate) * N), skip))
        
        data = {}
        data["pose_body"] = seq_data["poses"][keep_idx, 3:66]
        data["root_orient"] = seq_data["poses"][keep_idx, :3]
        data["betas"] = np.repeat(seq_data["betas"][None, :10], keep_idx.shape[0], axis=0)
        data["gender"] = seq_data["gender"]

        pose_body = np.append(pose_body, data['pose_body'], axis=0)
        root_orient = np.append(root_orient, data['root_orient'], axis=0)
        betas = np.append(betas, data['betas'], axis=0)
        gender = np.append(gender, [data['gender']]*keep_idx.shape[0])
        name = np.append(name, [f'{seq_name}']*keep_idx.shape[0])
        # print(f'{n_seq} | number of frames in {seq_name} -->  {keep_idx.shape}')
    np.savez(dt_npz,
            pose_body=pose_body,
            root_orient=root_orient,
            betas=betas,
            gender=gender,
            name=name,
            )
    print(f'Total pose samples in split {split}--> {pose_body.shape}')

if __name__ == "__main__":
    preprocess(
        "/home/zhqian/data/motion_datasets/BEDLAM/bodies/neutral_ground_truth_motioninfo",
        save_path="datasets/mesh_vq_vae/bedlam_animations/",
        skip=10,
    )