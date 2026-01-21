import os
from os import path as osp
from tqdm import trange
import numpy as np
from collections import defaultdict

import glob

amass_data_path = "/home/zhqian/data/motion_datasets/AMASS/AMASS_smplx_neutral"

amass_train = ['CMU', 'KIT', 'BMLrub', 'CNRS', 'DFaust', 'Eyes_Japan_Dataset', 'BMLmovi', 'TotalCapture', 'EKUT', 'ACCAD', 'TCDHands', 'PosePrior', 'SOMA', 'WEIZMANN']
amass_val = ['HumanEva', 'HDM05', 'SFU', 'MoSh']
amass_test = ['Transitions', 'SSM']

def get_all_paths():
    paths = defaultdict(list)

    for motion_file in glob.glob(osp.join(amass_data_path, '*', '*', '*.npz')):
        dataset_name = motion_file.split('/')[-3]
        paths[f'{dataset_name}'].append(motion_file)

    return paths

def preprocess(save_path, split="train", skip=10):
    paths = get_all_paths()
    trim_rate = 0.2
    skip_rate = 10

    pose_body = np.empty((0,63))
    root_orient = np.empty((0,3))
    betas = np.empty((0,10))
    gender = np.empty((0), dtype=str)
    name = np.empty((0), dtype=str) 

    dt_npz = osp.join(save_path, f'{split}.npz')

    for dt in eval(f'amass_{split}'):
        os.makedirs(save_path, exist_ok=True)

        for n_seq in trange(len(paths[dt])):
            seq_name = paths[dt][n_seq].split(dt)[-1]

            try:
                amass_seq = np.load(paths[dt][n_seq])
                N = amass_seq['poses'].shape[0]
                keep_idx = np.array(range(int(trim_rate * N), int((1-trim_rate) * N), skip_rate))
                data = {}
                data["pose_body"] = amass_seq["poses"][keep_idx, 3:66]
                data["root_orient"] = amass_seq["poses"][keep_idx, :3]
                data["betas"] = np.repeat(amass_seq["betas"][None, :10], keep_idx.shape[0], axis=0)
                data["gender"] = amass_seq["gender"]
            except:
                continue

            pose_body = np.append(pose_body, data['pose_body'], axis=0)
            root_orient = np.append(root_orient, data['root_orient'], axis=0)
            betas = np.append(betas, data['betas'], axis=0)
            gender = np.append(gender, [data['gender']]*keep_idx.shape[0])
            name = np.append(name, [f'{dt}{seq_name}']*keep_idx.shape[0])
            # print(f'{dt} | {n_seq} | number of frames in {seq_name} -->  {keep_idx.shape}')
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
        save_path="datasets/mesh_vq_vae/AMASS_smplx/",
        split="train",
        skip=10,
    )
    preprocess(
        save_path="datasets/mesh_vq_vae/AMASS_smplx/",
        split="val",
        skip=10,
    )
    preprocess(
        save_path="datasets/mesh_vq_vae/AMASS_smplx/",
        split="test",
        skip=10,
    )