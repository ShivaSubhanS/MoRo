import os
from os import path as osp
from tqdm import trange
import numpy as np

import glob


def preprocess(folder, save_path, split="train", skip=5):
    paths = glob.glob(osp.join(folder, split, "*", "*", "*.npz"))

    trim_rate = 0.1

    os.makedirs(save_path, exist_ok=True)
    dt_npz = osp.join(save_path, f'{split}.npz')
    pose_body = np.empty((0,63))
    root_orient = np.empty((0,3))
    betas = np.empty((0,10))
    gender = np.empty((0), dtype=str)
    name = np.empty((0), dtype=str) 
    idx = np.empty((0,1), dtype=int)

    for n_seq in trange(len(paths)):
        seq_name = os.path.basename(paths[n_seq])
        moyo_seq = np.load(paths[n_seq])
        
        N = moyo_seq['pose_body'].shape[0]
        keep_idx = np.array(range(int(trim_rate * N), int((1-trim_rate) * N), skip))
        
        data = {}
        data["pose_body"] = moyo_seq["pose_body"][keep_idx]
        data["root_orient"] = moyo_seq["root_orient"][keep_idx]
        data["betas"] = np.repeat(moyo_seq["betas"][None, :10], keep_idx.shape[0], axis=0)
        data["gender"] = moyo_seq["gender"]

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

    # root_orient_list = []
    # pose_body_list = []
    # betas_list = []
    # gender_list = []
    # for npy_file in tqdm(files):
    #     data = np.load(npy_file, allow_pickle=True)
    #     root_orient = data[:, :3]
    #     pose_body = data[:, 16:79]
    #     betas = data[:, 6:16]
    #     gender = np.array(['neutral'] * data.shape[0])

    #     root_orient_list.append(root_orient)
    #     pose_body_list.append(pose_body)
    #     betas_list.append(betas)
    #     gender_list.append(gender)

    # root_orient_list = np.concatenate(root_orient_list, axis=0)
    # pose_body_list = np.concatenate(pose_body_list, axis=0)
    # betas_list = np.concatenate(betas_list, axis=0)
    # gender_list = np.concatenate(gender_list, axis=0)

    # np.savez(
    #     save_path,
    #     root_orient=root_orient_list,
    #     pose_body=pose_body_list,
    #     betas=betas_list,
    #     gender=gender_list,
    # )

if __name__ == "__main__":
    preprocess(
        "/home/zhqian/data/motion_datasets/MOYO_smplx_neutral",
        save_path="datasets/mesh_vq_vae/MOYO/",
        split="train",
        skip=5,
    )
    preprocess(
        "/home/zhqian/data/motion_datasets/MOYO_smplx_neutral",
        save_path="datasets/mesh_vq_vae/MOYO/",
        split="val",
        skip=5,
    )
    preprocess(
        "/home/zhqian/data/motion_datasets/MOYO_smplx_neutral",
        save_path="datasets/mesh_vq_vae/MOYO/",
        split="test",
        skip=5,
    )