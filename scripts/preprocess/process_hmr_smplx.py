import numpy as np
import os
import glob
import shutil
from collections import defaultdict
from tqdm import tqdm

import torch
from smplfitter.pt import BodyModel, BodyConverter

dataset_file_dict = {
    "coco": "datasets/mask_transformer/coco/coco.npz",
    "h36m": "datasets/mask_transformer/h36m_train/h36m_train.npz",
    "mpi_inf_3dhp": "datasets/mask_transformer/mpi-inf-3dhp/mpi_inf_3dhp_train.npz",
    "mpii": "datasets/mask_transformer/mpii/mpii.npz",
}
os.environ["DATA_ROOT"] = "body_models/smplfitter"

if __name__ == "__main__":
    datasets = list(dataset_file_dict.keys())
    for dataset in datasets:
        dataset_file = dataset_file_dict[dataset]
        file_name = os.path.basename(dataset_file).split(".")[0]
        save_dir = os.path.dirname(dataset_file)
        save_path = os.path.join(save_dir, file_name + "_smplx.npz")

        data = np.load(dataset_file, allow_pickle=True)

        bm_in = BodyModel('smpl', 'neutral')
        bm_out = BodyModel('smplx', 'neutral')
        smpl2smplx = BodyConverter(bm_in, bm_out).cuda()
        smpl2smplx = torch.jit.script(smpl2smplx)

        pose_rotvecs_in = torch.from_numpy(data['pose']).float().cuda()
        shape_betas_in = torch.from_numpy(data['shape']).float().cuda()
        has_trans = 'global_t' in data 
        if has_trans:
            trans_in = torch.from_numpy(data['global_t']).float().cuda()
        else:
            trans_in = torch.zeros((pose_rotvecs_in.shape[0], 3), dtype=torch.float32).cuda()

        batch_size = 512
        num_samples = pose_rotvecs_in.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        pose_rotvecs_out_list = []
        shape_betas_out_list = []
        trans_out_list = []

        for i in tqdm(range(num_batches), desc=f"Processing {dataset}"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            pose_batch = pose_rotvecs_in[start_idx:end_idx]
            shape_batch = shape_betas_in[start_idx:end_idx]
            trans_batch = trans_in[start_idx:end_idx]
            
            out_batch = smpl2smplx.convert(pose_batch, shape_batch, trans_batch)
            
            pose_rotvecs_out = out_batch['pose_rotvecs'].cpu()
            shape_betas_out = out_batch['shape_betas'].cpu()
            trans_out = out_batch['trans'].cpu()

            pose_rotvecs_out_list.append(pose_rotvecs_out)
            shape_betas_out_list.append(shape_betas_out)
            trans_out_list.append(trans_out)

        pose_rotvecs_out = torch.cat(pose_rotvecs_out_list, dim=0)
        shape_betas_out = torch.cat(shape_betas_out_list, dim=0)
        trans_out = torch.cat(trans_out_list, dim=0)

        save_data = {}
        for key in data.files:
            if key not in ['pose', 'shape', 'global_t']:
                save_data[key] = data[key]
        save_data['pose'] = pose_rotvecs_out.numpy()
        save_data['shape'] = shape_betas_out.numpy()
        if has_trans:
            save_data['global_t'] = trans_out.numpy()

        np.savez(save_path, **save_data)
