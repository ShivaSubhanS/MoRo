# Masked Modeling for Human Motion Recovery Under Occlusions
### [Project Page](https://mikeqzy.github.io/MoRo) | [Paper](https://arxiv.org/)
> Masked Modeling for Human Motion Recovery Under Occlusions  
> [Zhiyin Qian](https://mikeqzy.github.io/),
> [Siwei Zhang](https://sanweiliti.github.io/),
> [Bharat Lal Bhatnagar](https://virtualhumans.mpi-inf.mpg.de/people/Bhatnagar.html),
> [Federica Bogo](https://fbogo.github.io/),
> [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html)  
> 3DV 2026

<p align="center">
    <video src="assets/intro.mp4" controls autoplay></video>
</p>

## Installation

```Bash
git clone https://github.com/mikeqzy/MoRo
conda env create -f environment.yml
conda activate moro
```

## Data preparation

### SMPL(-X) body model
We use [smplfitter](https://github.com/isarandi/smplfitter) to fit the non-parametric mesh to SMPL-X parameters. Please follow their provided [script](https://github.com/isarandi/posepile/blob/main/src/posepile/get_body_models.sh) to download these files and put them under `body_models`.

Additionally, you can download the mesh connection matrices for SMPL-X topology used for the fully convolutional mesh autoencoder in Mesh-VQ-VAE and other regressors for evaluation [TODO:here](). Please also put them under `body_models` .

### Tokenization
We train the tokenizer on [AMASS](https://amass.is.tue.mpg.de/), [MOYO](https://moyo.is.tue.mpg.de/) and [BEDLAM](https://bedlam.is.tuebingen.mpg.de/). Download the SMPL-X neutral annotations from their official project pages and unzip the files.

We preprocessed the datasets with the scripts at `models/mesh_vq_vae/data/preprocess`. Please change the dataset paths accordingly.

### MoRo

We train MoRo on a mixture of datasets that are prepared as follows:

* **AMASS**: 
  
  We preprocess AMASS into 30 fps sequences following RoHM, please refer to the instructions [here](https://github.com/sanweiliti/RoHM?tab=readme-ov-file#amass).
* **MPII, Human3.6M, MPI-INF-3DHP, COCO**: 
    
  The training dataset uses the SMPL annotations from BEDLAM. Follow the instructions [here](https://github.com/pixelite1201/BEDLAM/blob/master/docs/training.md) in the section `Training CLIFF model with real images` to obtain the required training images and annotations.
  
  We further convert the SMPL annotations to SMPL-X using the provided script at `scripts/preprocess/process_hmr_smplx.py`.
* **BEDLAM**: 

  Download the BEDLAM dataset from their official [project page](https://bedlam.is.tue.mpg.de/index.html). We use the SMPL-X neutral annotations for training.
* **EgoBody**:
  
  First, download the EgoBody dataset from the official [EgoBody dataset](https://sanweiliti.github.io/egobody/egobody.html).
  
  Additionally, download `keypoints_cleaned`, `mask_joint` and `egobody_occ_info.csv` from [TODO:here]() and place them under the dataset directory.
  
  Finally, run the provided preprocessing script at `scripts/preprocess/process_egobody_bbox.py` to generate the bounding box files.

### Testing
Additionally, we test our method on [PROX](https://prox.is.tue.mpg.de/) and [RICH](https://rich.is.tue.mpg.de/).

* **RICH**:

  Download the RICH dataset from their official [project page](https://rich.is.tue.mpg.de). The preprocessed annotations can be downloaded from [GVHMR](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md) repo. Put the `hmr4d_support` folder under the RICH dataset directory.

* **PROX**:
  
  First, download the PROX dataset from their official [project page](https://prox.is.tue.mpg.de/). 

  Additionally, download `keypoints_openpose` and `mask_joint` from [TODO:here]() and place them under the dataset directory.

  Finally, run the provided preprocessing script at `scripts/preprocess/process_prox_bbox.py` to generate the bounding box files.


### Checkpoints

For the ViT backbone, we use the pretrained weights from 4DHumans (download model from the `Training` section of official [repo](https://github.com/shubham-goel/4D-Humans/tree/main)). Place it at `ckpt/backbones/vit_pose_hmr2.pth`.

<!-- For custom videos, we leverage [SPEC](https://github.com/mkocabas/SPEC) to estimate the camera focal length. The checkpoint can be downloaded [here](https://drive.google.com/file/d/1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO/view?usp=sharing) and placed under `ckpt`.  -->

The checkpoint for Mesh-VQ-VAE and MoRo can be downloaded [TODO:here](). Place the tokenizer checkpoint at `ckpt/tokenizer/tokenizer.ckpt` and MoRo checkpoint at `exp/mask_transformer/MIMO-vit-release/video_train/checkpoints/last.ckpt`.

### Structure
The data should be organized as follows:
```
MoRo
├── body_models
│   ├── smplx
│   ├── smplx_ConnectionMatrices
├── ckpt
│   ├── backbones
│   │   ├── vit_pose_hmr2.pth
│   ├── tokenizer
│   │   ├── tokenizer.ckpt
├── exp
│   ├── mask_transformer
│   │   ├── MIMO-vit-release
│   │   │   ├── video_train
│   │   │   │   ├── checkpoints
│   │   │   │   │   ├── last.ckpt
├── datasets
│   ├── mesh_vq_vae
│   │   ├── bedlam_animations
│   │   ├── AMASS_smplx
│   │   ├── MOYO
│   ├── mask_transformer
│   │   ├── AMASS
│   │   ├── BEDLAM
│   │   ├── coco
│   │   ├── h36m_train
│   │   ├── mpi-inf-3dhp
│   │   ├── mpii
│   │   ├── EgoBody
│   │   ├── PROX
│   │   ├── rich
```

<!-- ## Demo
For a quick demo on custom video taken from static camera, run the following command on a 30 fps video:
```Bash
python demo.py demo.video_path=/path/to/demo.mp4
```
or an image directory with sorted frames:
```Bash
python demo.py demo.video_path=/path/to/image_dir
```
By default, the rendering result will be saved to `./exp/mask_transformer/MIMO-vit-release/video_train`, under the same directory of the released model checkpoint. -->

## Training

### Tokenization
You can train the mesh tokenizer by running:
```Bash
python train_mesh_vqvae.py
```
The configuration file is at `configs/mesh_vq_vae/config.yaml`.

### MoRo
We adopt a multi-stage training strategy for MoRo:
```Bash
# Stage 1: Pose pretraining
python train_mask_transformer.py option=pose_pretrain tag=default
# Stage 2: Motion pretraining
python train_mask_transformer.py option=motion_pretrain tag=default
# Stage 3: Image pretraining on image datasets
python train_mask_transformer.py option=image_pretrain tag=default
# Stage 4: Image pretraining on video datasets
python train_mask_transformer.py option=video_pretrain tag=default
# Stage 5: Finetuning on video datasets
python train_mask_transformer.py option=video_train tag=default
```
The configuration files are at `configs/mask_transformer/config.yaml`, the specific options for each training stage can be found in `configs/option`.

The training logs and checkpoints will be saved under `exp/mask_transformer/MIMO-vit-<tag>/<stage>`.

## Testing and Evaluation
We test on the trained MoRo model and save the results to corresponding `exp/mask_transformer/MIMO-vit-<tag>/video_train` directory. Then we evaluate the results with the provided scripts.

We set `tag=release` here to reproduce the results reported on the paper.

### EgoBody
```Bash
python train_mask_transformer.py option=[inference,video_train] tag=release data=egobody
python eval_egobody.py --saved_data_dir=./exp/mask_transformer/MIMO-vit-release/video_train/result_egobody/inference_5_1 --recording_name=all --render
```

### RICH
```Bash
python train_mask_transformer.py option=[inference,video_train] tag=release data=rich
python eval_rich.py --saved_data_dir=./exp/mask_transformer/MIMO-vit-release/video_train/result_rich/inference_5_1 --seq_name=all --render
```

### PROX
```Bash
python train_mask_transformer.py option=[inference,video_train] tag=release data=prox
python eval_egobody.py --saved_data_dir=./exp/mask_transformer/MIMO-vit-release/video_train/result_prox/inference_5_1 --recording_name=all --render
```

## Acknowledgements
This work was supported as part of the Swiss AI initiative by a grant from the Swiss National Supercomputing Centre (CSCS) under project IDs \#36 on Alps, enabling large-scale training.

Some code in this repository is adapted from the following repositories:
* [RoHM](https://github.com/sanweiliti/RoHM)
* [MoMask](https://github.com/EricGuo5513/momask-codes)
* [MotionBERT](https://github.com/Walter0807/MotionBERT)
* [VQ-HPS](https://github.com/g-fiche/VQ-HPS)
* [TokenHMR](https://github.com/saidwivedi/TokenHMR)
* [GVHMR](https://github.com/zju3dv/GVHMR)
* [BEDLAM](https://github.com/pixelite1201/BEDLAM)

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{qian2026moro,
  title={Masked Modeling for Human Motion Recovery Under Occlusions},
  author={Qian, Zhiyin and Zhang, Siwei and Bhatnagar, Bharat Lal and Bogo, Federica and Tang, Siyu},
  booktitle={3DV},
  year={2026}
}
```