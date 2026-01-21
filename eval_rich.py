import os
import pickle as pkl
import joblib
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm, trange
from collections import defaultdict
from pytorch3d.transforms import axis_angle_to_matrix

from models.mask_transformer.utils.body_model import BodyModel
from smplfitter.pt import BodyModel as BodyModelFit
from smplfitter.pt import BodyFitter

import argparse
import cv2
import PIL.Image as pil_img

from models.mask_transformer.utils.motion_utils import (
    get_bm_params,
)
from models.mask_transformer.utils.rich_utils import (
    get_cam2params,
    get_w2az_sahmr,
    parse_seqname_info,
    get_cam_key_wham_vid,
    transform_mat,
)
from models.mask_transformer.system.renderer import BatchRenderer
from utils.vis3d_utils import Visualizer3D
from utils.eval_utils import EvalMetrics, EvalResultsManager, fit_smplx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
body_model_path = "body_models"

method_cfg = {
    "MoRo": ("mesh", "smplx"),
    "TokenHMR": ("param", "smplh"),
    "WHAM": ("param", "smpl"),
    "GVHMR": ("param", "smplx"),
    "PromptHMR": ("param", "smplx"),
}

# Method-specific colors for visualization
method_colors = {
    "MoRo": "skin",  # Default color
    "TokenHMR": "purple",
    "WHAM": "orange",
    "GVHMR": "cyan",
    "PromptHMR": "pink",
}

class RichEvaluator:
    """Main evaluator class for RICH dataset."""

    def __init__(self, args):
        self.args = args
        self.body_repr, self.bm_type = method_cfg[args.method]
        self.mesh_color = method_colors.get(args.method, "skin")  # Default to 'skin' if not specified

        # Initialize body models
        # Prediction
        self.bm_neutral = BodyModel(
            bm_path=body_model_path,
            model_type=self.bm_type,
            gender="neutral",
            ext="pkl" if self.bm_type == "smpl" else "npz",
        ).to(device)
        # GT
        self.smplx_male = BodyModel(
            bm_path=body_model_path, model_type="smplx", gender="male"
        ).to(device)
        self.smplx_female = BodyModel(
            bm_path=body_model_path, model_type="smplx", gender="female"
        ).to(device)
        # Render
        self.smpl_neutral = BodyModel(
            bm_path=body_model_path, model_type="smplh", gender="neutral"
        ).to(device) 

        # convert to SMPL
        self.smplx2smpl = (
            torch.load("body_models/smplx2smpl_sparse.pt", weights_only=True)
            .to(device)
            .to_dense()
        )
        self.J_regressor = torch.load(
            "body_models/smpl_neutral_J_regressor.pt", weights_only=True
        ).to(
            device
        )  # use first 22 joints, remove hands

        self.use_smplfitter = (
            self.body_repr == "mesh" and self.bm_type == "smplx" and args.smplfitter
        )
        if self.use_smplfitter:
            os.environ["DATA_ROOT"] = "."

            n_verts = self.bm_neutral.joint_regressor.shape[1]
            J_regressor = (
                self.bm_neutral.bm.J_regressor[:, :n_verts].clone().detach().cpu()
            )

            body_model = BodyModelFit(
                "smplx",
                "neutral",
                num_betas=10,
                vertex_subset=list(range(n_verts)),
                joint_regressor_post_lbs=J_regressor,
            ).to(device)
            self.fitter = BodyFitter(body_model).to(device)

            jid_ignore = [23, 24]  # ignore left and right eyeballs
            joint_weights = torch.ones(1, J_regressor.shape[0], device=device)
            joint_weights[:, jid_ignore] = 0.0
            self.joint_weights = joint_weights

        if args.smplfitter:
            os.environ["DATA_ROOT"] = "."

        # Initialize evaluation utilities
        self.eval_metrics = EvalMetrics(fps=30)
        self.results_manager = EvalResultsManager()
        self.visualizer = (
            Visualizer3D(fps=args.vis_3d_fps, ground_axis='z') if args.visualize_3d else None
        )

        # RICH dataset specific setup
        self.rich_dir = Path(args.dataset_root)
        self.labels = torch.load(self.rich_dir / "hmr4d_support/rich_test_labels.pt")
        self.preproc_data = torch.load(
            self.rich_dir / "hmr4d_support/rich_test_preproc.pt"
        )

        # Load camera and transformation data
        self.w2az = get_w2az_sahmr(self.rich_dir)
        self.cam2params = get_cam2params(self.rich_dir)
        seqname_info = parse_seqname_info(self.rich_dir, skip_multi_persons=True)
        self.seqname_to_scanname = {k: v[0] for k, v in seqname_info.items()}

        # Current sequence data (will be set for each sequence)
        self.vid = None
        self.seq_name = None
        self.cam_name = None
        self.gender = None
        self.frame_name_list = None
        self.K = None
        self.T_w2c = None
        self.T_w2az = None
        self.bbox_data = None  # New field for bounding box data

    def load_sequence_data(self, vid):
        """Load sequence-specific data for RICH dataset."""
        self.vid = vid
        self.seq_name, self.cam_name = vid.split("/")[1:]

        # Get camera and transformation data
        cam_key = get_cam_key_wham_vid(vid)
        scan_name = self.seqname_to_scanname[self.seq_name]
        self.T_w2c, self.K = self.cam2params[cam_key]
        self.T_w2az = self.w2az[scan_name]
        self.T_w2c, self.T_w2az = self.T_w2c.to(device), self.T_w2az.to(device)

        # Get sequence metadata
        label = self.labels[vid]
        self.gender = label["gender"]

        print(f"Loaded sequence: {self.seq_name}, gender: {self.gender}")

    def load_bbox_data(self):
        """Load bounding box data for current sequence."""
        if self.vid in self.preproc_data:
            preproc_data = self.preproc_data[self.vid]
            if "bbx_xys" in preproc_data:
                bbx_xys = preproc_data["bbx_xys"]  # [F, 3] - x, y, size
                
                # Get frame IDs for alignment
                label = self.labels[self.vid]
                frame_ids = label["frame_id"].numpy()
                
                # Get image paths to create frame name mapping
                img_dir = self.rich_dir / "images" / self.vid
                img_paths = sorted(img_dir.glob("*.jpeg"))
                frame_names = [img_paths[frame_id].name for frame_id in frame_ids]
                
                # Create mapping from frame name to bbox info
                self.bbox_data = {}
                for i, frame_name in enumerate(frame_names):
                    if i < len(bbx_xys):
                        center = bbx_xys[i][:2].numpy()  # x, y
                        scale = bbx_xys[i][2].numpy() / 200.0  # normalize scale
                        self.bbox_data[frame_name] = {
                            "center": center,
                            "scale": scale
                        }
                
                print("Bounding box data loaded")
            else:
                print(f"No bbox data found in preproc_data for {self.vid}")
                self.bbox_data = None
        else:
            print(f"No preproc data found for {self.vid}")
            self.bbox_data = None

    def draw_bbox_on_image(self, img, frame_name, downsample_factor=1, crop=False):
        """Draw bounding box on image if bbox data is available."""
        if self.bbox_data is None or frame_name not in self.bbox_data:
            return img
            
        bbox_info = self.bbox_data[frame_name]
        center = bbox_info["center"]
        scale = bbox_info["scale"]

        if crop:
            center[1] -= scale * 50
            scale /= 2.0
        
        # Apply downsampling to bbox coordinates
        center_downsampled = center / downsample_factor
        scale_downsampled = scale / downsample_factor
        
        # Convert center and scale to bbox coordinates
        # Scale is normalized, multiply by 200 to get pixel size (following dataset convention)
        bbox_size = scale_downsampled * 200

        
        # Calculate top-left and bottom-right corners
        x1 = int(center_downsampled[0] - bbox_size / 2)
        y1 = int(center_downsampled[1] - bbox_size / 2)
        x2 = int(center_downsampled[0] + bbox_size / 2)
        y2 = int(center_downsampled[1] + bbox_size / 2)
        
        # Ensure bbox is within image bounds
        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        
        # Draw bounding box (green color, thickness 3)
        img_with_bbox = img.copy()
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        return img_with_bbox

    def load_prediction_data(self):
        """Load prediction data for current sequence."""
        # Extract cam_name from vid
        saved_data_path = os.path.join(
            self.args.saved_data_dir,
            self.seq_name,
            self.cam_name,
            "results.pkl",
        )

        saved_data = joblib.load(saved_data_path)
        self.frame_name_list = saved_data["frame_name"]  # [B, F]
        self.frame_name_list = np.vectorize(lambda x: os.path.basename(x))(
            self.frame_name_list
        )

        if self.body_repr == "param":
            pred_bm_params_cam = {
                "body_pose": saved_data["body_pose"],
                "betas": saved_data["betas"],
                "global_orient": saved_data["global_orient"],
                "transl": saved_data["transl"],
            }
            pred_bm_params_cam = {
                k: torch.from_numpy(v).to(device) for k, v in pred_bm_params_cam.items()
            }
            pred_bm_output_cam = self.bm_neutral(**pred_bm_params_cam)

            # Convert to SMPL for evaluation
            if self.bm_type == "smplx":
                pred_verts_cam = (
                    self.smplx2smpl @ pred_bm_output_cam.full_vertices.clone().detach()
                )
            else:
                pred_verts_cam = pred_bm_output_cam.vertices.clone().detach()
        elif self.body_repr == "mesh":
            pred_verts_cam = torch.from_numpy(saved_data["verts"]).to(device)

            if self.bm_type == "smplx":
                pred_partial_verts_cam = pred_verts_cam
                if self.args.smplfitter:
                    fit_verts_cam = fit_smplx(
                        pred_partial_verts_cam, self, ret_error=False
                    )
                    pred_verts_cam = self.smplx2smpl @ fit_verts_cam
                else:
                    # Use SMPLX vertices directly
                    # Note that this is only approximate, the SMPLX eyeballs are neglected
                    pred_verts_cam = (
                        self.smplx2smpl[:, : pred_partial_verts_cam.shape[-2]]
                        @ pred_partial_verts_cam
                    )

        pred_joints_cam = self.J_regressor @ pred_verts_cam

        # Transform to world coordinates using camera transformation
        pred_joints_world = (
            pred_joints_cam @ torch.linalg.inv(self.T_w2c)[:3, :3].T
            + torch.linalg.inv(self.T_w2c)[:3, 3]
        )
        pred_verts_world = (
            pred_verts_cam @ torch.linalg.inv(self.T_w2c)[:3, :3].T
            + torch.linalg.inv(self.T_w2c)[:3, 3]
        )

        # Keep as torch tensors
        self.pred_joints_cam = pred_joints_cam
        self.pred_joints_world = pred_joints_world
        self.pred_verts_cam = pred_verts_cam
        self.pred_verts_world = pred_verts_world

        print("Prediction data loaded")

    def load_ground_truth_data(self):
        """Load ground truth data for current sequence without dividing into clips."""
        label = self.labels[self.vid]

        # Get ground truth parameters for the full sequence
        frame_ids = label["frame_id"].numpy()
        img_dir = self.rich_dir / "images" / self.vid
        img_paths = sorted(img_dir.glob("*.jpeg"))
        frame_name_dict = {img_paths[frame_id].name: idx for idx, frame_id in enumerate(frame_ids)}
        gt_params_world = label["gt_smplx_params"]

        # Now align GT sequence to prediction clips based on frame_name_list
        gt_params_full = {}
        for key in gt_params_world:
            gt_params_full[key] = np.stack([
                gt_params_world[key][frame_name_dict[frame_name]]
                for frame_name in self.frame_name_list.flatten()
            ]).reshape(self.frame_name_list.shape + gt_params_world[key].shape[1:])
            # gt_params_full[key] = np.stack(gt_params_full[key], axis=0)

        # Create body model and get full sequence joints/vertices in world coordinates
        bm_params_world = get_bm_params(gt_params_full, device=device)
        bm_model = self.smplx_male if self.gender == "male" else self.smplx_female
        gt_bm_output_world = bm_model(**bm_params_world)
        gt_verts_world = gt_bm_output_world.full_vertices.clone().detach()

        # Convert SMPLX to SMPL for evaluation
        gt_verts_world = self.smplx2smpl @ gt_verts_world
        gt_joints_world = self.J_regressor @ gt_verts_world

        # Transform to camera coordinates
        gt_joints_cam = gt_joints_world @ self.T_w2c[:3, :3].T + self.T_w2c[:3, 3]
        gt_verts_cam = gt_verts_world @ self.T_w2c[:3, :3].T + self.T_w2c[:3, 3]

        self.gt_joints_cam = gt_joints_cam
        self.gt_joints_world = gt_joints_world
        self.gt_verts_cam = gt_verts_cam
        self.gt_verts_world = gt_verts_world

        print("Ground truth data loaded")

    def render_results(self):
        """Render prediction results with input images, downsampled by 4."""
        if not self.args.render:
            return

        if self.args.render_gt:
            save_dir = self.args.saved_data_dir
            save_dir = os.path.join(os.path.dirname(save_dir), "gt_render")
        else:
            save_dir = self.args.saved_data_dir

        # Extract cam_name from vid
        cam_name = self.vid.split("/")[2]

        render_save_dir = os.path.join(
            save_dir, self.seq_name, cam_name, "render"
        )
        os.makedirs(render_save_dir, exist_ok=True)

        # Get image dimensions from preproc data and downsample by 4
        preproc_data = self.preproc_data[self.vid]
        W_orig, H_orig = preproc_data["img_wh"]
        W_orig, H_orig = W_orig.item(), H_orig.item() 
        downsample_factor = 4
        W, H = W_orig // downsample_factor, H_orig // downsample_factor

        if self.args.render_gt:
            B, F = self.gt_joints_cam.shape[:2]
            total_frames = B * F
            verts = self.gt_verts_cam.reshape(total_frames, -1, 3)
            joints = self.gt_joints_cam.reshape(total_frames, -1, 3)
        else:
            B, F = self.pred_joints_cam.shape[:2]
            total_frames = B * F
            verts = self.pred_verts_cam.reshape(total_frames, -1, 3)
            joints = self.pred_joints_cam.reshape(total_frames, -1, 3)

        frame_names = self.frame_name_list.reshape(total_frames)
        device = verts.device

        render_batch = 150

        # Adapt camera intrinsics for downsampling
        K_downsampled = self.K.clone().float()
        K_downsampled[
            :2, :
        ] /= downsample_factor  # Scale focal length and principal point
        K_batch = K_downsampled.unsqueeze(0).expand(render_batch, -1, -1).to(device)

        # Initialize renderer once with batch K
        renderer = BatchRenderer(
            K=K_batch, img_w=W, img_h=H, faces=self.smpl_neutral.faces, 
            mesh_color="white" if self.args.render_gt else self.mesh_color,
        ).to(device)

        for i in trange(0, total_frames, render_batch):
            end_idx = min(total_frames, i + render_batch)
            current_batch_size = end_idx - i

            verts_batch = verts[i:end_idx]
            joints_batch = joints[i:end_idx]
            frame_names_batch = frame_names[i:end_idx] 

            # BatchRenderer handles variable batch sizes automatically
            fg_imgs_batch = renderer(verts_batch, joints_batch)

            for b in range(current_batch_size):
                frame_name = frame_names_batch[b]
                # Construct full image path for RICH dataset
                img_dir = self.rich_dir / "images" / self.vid
                img_path = img_dir / frame_name

                bg_img = cv2.imread(str(img_path))
                # Downsample background image
                bg_img = cv2.resize(bg_img, (W, H), interpolation=cv2.INTER_LINEAR)
                
                # Draw bounding box on downsampled image
                if self.args.use_bbox:
                    if self.args.crop_bbox and i % 200 >= 100:
                        bg_img = self.draw_bbox_on_image(bg_img, frame_name, downsample_factor, crop=True)
                    else:
                        bg_img = self.draw_bbox_on_image(bg_img, frame_name, downsample_factor)
                
                fg_img = fg_imgs_batch[b]

                bg_img = pil_img.fromarray(bg_img[..., ::-1])
                fg_img = pil_img.fromarray(fg_img)

                render_img = bg_img.copy()
                # Separate RGB and alpha channels for proper compositing
                fg_img_rgb = fg_img.convert("RGB")
                fg_img_alpha = fg_img.split()[-1]  # Extract alpha channel
                render_img.paste(fg_img_rgb, (0, 0), mask=fg_img_alpha)
                render_img.save(os.path.join(render_save_dir, frame_name))

        # Generate video
        video_dir = os.path.join(save_dir, "video")
        os.makedirs(video_dir, exist_ok=True)
        video_name = f"{self.seq_name}_{cam_name}.mp4"
        ffmpeg_cmd = f'ffmpeg -y -r 30 -pattern_type glob -i "{render_save_dir}/*.jpeg" -c:v libx264 {os.path.join(video_dir, video_name)}'
        os.system(ffmpeg_cmd)

    def run_3d_visualization(self):
        """Run 3D visualization if enabled."""
        if not (self.args.visualize_3d and self.visualizer):
            return

        print("Preparing 3D visualization...")
        try:
            # Prepare meshes from clips
            frame_to_verts, ordered_frames = self.visualizer.prepare_meshes_from_clips(
                self.pred_verts_cam.cpu().numpy(),
                self.frame_name_list,
                self.bm_neutral.faces.cpu().numpy(),
            )
            frame_to_verts_gt, _ = self.visualizer.prepare_meshes_from_clips(
                self.gt_verts_cam.cpu().numpy(),
                self.frame_name_list,
                self.bm_neutral.faces.cpu().numpy(),
            )

            # Transform vertices to world coordinates using inverse camera transformation
            cam2world_np = torch.linalg.inv(self.T_w2c).cpu().numpy()
            for frame_name in frame_to_verts:
                verts_cam = frame_to_verts[frame_name]
                frame_to_verts[frame_name] = (
                    verts_cam @ cam2world_np[:3, :3].T + cam2world_np[:3, 3]
                )

                verts_cam_gt = frame_to_verts_gt[frame_name]
                frame_to_verts_gt[frame_name] = (
                    verts_cam_gt @ cam2world_np[:3, :3].T + cam2world_np[:3, 3]
                )

            # Estimate ground height from GT joints in az coordinate
            gt_joints_az = (
                self.gt_joints_world @ self.T_w2az[:3, :3].T + self.T_w2az[:3, 3]
            )
            ground_height = gt_joints_az.amin(dim=(0, 1))[2].item()

            # Start visualization
            self.visualizer.visualize_sequence(
                frame_to_verts=frame_to_verts,
                ordered_frames=ordered_frames,
                bm_faces=self.bm_neutral.faces.cpu().numpy(),
                ground_height=ground_height,
                cam2world=torch.linalg.inv(self.T_w2c),
                frame_to_verts_gt=frame_to_verts_gt,
                vis_mode=self.args.vis_3d_mode,
            )

        except KeyboardInterrupt:
            print("\nVisualization interrupted by user. Continuing with evaluation...")
        except Exception as e:
            print(f"Error in 3D visualization: {e}")
            print("Continuing with evaluation without 3D visualization...")

    def evaluate_sequence(self, vid):
        """Evaluate a single sequence."""
        print(f"Evaluating sequence: {vid}")

        # Load all data
        self.load_sequence_data(vid)
        self.load_prediction_data()
        self.load_ground_truth_data()
        self.load_bbox_data()  # Load bounding box data

        # Estimate ground height from GT joints in az coordinate
        gt_joints_az = self.gt_joints_world @ self.T_w2az[:3, :3].T + self.T_w2az[:3, 3]
        ground_height = gt_joints_az[..., 2].amin().item()

        # Compute all metrics at once
        metrics = self.eval_metrics.compute_all_metrics(
            pred_joints_world=self.pred_joints_world,
            gt_joints_world=self.gt_joints_world,
            pred_verts_world=self.pred_verts_world,
            gt_verts_world=self.gt_verts_world,
            ground_height=ground_height,
        )

        # Add results to manager using vid as key
        self.results_manager.add_results_from_metrics_dict(vid, metrics)

        # Run visualization and rendering
        # self.run_3d_visualization()
        self.render_results()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluation on RICH dataset",
    )
    parser.add_argument(
        "--saved_data_dir",
        type=str,
        default="exp/mask_transformer/MIMO-vit-release/video_train/result_rich/inference_5_1",
        help="path to saved test results",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="datasets/mask_transformer/rich",
        help="path to RICH dataset root",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default="Gym_011_pushup1",
        help="all - evaluate on all sequences; otherwise specify sequence name to evaluate/visualize",
    )

    # Visualization
    parser.add_argument(
        "--visualize_3d",
        action="store_true",
        help="Enable 3D interactive visualization",
    )
    parser.add_argument(
        "--vis_3d_fps", default=30, type=int, help="FPS for 3D visualization playback"
    )
    parser.add_argument(
        "--vis_3d_mode",
        default=3,
        type=int,
        choices=[1, 2, 3],
        help="Visualization mode: 1=prediction only, 2=ground truth only, 3=both (default: 3)",
    )

    # Rendering
    parser.add_argument(
        "--render", action="store_true", help="render the results with input image"
    )
    parser.add_argument(
        "--render_gt", action="store_true", help="render the ground truth results"
    )
    parser.add_argument(
        "--crop_bbox", action="store_true", help="whether the bbox is cropped"
    )

    parser.add_argument(
        "--no_bbox", action="store_false", dest="use_bbox", help="disable drawing bounding boxes on images"
    )

    parser.add_argument("--method", type=str, default="MoRo", choices=method_cfg.keys(), help="Method name for evaluation configuration.")
    
    args = parser.parse_args() 
    args.smplfitter = args.method == "MoRo"
    return args


if __name__ == "__main__":
    # Parse arguments and setup
    args = parse_arguments()
    args.crop_bbox = args.crop_bbox or "crop" in args.saved_data_dir

    if args.render:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        assert (
            not args.visualize_3d
        ), "Rendering requires EGL backend, cannot visualize 3D at the same time."

    # Initialize evaluator
    evaluator = RichEvaluator(args)

    # Determine test sequences
    if args.seq_name != "all":
        # Find matching video(s) by sequence name
        test_vids = [vid for vid in evaluator.labels.keys() if args.seq_name in vid]
        if not test_vids:
            raise ValueError(f"No videos found matching sequence name: {args.seq_name}")
    else:
        test_vids = list(evaluator.labels.keys())

    # test_vids = test_vids[::10]

    print(f"Evaluating {len(test_vids)} sequences...")

    # Process each sequence
    for vid in tqdm(test_vids, desc="Evaluating sequences"):
        evaluator.evaluate_sequence(vid)

    # Compute and print final results using vid as keys
    final_metrics = evaluator.results_manager.aggregate_results(test_vids)
    evaluator.results_manager.print_results(final_metrics)

    # Save results
    joblib.dump(
        final_metrics, os.path.join(args.saved_data_dir, f"metrics_{args.seq_name}.pkl")
    )
