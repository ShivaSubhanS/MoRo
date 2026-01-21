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

from models.mask_transformer.system.renderer import BatchRenderer, create_ground_mesh, setup_global_camera
from utils.vis3d_utils import Visualizer3D
from utils.eval_utils import EvalMetrics, EvalResultsManager, fit_smplx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
body_model_path = "body_models"

method_cfg = {
    "MoRo": ("mesh", "smplx"),
}

recording_name_list = [
    "MPH1Library_00034_01",
    "N0Sofa_00034_01",
    "N0Sofa_00034_02",
    "N0Sofa_00141_01",
    "N0Sofa_00145_01",
    "N3Library_00157_01",
    "N3Library_00157_02",
    "N3Library_03301_01",
    "N3Library_03301_02",
    "N3Library_03375_01",
    "N3Library_03375_02",
    "N3Library_03403_01",
    "N3Library_03403_02",
    "N3Office_00034_01",
    "N3Office_00139_01",
    "N3Office_00150_01",
    "N3Office_00153_01",
    "N3Office_00159_01",
    "N3Office_03301_01",
]

# PROX scene floor heights
prox_floor_height = {'N0Sofa': -0.9843093165454873,
                     'MPH1Library': -0.34579620031341207,
                     'N3Library': -0.6736229583361132,
                     'N3Office': -0.7772727989022952,
                     'BasementSittingBooth': -0.767080139846674,
                     'MPH8': -0.41432886722717904,
                     'MPH11': -0.7169139211234009,
                     'MPH16': -0.8408992040141058,
                     'MPH112': -0.6419028605753081,
                     'N0SittingBooth': -0.6677103008966809,
                     'N3OpenArea': -1.0754909672969915,
                     'Werkraum': -0.6777057869851316}


class ProxEvaluator:
    """Main evaluator class for PROX dataset (GT-free evaluation)."""

    def __init__(self, args):
        self.args = args
        self.body_repr, self.bm_type = method_cfg[args.method]

        # Initialize body models
        # Prediction
        self.bm_neutral = BodyModel(
            bm_path=body_model_path,
            model_type=self.bm_type,
            gender="neutral",
            ext="pkl" if self.bm_type == "smpl" else "npz",
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
        self.eval_metrics = EvalMetrics(fps=30, gt_free_mode=True, ground_axis='y')  # Convert to y-up for visualization
        self.results_manager = EvalResultsManager(gt_free_mode=True)
        self.visualizer = (
            Visualizer3D(fps=args.vis_3d_fps, ground_axis='y') if args.visualize_3d else None  # Y-up for visualization
        )

        # PROX dataset specific setup
        self.prox_dir = Path(args.dataset_root)

        # Current sequence data (will be set for each sequence)
        self.recording_name = None
        self.scene_name = None
        self.frame_name_list = None
        self.K = None
        self.cam2world = None
        self.ground_height = None
        self.bbox_data = None  

    def load_sequence_data(self, recording_name):
        """Load sequence-specific data for PROX dataset."""
        self.recording_name = recording_name
        self.scene_name = recording_name.split("_")[0]

        # Get camera and transformation data
        with open(self.prox_dir / "cam2world" / f"{self.scene_name}.json", 'r') as f:
            cam2world = np.array(json.load(f))
        
        # Convert from z-up to y-up coordinate system
        # Transformation: x'=x, y'=z, z'=-y
        z_to_y_transform = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        cam2world = z_to_y_transform @ cam2world
        
        self.cam2world = torch.from_numpy(cam2world).float().to(device)
        
        with open(self.prox_dir / "calibration" / "Color.json", 'r') as f:
            color_cam = json.load(f)
        self.K = np.asarray(color_cam["camera_mtx"])
        self.dist_coeffs = np.asarray(color_cam["k"])

        # Ground height will be computed from predictions
        self.ground_height = None

        print(f"Loaded sequence: {self.recording_name}, scene: {self.scene_name}")

    def load_prediction_data(self):
        """Load prediction data for current sequence."""
        saved_data_path = os.path.join(
            self.args.saved_data_dir,
            self.recording_name,
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
            pred_joints_cam @ self.cam2world[:3, :3].T
            + self.cam2world[:3, 3]
        )
        pred_verts_world = (
            pred_verts_cam @ self.cam2world[:3, :3].T
            + self.cam2world[:3, 3]
        )

        # Compute ground height from the lowest y-value of predicted vertices
        self.ground_height = pred_verts_world[..., 1].min().item()
        print(f"Computed ground height from predictions: {self.ground_height:.4f}")

        # Keep as torch tensors
        self.pred_joints_cam = pred_joints_cam
        self.pred_joints_world = pred_joints_world
        self.pred_verts_cam = pred_verts_cam
        self.pred_verts_world = pred_verts_world

        print("Prediction data loaded")

    def load_bbox_data(self):
        """Load bounding box data for current recording."""
        bbox_path = os.path.join(
            self.prox_dir,
            "keypoints_openpose",
            self.recording_name,
            "bbox.npz"
        )
        
        if os.path.exists(bbox_path):
            bbox_data = np.load(bbox_path)
            frame_names_bbox = bbox_data["frame_names"]
            centers_bbox = bbox_data["centers"]
            scales_bbox = bbox_data["scales"]
            
            # Create mapping from frame name to bbox info
            self.bbox_data = {
                "centers": dict(zip(frame_names_bbox, centers_bbox)),
                "scales": dict(zip(frame_names_bbox, scales_bbox))
            }
            print("Bounding box data loaded")
        else:
            print(f"Bounding box data not found at {bbox_path}")
            self.bbox_data = None

    def draw_bbox_on_image(self, img, frame_name):
        """Draw bounding box on image if bbox data is available."""
        if self.bbox_data is None or frame_name not in self.bbox_data["centers"]:
            return img
            
        center = self.bbox_data["centers"][frame_name]
        scale = self.bbox_data["scales"][frame_name]
        
        # Convert center and scale to bbox coordinates
        # Scale represents the size, multiply by 200 to get pixel size (following dataset convention)
        bbox_size = scale * 200
        
        # Calculate top-left and bottom-right corners
        x1 = int(center[0] - bbox_size / 2)
        y1 = int(center[1] - bbox_size / 2)
        x2 = int(center[0] + bbox_size / 2)
        y2 = int(center[1] + bbox_size / 2)
        
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

    def render_results(self):
        """Render prediction results with input images for PROX (no downsampling)."""
        if not self.args.render:
            return

        save_dir = self.args.saved_data_dir
        if self.args.render_global:
            save_dir += "_global"
        render_save_dir = os.path.join(
            save_dir, self.recording_name, "render"
        )
        os.makedirs(render_save_dir, exist_ok=True)

        # Use PROX image dimensions without downsampling
        W, H = 1920, 1080

        B, F = self.pred_joints_cam.shape[:2]
        total_frames = B * F

        verts_cam = self.pred_verts_cam.reshape(total_frames, -1, 3)
        joints_cam = self.pred_joints_cam.reshape(total_frames, -1, 3)
        verts_world = self.pred_verts_world.reshape(total_frames, -1, 3) if self.args.render_global else None
        frame_names = self.frame_name_list.reshape(total_frames)
        device = verts_cam.device

        # Pre-compute ground mesh and camera transform for entire sequence
        ground_mesh = None
        R_global = None
        T_global = None
        
        if self.args.render_global:
            # Compute XZ range for entire sequence (in y-up coordinates)
            verts_world_flat = verts_world.reshape(-1, 3).cpu().numpy()
            x_min, x_max = verts_world_flat[:, 0].min(), verts_world_flat[:, 0].max()
            z_min, z_max = verts_world_flat[:, 2].min(), verts_world_flat[:, 2].max()
            x_range = (x_min, x_max)
            z_range = (z_min, z_max)
            print(f"Computed XZ range for entire sequence: X={x_range}, Z={z_range}")
            
            # Create ground mesh once for entire sequence
            ground_mesh = create_ground_mesh(
                ground_height=self.ground_height,
                x_range=x_range,
                z_range=z_range,
                tile_size=0.5,
                padding=0.2
            )
            print(f"Ground mesh created with {len(ground_mesh.vertices)} vertices")
            
            # Setup camera once for entire sequence (use mean human center)
            human_center = torch.mean(verts_world.reshape(-1, 3), dim=0).cpu().numpy()
            x_span = x_range[1] - x_range[0]
            z_span = z_range[1] - z_range[0]
            max_span = max(x_span, z_span)
            
            R_global, T_global = setup_global_camera(
                human_center=human_center,
                cam_distance=max(4.0, max_span * 0.5),
                cam_height=2.0
            )
            print(f"Camera transform computed for entire sequence")

        render_batch = 150

        # Use original camera intrinsics without downsampling
        K_batch = torch.from_numpy(self.K).float().unsqueeze(0).expand(render_batch, -1, -1).to(device)

        # Initialize renderer with full resolution
        renderer = BatchRenderer(
            K=K_batch, img_w=W, img_h=H, faces=self.smpl_neutral.faces
        ).to(device)
        
        with torch.no_grad(): 
            for i in trange(0, total_frames, render_batch):
                end_idx = min(total_frames, i + render_batch)
                current_batch_size = end_idx - i

                verts_batch = verts_cam[i:end_idx]
                joints_batch = joints_cam[i:end_idx]
                frame_names_batch = frame_names[i:end_idx] 

                # Render camera view
                fg_imgs_batch = renderer(verts_batch, joints_batch)
                
                # Render global view if needed
                if self.args.render_global:
                    verts_world_batch = verts_world[i:end_idx]
                    # Pass pre-computed ground mesh and camera transform
                    global_imgs_batch = renderer.render_global(
                        verts_world_batch,
                        ground_mesh,
                        R_global,
                        T_global
                    )

                for b in range(current_batch_size):
                    frame_name = frame_names_batch[b]
                    # Construct full image path for PROX dataset
                    img_path = (
                        self.prox_dir / "recordings" / self.recording_name / "Color" / frame_name 
                    )

                    if img_path.exists():
                        bg_img = cv2.imread(str(img_path))
                        # Undistort and flip horizontally as in dataset using loaded parameters
                        bg_img = cv2.undistort(bg_img, cameraMatrix=self.K, distCoeffs=self.dist_coeffs)
                        bg_img = cv2.flip(bg_img, 1)
                        
                        # Draw bounding box on background image
                        if self.args.use_bbox:
                            frame_name_without_ext = os.path.splitext(frame_name)[0]
                            bg_img = self.draw_bbox_on_image(bg_img, frame_name_without_ext)
                    else:
                        bg_img = np.zeros((H, W, 3), dtype=np.uint8)
                    
                    fg_img = fg_imgs_batch[b]

                    bg_img = pil_img.fromarray(bg_img[..., ::-1])
                    fg_img = pil_img.fromarray(fg_img)

                    render_img = bg_img.copy()
                    # Separate RGB and alpha channels for proper compositing
                    fg_img_rgb = fg_img.convert("RGB")
                    fg_img_alpha = fg_img.split()[-1]  # Extract alpha channel
                    render_img.paste(fg_img_rgb, (0, 0), mask=fg_img_alpha)
                    
                    if self.args.render_global:
                        global_img = global_imgs_batch[b]
                        global_img_pil = pil_img.fromarray(global_img)
                        
                        combined_width = render_img.width + global_img_pil.width
                        combined_img = pil_img.new('RGB', (combined_width, render_img.height))
                        combined_img.paste(render_img, (0, 0))
                        combined_img.paste(global_img_pil, (render_img.width, 0))
                        combined_img.save(os.path.join(render_save_dir, f"{frame_name}"))
                    else:
                        render_img.save(os.path.join(render_save_dir, f"{frame_name}"))

        # Generate video
        video_dir = os.path.join(save_dir, "video")
        os.makedirs(video_dir, exist_ok=True)
        video_name = f"{self.recording_name}.mp4"
        ffmpeg_cmd = f'ffmpeg -y -r 30 -pattern_type glob -i "{render_save_dir}/*.jpg" -c:v libx264 {os.path.join(video_dir, video_name)}'
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

            # Transform vertices to world coordinates using inverse camera transformation
            cam2world_np = self.cam2world.cpu().numpy()
            for frame_name in frame_to_verts:
                verts_cam = frame_to_verts[frame_name]
                frame_to_verts[frame_name] = (
                    verts_cam @ cam2world_np[:3, :3].T + cam2world_np[:3, 3]
                )

            # Start visualization
            self.visualizer.visualize_sequence(
                frame_to_verts=frame_to_verts,
                ordered_frames=ordered_frames,
                bm_faces=self.bm_neutral.faces.cpu().numpy(),
                ground_height=self.ground_height,
                cam2world=self.cam2world,
                frame_to_verts_gt=None,  # No GT available
                vis_mode=1,  # Prediction only
            )

        except KeyboardInterrupt:
            print("\nVisualization interrupted by user. Continuing with evaluation...")
        except Exception as e:
            print(f"Error in 3D visualization: {e}")
            print("Continuing with evaluation without 3D visualization...")

    def evaluate_sequence(self, recording_name):
        """Evaluate a single sequence (GT-free)."""
        print(f"Evaluating sequence: {recording_name}")

        # Load all data
        self.load_sequence_data(recording_name)
        self.load_prediction_data()
        self.load_bbox_data()  # Load bounding box data

        # Compute GT-free metrics only
        metrics = self.eval_metrics.compute_gt_free_metrics(
            pred_joints_world=self.pred_joints_world,
            pred_verts_world=self.pred_verts_world,
            ground_height=self.ground_height,
        )

        # Add results to manager using recording_name as key
        self.results_manager.add_results_from_metrics_dict(recording_name, metrics)

        # Run visualization and rendering
        self.run_3d_visualization()
        self.render_results()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluation on PROX dataset (GT-free)",
    )
    parser.add_argument(
        "--saved_data_dir",
        type=str,
        default="exp/mask_transformer/MIMO-vit-release/video_train/result_prox/inference_5_1",
        help="path to saved test results",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="datasets/mask_transformer/PROX",
        help="path to PROX dataset root",
    )
    parser.add_argument(
        "--recording_name",
        type=str,
        default="N0Sofa_00145_01",
        help="all - evaluate on all recordings; otherwise specify recording name to evaluate/visualize",
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

    # Rendering
    parser.add_argument(
        "--render", action="store_true", help="render the results with input image"
    )

    parser.add_argument(
        "--no_bbox", action="store_false", dest="use_bbox", help="disable drawing bounding boxes on images"
    )
    parser.add_argument(
        "--render_global",
        action="store_true",
        help="Render additional global view with world-space mesh and ground plane, concatenated horizontally"
    )

    parser.add_argument("--method", type=str, default="MoRo", choices=["MoRo"], help="Method name for evaluation configuration.")
    
    args = parser.parse_args() 
    args.smplfitter = args.method == "MoRo"
    return args


if __name__ == "__main__":
    # Parse arguments and setup
    args = parse_arguments()

    if args.render:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        assert (
            not args.visualize_3d
        ), "Rendering requires EGL backend, cannot visualize 3D at the same time."

    # Initialize evaluator
    evaluator = ProxEvaluator(args)

    # Determine test recordings
    if args.recording_name != "all":
        test_recordings = [args.recording_name]
    else:
        test_recordings = recording_name_list
    print(f"Evaluating {len(test_recordings)} recordings...")

    # Process each recording
    for recording_name in tqdm(test_recordings, desc="Evaluating recordings"):
        evaluator.evaluate_sequence(recording_name)

    # Compute and print final results
    final_metrics = evaluator.results_manager.aggregate_results(test_recordings)
    evaluator.results_manager.print_results(final_metrics)

    # Save results
    joblib.dump(
        final_metrics, os.path.join(args.saved_data_dir, f"metrics_{args.recording_name}.pkl")
    )