import os
import pickle as pkl
import joblib
import pandas as pd
import numpy as np
import torch
import json
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
from models.mask_transformer.system.renderer import BatchRenderer, create_ground_mesh, setup_global_camera
from utils.vis3d_utils import Visualizer3D
from utils.eval_utils import EvalMetrics, EvalResultsManager, fit_smplx
import trimesh

# y axis up
egobody_floor_height = {
    "seminar_g110": -1.660,
    "seminar_d78": -0.810,
    "seminar_j716": -0.8960,
    "seminar_g110_0315": -0.73,
    "seminar_d78_0318": -1.03,
    "seminar_g110_0415": -0.77,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
body_model_path = "body_models"
dataset = "egobody"
dataset_root = "datasets/mask_transformer/EgoBody"

method_cfg = {
    "MoRo": ("mesh", "smplx"),
    "RoHM": ("param", "smplx"), 
    "TokenHMR": ("param", "smplh"),
    "WHAM": ("param", "smpl"),
    "GVHMR": ("param", "smplx"),
    "PromptHMR": ("param", "smplx"),
}

# Method-specific colors for visualization
method_colors = {
    "MoRo": "skin",  # Default color
    "RoHM": "red",
    "TokenHMR": "purple",
    "WHAM": "orange",
    "GVHMR": "cyan",
    "PromptHMR": "pink",
}


class EgoBodyEvaluator:
    """Main evaluator class for EgoBody dataset."""

    def __init__(self, args):
        self.args = args
        self.body_repr, self.bm_type = method_cfg[args.method]
        self._bm_type = "smplx" if self.bm_type == "smplx" else "smpl"
        self.mesh_color = method_colors.get(args.method, "skin")

        # Initialize body models
        self.bm_neutral = BodyModel(
            bm_path=body_model_path, 
            model_type=self.bm_type, 
            gender="neutral",
            ext="pkl" if self.bm_type == "smpl" else "npz",
        ).to(device)
        # For GT
        self.smpl_male = BodyModel(
            bm_path=body_model_path, model_type="smplh", gender="male"
        ).to(device)
        self.smpl_female = BodyModel(
            bm_path=body_model_path, model_type="smplh", gender="female"
        ).to(device)

        # convert to SMPL
        self.smplx2smpl = (
            torch.load("body_models/smplx2smpl_sparse.pt", weights_only=True)
            .to(device)
            .to_dense()
        )
        self.J_regressor = torch.load(
            "body_models/smpl_neutral_J_regressor.pt", weights_only=True
        )[:22].to(
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

        # Initialize evaluation utilities with visibility mask support for EgoBody
        self.eval_metrics = EvalMetrics(fps=30, has_visibility_mask=True)
        self.results_manager = EvalResultsManager()
        self.visualizer = Visualizer3D(fps=args.vis_3d_fps)

        # Current recording data (will be set for each recording)
        self.recording_name = None
        self.view = None
        self.body_idx = None
        self.scene_name = None
        self.gender_gt = None
        self.body_idx_fpv = None
        self.cam2world = None
        self.master2world = None
        self.ground_height = None
        self.frame_name_list = None
        self.visibility_mask = None  # New field for visibility mask
        self.bbox_data = None  # New field for bounding box data

    def load_calibration_data(self):
        """Load camera calibration data for current recording."""
        calib_trans_dir = os.path.join(
            dataset_root, "calibrations", self.recording_name
        )

        # Load master to world transformation
        with open(
            os.path.join(
                calib_trans_dir,
                "cal_trans",
                "kinect12_to_world",
                self.scene_name + ".json",
            ),
            "r",
        ) as f:
            master2world = np.asarray(json.load(f)["trans"])

        # Load sub camera to master transformation
        if self.view == "sub_1":
            trans_subtomain_path = os.path.join(
                calib_trans_dir, "cal_trans", "kinect_11to12_color.json"
            )
        elif self.view == "sub_2":
            trans_subtomain_path = os.path.join(
                calib_trans_dir, "cal_trans", "kinect_13to12_color.json"
            )
        elif self.view == "sub_3":
            trans_subtomain_path = os.path.join(
                calib_trans_dir, "cal_trans", "kinect_14to12_color.json"
            )
        elif self.view == "sub_4":
            trans_subtomain_path = os.path.join(
                calib_trans_dir, "cal_trans", "kinect_15to12_color.json"
            )

        if self.view != "master":
            with open(trans_subtomain_path, "r") as f:
                trans_subtomain = np.asarray(json.load(f)["trans"])
            cam2world = np.matmul(master2world, trans_subtomain)
        else:
            cam2world = master2world

        self.cam2world = torch.from_numpy(cam2world).float().to(device)
        self.master2world = torch.from_numpy(master2world).float().to(device)

        # Load camera intrinsics
        with open(
            os.path.join(
                dataset_root, "kinect_cam_params", f"kinect_{self.view}", "Color.json"
            ),
            "r",
        ) as f:
            color_cam = json.load(f)

        self.K = np.array(color_cam["camera_mtx"])
        self.dist_coeffs = np.array(color_cam["k"])

    def load_prediction_data(self):
        """Load prediction data for current recording."""
        saved_data_path = os.path.join(
            self.args.saved_data_dir,
            self.recording_name,
            self.view,
            f"body_idx_{self.body_idx}",
            "results.pkl",
        )

        saved_data = joblib.load(saved_data_path)

        self.frame_name_list = saved_data["frame_name"]  # [B, F]
        self.frame_name_list = np.vectorize(lambda x: os.path.basename(x))(
            self.frame_name_list
        )  # keep only file names

        if self.body_repr == "param":
            pred_bm_params_cam = {
                "body_pose": saved_data["body_pose"],
                "betas": saved_data["betas"],
                "global_orient": saved_data["global_orient"],
                "transl": saved_data["transl"],
            }
            pred_bm_params_cam = {
                k: torch.from_numpy(v).float().to(device) for k, v in pred_bm_params_cam.items()
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

        pred_joints_world = (
            pred_joints_cam @ self.cam2world[:3, :3].T + self.cam2world[:3, 3]
        )
        pred_verts_world = (
            pred_verts_cam @ self.cam2world[:3, :3].T + self.cam2world[:3, 3]
        )

        # Keep as torch tensors
        self.pred_joints_cam = pred_joints_cam
        self.pred_joints_world = pred_joints_world
        self.pred_verts_cam = pred_verts_cam
        self.pred_verts_world = pred_verts_world

        print("Prediction data loaded")

    def load_ground_truth_data(self):
        """Load ground truth data for current recording."""
        # Determine GT data path
        interactee_idx = int(self.body_idx_fpv.split(" ")[0])
        if self.body_idx == interactee_idx:
            fitting_gt_root = os.path.join(
                dataset_root,
                f"smpl_interactee",
                self.recording_name,
                f"body_idx_{self.body_idx}",
                "results",
            )
        else:
            fitting_gt_root = os.path.join(
                dataset_root,
                f"smpl_camera_wearer",
                self.recording_name,
                f"body_idx_{self.body_idx}",
                "results",
            )

        frame_list = sorted(os.listdir(fitting_gt_root))
        frame_idx_dict = {}
        param_gt_list = []

        for frame_idx, cur_frame_name in enumerate(frame_list):
            frame_idx_dict[cur_frame_name] = frame_idx
            with open(
                os.path.join(fitting_gt_root, cur_frame_name, "000.pkl"), "rb"
            ) as f:
                param_gt = pkl.load(f)
            param_gt_list.append(param_gt)

        # Extract GT parameters for current sequences
        gt_params_master_list = [
            param_gt_list[
                frame_idx_dict[os.path.splitext(seq_frames[0])[0]] : frame_idx_dict[
                    os.path.splitext(seq_frames[-1])[0]
                ]
                + 1
            ]
            for seq_frames in self.frame_name_list
        ]

        # Load visibility mask using the same frame indexing
        mask_path = os.path.join(
            dataset_root, "mask_joint", self.recording_name, self.view, "mask_joint.npy"
        )
        
        if os.path.exists(mask_path):
            mask_joint = np.load(mask_path)  # [F, 24]
            # Take first 22 joints as specified
            mask_joint = mask_joint[:, :22]  # [F, 22]
            
            self.visibility_mask = torch.from_numpy(mask_joint).unsqueeze(0).to(device)
            print("Visibility mask loaded")
        else:
            print(f"Visibility mask not found at {mask_path}")
            self.visibility_mask = None

        gt_params_master = {}
        for key in ["transl", "global_orient", "body_pose", "betas"]:
            gt_params_master[key] = np.stack(
                [
                    np.concatenate([param_gt[key] for param_gt in seq_param])
                    for seq_param in gt_params_master_list
                ]
            )
            if key == "body_pose":
                gt_params_master[key] = gt_params_master[key][..., :63]

        # Generate GT meshes and joints
        master2cam = torch.linalg.solve(self.cam2world, self.master2world)
        gt_bm_params_master = get_bm_params(gt_params_master, device=device)
        gt_bm_model = self.smpl_male if self.gender_gt == "male" else self.smpl_female
        gt_bm_output_master = gt_bm_model(**gt_bm_params_master)
        gt_verts_master = gt_bm_output_master.vertices.clone().detach()
        gt_joints_master = self.J_regressor @ gt_verts_master

        # Transform to camera and world coordinates
        gt_joints_cam = gt_joints_master @ master2cam[:3, :3].T + master2cam[:3, 3]
        gt_joints_world = (
            gt_joints_master @ self.master2world[:3, :3].T + self.master2world[:3, 3]
        )
        gt_verts_cam = gt_verts_master @ master2cam[:3, :3].T + master2cam[:3, 3]
        gt_verts_world = (
            gt_verts_master @ self.master2world[:3, :3].T + self.master2world[:3, 3]
        )

        # Keep as torch tensors
        self.gt_joints_cam = gt_joints_cam
        self.gt_joints_world = gt_joints_world
        self.gt_verts_cam = gt_verts_cam
        self.gt_verts_world = gt_verts_world

        print("Ground truth data loaded")

    def load_bbox_data(self):
        """Load bounding box data for current recording."""
        bbox_path = os.path.join(
            dataset_root,
            "keypoints_cleaned",
            self.recording_name,
            self.view,
            f"bbox_idx{self.body_idx}.npz"
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
        """Render prediction results with input images."""
        if not self.args.render:
            return

        if self.args.render_gt:
            save_dir = self.args.saved_data_dir
            save_dir = os.path.join(os.path.dirname(save_dir), "gt_render")
        else:
            save_dir = self.args.saved_data_dir

        if self.args.render_global:
            save_dir += "_global" 

        render_save_dir = os.path.join(
            save_dir,
            self.recording_name,
            self.view,
            f"body_idx_{self.body_idx}",
            "render",
        )
        os.makedirs(render_save_dir, exist_ok=True)

        H, W = 1080, 1920

        if self.args.render_gt:
            B, F = self.gt_joints_cam.shape[:2]
            total_frames = B * F
            verts_cam = self.gt_verts_cam.reshape(total_frames, -1, 3)
            joints_cam = self.gt_joints_cam.reshape(total_frames, -1, 3)
            verts_world = self.gt_verts_world.reshape(total_frames, -1, 3) if self.args.render_global else None
        else:
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
            # Compute XZ range for entire sequence
            verts_world_flat = verts_world.reshape(-1, 3).cpu().numpy()
            x_min, x_max = verts_world_flat[:, 0].min(), verts_world_flat[:, 0].max()
            z_min, z_max = verts_world_flat[:, 2].min(), verts_world_flat[:, 2].max()
            x_range = (x_min, x_max)
            z_range = (z_min, z_max)
            
            # Create ground mesh once for entire sequence
            ground_mesh = create_ground_mesh(
                ground_height=self.ground_height,
                x_range=x_range,
                z_range=z_range,
                tile_size=0.5,
                padding=0.2
            )
            
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

        render_batch = 150

        # Create K matrix - same for both views
        K_full = (
            torch.from_numpy(self.K)
            .float()
            .unsqueeze(0)
            .to(device)
            .expand(render_batch, -1, -1)
        )

        # Initialize renderers
        renderer = BatchRenderer(
            K=K_full, img_w=W, img_h=H, faces=self.smpl_male.faces, 
            mesh_color="white" if self.args.render_gt else self.mesh_color
        ).to(device)
        
        # Disable gradients for rendering
        with torch.no_grad():
            for i in trange(0, total_frames, render_batch):
                end_idx = min(total_frames, i + render_batch)
                current_batch_size = end_idx - i

                verts_cam_batch = verts_cam[i:end_idx]
                joints_cam_batch = joints_cam[i:end_idx]
                frame_names_batch = frame_names[i:end_idx]

                # Render camera view
                fg_imgs_batch = renderer(verts_cam_batch, joints_cam_batch)
                
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

                # Process each frame in batch
                for b in range(current_batch_size):
                    img_path = frame_names_batch[b]
                    img_name = os.path.basename(img_path)
                    full_img_path = os.path.join(
                        dataset_root,
                        "kinect_color",
                        self.recording_name,
                        self.view,
                        img_path,
                    )
                    bg_img = cv2.imread(full_img_path)

                    if self.dist_coeffs is not None:
                        bg_img = cv2.undistort(bg_img.copy(),
                                                np.asarray(self.K),
                                                np.asarray(self.dist_coeffs))

                    frame_name_without_ext = os.path.splitext(img_name)[0]
                    if self.args.use_bbox:
                        bg_img = self.draw_bbox_on_image(bg_img, frame_name_without_ext)
                    
                    fg_img = fg_imgs_batch[b]

                    bg_img = pil_img.fromarray(bg_img[..., ::-1])
                    fg_img = pil_img.fromarray(fg_img)

                    render_img = bg_img.copy()
                    fg_img_rgb = fg_img.convert("RGB")
                    fg_img_alpha = fg_img.split()[-1]
                    render_img.paste(fg_img_rgb, (0, 0), mask=fg_img_alpha)
                    
                    if self.args.render_global:
                        global_img = global_imgs_batch[b]
                        global_img_pil = pil_img.fromarray(global_img)
                        
                        combined_width = render_img.width + global_img_pil.width
                        combined_img = pil_img.new('RGB', (combined_width, render_img.height))
                        combined_img.paste(render_img, (0, 0))
                        combined_img.paste(global_img_pil, (render_img.width, 0))
                        combined_img.save(os.path.join(render_save_dir, img_name))
                    else:
                        render_img.save(os.path.join(render_save_dir, img_name))

        # Generate video
        video_dir = os.path.join(save_dir, "video")
        os.makedirs(video_dir, exist_ok=True)
        video_name = f"{self.recording_name}-{self.view}-{self.body_idx}.mp4"
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
            frame_to_verts_gt, _ = self.visualizer.prepare_meshes_from_clips(
                self.gt_verts_cam.cpu().numpy(),
                self.frame_name_list,
                self.bm_neutral.faces.cpu().numpy(),
            )

            # Transform vertices to world coordinates
            cam2world_np = self.cam2world.cpu().numpy()
            for frame_name in frame_to_verts:
                verts_cam = frame_to_verts[frame_name]
                frame_to_verts[frame_name] = (
                    verts_cam @ cam2world_np[:3, :3].T + cam2world_np[:3, 3]
                )

                verts_cam_gt = frame_to_verts_gt[frame_name]
                frame_to_verts_gt[frame_name] = (
                    verts_cam_gt @ cam2world_np[:3, :3].T + cam2world_np[:3, 3]
                )

            # Start visualization
            self.visualizer.visualize_sequence(
                frame_to_verts=frame_to_verts,
                ordered_frames=ordered_frames,
                bm_faces=self.bm_neutral.faces.cpu().numpy(),
                ground_height=self.ground_height,
                cam2world=self.cam2world,
                frame_to_verts_gt=frame_to_verts_gt,
                vis_mode=self.args.vis_3d_mode,
            )

        except KeyboardInterrupt:
            print("\nVisualization interrupted by user. Continuing with evaluation...")
        except Exception as e:
            print(f"Error in 3D visualization: {e}")
            print("Continuing with evaluation without 3D visualization...")

    def get_eval_frame_range(self):
        """Get the frame range for evaluation based on target start/end frames."""
        # Assuming B=1, get the frame names for the sequence
        frame_names = self.frame_name_list[0] # [F]

        start_frame_name = f"frame_{self.target_start_frame:05d}.jpg" 
        end_frame_name = f"frame_{self.target_end_frame:05d}.jpg"

        # Find indices where start and end frames appear
        start_indices = np.where(frame_names == start_frame_name)[0]
        end_indices = np.where(frame_names == end_frame_name)[0]

        assert len(start_indices) == 1
        start_idx = int(start_indices[0])

        if len(end_indices) == 1:
            end_idx = int(end_indices[0]) + 1
        else:
            assert self.args.method == "RoHM"
            end_idx = len(frame_names)
            self.visibility_mask = self.visibility_mask[:, start_idx:end_idx]

        return start_idx, end_idx

    def evaluate_recording(self, recording_info):
        """Evaluate a single recording."""
        # Set recording information
        self.recording_name = recording_info["recording_name"]
        self.view = recording_info["view"]
        self.body_idx = recording_info["body_idx"]
        self.scene_name = recording_info["scene_name"]
        self.gender_gt = recording_info["gender_gt"]
        self.body_idx_fpv = recording_info["body_idx_fpv"]
        self.target_start_frame = recording_info.get("target_start_frame")
        self.target_end_frame = recording_info.get("target_end_frame")
        self.ground_height = egobody_floor_height[self.scene_name]

        print(f"Evaluating recording: {self.recording_name}")

        # Load all data
        self.load_calibration_data()
        self.load_prediction_data()
        self.load_ground_truth_data()
        self.load_bbox_data()  # Load bounding box data

        # Get evaluation frame range
        eval_start_idx, eval_end_idx = self.get_eval_frame_range()

        # Compute all metrics at once (pass torch tensors directly)
        metrics = self.eval_metrics.compute_all_metrics(
            pred_joints_world=self.pred_joints_world.clone(),
            gt_joints_world=self.gt_joints_world.clone(),
            pred_verts_world=self.pred_verts_world.clone(),
            gt_verts_world=self.gt_verts_world.clone(),
            ground_height=self.ground_height,
            visibility_mask=self.visibility_mask.clone(),
            eval_frame_range=(eval_start_idx, eval_end_idx) if eval_start_idx is not None else None,
        )

        # Add results to manager using the new method
        self.results_manager.add_results_from_metrics_dict(self.recording_name, metrics)

        # # Run visualization and rendering (use full sequence)
        # self.run_3d_visualization()
        self.render_results()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluation on EgoBody dataset",
    )
    parser.add_argument(
        "--saved_data_dir",
        type=str,
        default="exp/mask_transformer/MIMO-vit-release/video_train/result_egobody/inference_5_1",
        help="path to saved test results",
    )
    parser.add_argument(
        "--recording_name",
        type=str,
        default="recording_20220318_S33_S34_01",
        help="all - evaluate on all subsequences; otherwise specify the recording name to evaluate/visualize",
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
        "--no_bbox", action="store_false", dest="use_bbox", help="disable drawing bounding boxes on images"
    )
    parser.add_argument(
        "--render_global",
        action="store_true",
        help="Render additional global view with world-space mesh and ground plane, concatenated horizontally"
    )

    parser.add_argument("--method", type=str, default="MoRo", choices=method_cfg.keys(), help="Method name for evaluation configuration.")
    parser.add_argument(
        "--mesh_color",
        type=str,
        default=None,
        choices=["skin", "red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "lime", "magenta", "teal"],
        help="Override mesh color (if not specified, uses method-specific color)"
    )

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

    # Load dataset information
    df = pd.read_csv(
        os.path.join(dataset_root, "egobody_occ_info.csv")
    )

    # Create data dictionaries
    data_dict = {
        "view": dict(zip(df["recording_name"], df["view"])),
        "body_idx": dict(zip(df["recording_name"], df["target_idx"])),
        "scene_name": dict(zip(df["recording_name"], df["scene_name"])),
        "gender_gt": dict(zip(df["recording_name"], df["target_gender"])),
        "body_idx_fpv": dict(zip(df["recording_name"], df["body_idx_fpv"])),
        "target_start_frame": dict(zip(df["recording_name"], df["target_start_frame"])),
        "target_end_frame": dict(zip(df["recording_name"], df["target_end_frame"])),
    }

    # Determine test recordings
    if args.recording_name != "all":
        test_recording_name_list = [args.recording_name]
    else:
        test_recording_name_list = list(df["recording_name"])

    # Initialize evaluator
    evaluator = EgoBodyEvaluator(args)
    
    # Override mesh color if specified
    if args.mesh_color:
        evaluator.mesh_color = args.mesh_color

    # Process each recording
    for recording_name in test_recording_name_list:
        recording_info = {
            "recording_name": recording_name,
            "view": data_dict["view"][recording_name],
            "body_idx": data_dict["body_idx"][recording_name],
            "scene_name": data_dict["scene_name"][recording_name],
            "gender_gt": data_dict["gender_gt"][recording_name],
            "body_idx_fpv": data_dict["body_idx_fpv"][recording_name],
            "target_start_frame": data_dict["target_start_frame"][recording_name],
            "target_end_frame": data_dict["target_end_frame"][recording_name],
        }
        evaluator.evaluate_recording(recording_info)

    # Compute and print final results
    final_metrics = evaluator.results_manager.aggregate_results(
        test_recording_name_list
    )
    evaluator.results_manager.print_results(final_metrics)

    # Save results
    joblib.dump(
        final_metrics,
        os.path.join(args.saved_data_dir, f"metrics_{args.recording_name}.pkl"),
    )
