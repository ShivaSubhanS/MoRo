import hydra
import os
from tqdm import trange

import numpy as np
from omegaconf import OmegaConf, DictConfig

import torch
import lightning as L
from lightning import Trainer
import joblib
import cv2
import PIL.Image as pil_img

from smplfitter.pt import BodyModel as BodyModelFit
from smplfitter.pt import BodyFitter

from models.mask_transformer import MaskTransformerModule
from models.mask_transformer.data import DataModuleDemo 
from models.mask_transformer.utils.body_model import BodyModel
from models.mask_transformer.system.renderer import BatchRenderer, create_ground_mesh, setup_global_camera
from utils.tracker import Tracker
from utils.demo_utils import read_video, read_video_np, get_video_lwh, draw_bbx_xys_on_images, save_video, fit_ground_plane
from utils.eval_utils import fit_smplx
from utils.gender_utils import predict_gender
from utils.HumanFOV import estimate_focal_length


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoRoDemo:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.cfg.exp_dir = os.path.join("./exp/mask_transformer", cfg.name, cfg.benchmark)
        os.environ["DATA_ROOT"] = "."
        # Body models are initialized lazily after inference to avoid OOM

    def init_body_models(self):
        """Initialize body models on GPU. Called after inference to avoid VRAM conflict."""
        body_model_path = "body_models"

        # SMPLX body model for fitting
        self.bm_neutral = BodyModel(
            bm_path=body_model_path,
            model_type="smplx",
            gender="neutral",
            ext="npz",
        ).to(device)

        # SMPLH body model for rendering
        self.smpl_neutral = BodyModel(
            bm_path=body_model_path, model_type="smplh", gender="male", ext="pkl"
        ).to(device)

        # SMPLX to SMPL conversion
        self.smplx2smpl = (
            torch.load("body_models/smplx2smpl_sparse.pt", weights_only=True)
            .to(device)
            .to_dense()
        )
        self.J_regressor = torch.load(
            "body_models/smpl_neutral_J_regressor.pt", weights_only=True
        ).to(device)

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
        print("Body models initialized on GPU.")

    # Processing video
    def preprocess_video(self):
        video_path = self.cfg.demo.video_path
        verbose = self.cfg.demo.verbose

        video_dir, image_paths = read_video(video_path)
        self.L, self.W, self.H = get_video_lwh(image_paths)

        # Get bbx tracking result
        bbx_path = os.path.join(video_dir, "bbx.pt")
        if not os.path.exists(bbx_path):
            tracker = Tracker()
            bbx_xys = tracker.get_tracks(image_paths).float()  # (N, L, 3)
            torch.save({"bbx_xys": bbx_xys}, bbx_path)
            del tracker
            torch.cuda.empty_cache()
        else:
            bbx_xys = torch.load(bbx_path)["bbx_xys"]

        if verbose:
            video = read_video_np(image_paths)
            video_overlay = draw_bbx_xys_on_images(bbx_xys, video)
            bbx_vis_path = os.path.join(video_dir, "bbx_overlay.mp4")
            save_video(video_overlay, bbx_vis_path)

        if self.cfg.demo.focal_length is not None:
            focal = self.cfg.demo.focal_length
            print(f"Using provided focal length: {focal:.2f}")
        else:
            # use HumanFOV to estimate focal length 
            focal_lengths = estimate_focal_length(image_paths)
            focal = np.median(focal_lengths)
            print(f"Estimated focal length from HumanFOV: {focal:.2f}")

        K = np.array([[focal, 0, self.W / 2], [0, focal, self.H / 2], [0, 0, 1]], dtype=np.float32)

        self.video_dir = video_dir
        self.image_paths = image_paths
        self.bbx_xys = bbx_xys
        self.num_tracks = bbx_xys.shape[0]
        self.K = K

        # Detect gender per track (or use provided override)
        forced_gender = self.cfg.demo.get("gender", None)
        if forced_gender is not None and forced_gender in ("male", "female", "neutral"):
            print(f"[Gender] Using provided gender: {forced_gender} for all tracks")
            self.track_genders = [forced_gender] * self.num_tracks
        else:
            print("[Gender] Auto-detecting gender per track ...")
            self.track_genders = [
                predict_gender(image_paths, bbx_xys, track_idx=i)
                for i in range(self.num_tracks)
            ]

        if self.cfg.demo.name is None:
            self.cfg.demo.name = os.path.basename(video_dir)


    # Testing
    def predict_results(self):
        # setup datamodule
        datamodule = DataModuleDemo( 
            self.cfg.dataset,
            self.image_paths,
            self.bbx_xys,
            self.K,
        )

        torch.cuda.empty_cache()

        # load trained model (map to CPU first, Trainer will move to GPU)
        ckpt_path = os.path.join(
            self.cfg.exp_dir,
            "checkpoints",
            "last.ckpt",
        )
        mask_transformer_module = MaskTransformerModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location="cpu",
            weights_only=False,
            strict=True,
            cfg=self.cfg,
        )
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            **self.cfg.trainer,
        )
        trainer.predict(mask_transformer_module, datamodule=datamodule)
        self.result_dir = mask_transformer_module.result_dir

        # clean up transformer from GPU entirely before loading body models
        del datamodule
        del mask_transformer_module
        del trainer
        torch.cuda.empty_cache()

    def load_prediction_data(self, track_idx):
        """Load prediction data from saved results."""
        saved_data_path = os.path.join(self.result_dir, f"id{track_idx}", "results.pkl")

        saved_data = joblib.load(saved_data_path)
        self.frame_name_list = saved_data["frame_name"]  # [B, F]
        self.frame_name_list = np.vectorize(lambda x: os.path.basename(x))(
            self.frame_name_list
        )

        # Load mesh vertices [B, F, V, 3]
        pred_partial_verts_cam = torch.from_numpy(saved_data["verts"]).to(device)

        fit_verts_cam, pose_rotvecs, betas, trans = fit_smplx(
            pred_partial_verts_cam, self, ret_params=True
        )
        pred_verts_cam = self.smplx2smpl @ fit_verts_cam

        # Compute joints for rendering
        pred_joints_cam = self.J_regressor @ pred_verts_cam

        self.pred_verts_cam = pred_verts_cam
        self.pred_joints_cam = pred_joints_cam
        self.smplx_pose_rotvecs = pose_rotvecs  # (B, F, J, 3)
        self.smplx_betas = betas                # (B, num_betas)
        self.smplx_trans = trans                # (B, F, 3)

        print("Prediction data loaded")

    def save_npz(self, track_idx):
        """Save SMPL-X parameters as NPZ compatible with the SMPL-X Blender addon (SMPL-X format)."""
        self.load_prediction_data(track_idx)

        track_result_dir = os.path.join(self.result_dir, f"id{track_idx}")
        os.makedirs(track_result_dir, exist_ok=True)

        gender = self.track_genders[track_idx] if hasattr(self, "track_genders") else "neutral"

        B, F, J, _ = self.smplx_pose_rotvecs.shape
        NUM_SMPLX_JOINTS = 55  # SMPL-X has 55 joints total

        for b in range(B):
            # poses: (F, 55*3) axis-angle, all SMPL-X joints
            poses_raw = self.smplx_pose_rotvecs[b].reshape(F, -1).cpu().numpy().astype(np.float32)

            # Pad to (F, 165) if fitter returned fewer joints
            if poses_raw.shape[1] < NUM_SMPLX_JOINTS * 3:
                pad = np.zeros((F, NUM_SMPLX_JOINTS * 3 - poses_raw.shape[1]), dtype=np.float32)
                poses_raw = np.concatenate([poses_raw, pad], axis=1)

            betas = self.smplx_betas[b].cpu().numpy().astype(np.float32)
            trans = self.smplx_trans[b].cpu().numpy().astype(np.float32)

            npz_path = os.path.join(track_result_dir, f"smplx_seq{b:02d}.npz")
            np.savez(
                npz_path,
                poses=poses_raw,              # (F, 165)
                betas=betas,                  # (10,)
                trans=trans,                  # (F, 3)
                gender=gender,                # detected or provided
                mocap_framerate=np.float32(30.0),
            )
            print(f"Saved NPZ for track {track_idx} seq {b} gender={gender}: {npz_path}")

    def load_bbox_data(self):
        """Load bounding box data for visualization."""
        bbox_path = os.path.join(self.video_dir, "bbx.pt")

        if os.path.exists(bbox_path):
            bbox_data = torch.load(bbox_path)
            self.bbox_xys = bbox_data["bbx_xys"]  # [N, L, 3]
            print("Bounding box data loaded")
        else:
            print(f"Bounding box data not found at {bbox_path}")
            self.bbox_xys = None

    def draw_bbox_on_image(self, img, frame_idx, track_idx=0):
        """Draw bounding box on image for specific frame and track."""
        if self.bbox_xys is None:
            return img

        if track_idx >= self.bbox_xys.shape[0] or frame_idx >= self.bbox_xys.shape[1]:
            return img

        cx, cy, scale = self.bbox_xys[track_idx, frame_idx].cpu().numpy()
        half_size = scale * 100.0

        x1 = int(cx - half_size)
        y1 = int(cy - half_size)
        x2 = int(cx + half_size)
        y2 = int(cy + half_size)

        # Ensure bbox is within image bounds
        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        img_with_bbox = img.copy()
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return img_with_bbox

    # Rendering
    def render_results(self, track_idx):
        """Render prediction results with input images for a specific track."""
        if not self.cfg.demo.get("render", True):
            return

        # Load prediction and bbox data for this track
        self.load_prediction_data(track_idx)
        self.load_bbox_data()
        
        # Set up rendering directory for this track
        track_result_dir = os.path.join(self.result_dir, f"id{track_idx}")
        render_save_dir = os.path.join(track_result_dir, "render")
        os.makedirs(render_save_dir, exist_ok=True)

        B, F = self.pred_joints_cam.shape[:2]
        total_frames = B * F

        verts_cam = self.pred_verts_cam.reshape(total_frames, -1, 3)
        joints_cam = self.pred_joints_cam.reshape(total_frames, -1, 3)
        frame_names = self.frame_name_list.reshape(total_frames)

        render_batch = 150
        render_global = self.cfg.demo.get("render_global", True)

        # Use precomputed K instead of recomputing
        K_batch = torch.from_numpy(self.K).float().unsqueeze(0).expand(render_batch, -1, -1).to(device)

        # Initialize renderer
        renderer = BatchRenderer(
            K=K_batch, img_w=self.W, img_h=self.H, faces=self.smpl_neutral.faces
        ).to(device)

        # Pre-compute ground mesh and camera for global rendering
        ground_mesh = None
        R_global = None
        T_global = None
        verts_world = None
        
        if render_global:
            # Simple camera model: assume identity for demo (no actual camera extrinsics)

            # Fit ground plane from vertices
            verts_world = fit_ground_plane(verts_cam)
 
            # Compute XZ range for entire sequence (assuming y-up)
            verts_world_flat = verts_world.reshape(-1, 3).cpu().numpy()
            x_min, x_max = verts_world_flat[:, 0].min(), verts_world_flat[:, 0].max()
            z_min, z_max = verts_world_flat[:, 2].min(), verts_world_flat[:, 2].max()
            x_range = (x_min, x_max)
            z_range = (z_min, z_max)
            
            # Estimate ground height from feet
            # ground_height = verts_world_flat[:, 1].min() - 0.05  # 5cm below lowest point
            ground_height = 0. 
            
            # Create ground mesh
            ground_mesh = create_ground_mesh(
                ground_height=ground_height,
                x_range=x_range,
                z_range=z_range,
                tile_size=0.5,
                padding=0.2
            )
            
            # Setup global camera
            human_center = torch.mean(verts_world.reshape(-1, 3), dim=0).cpu().numpy()
            x_span = x_range[1] - x_range[0]
            z_span = z_range[1] - z_range[0]
            max_span = max(x_span, z_span)
            
            R_global, T_global = setup_global_camera(
                human_center=human_center,
                cam_distance=max(4.0, max_span * 0.5),
                cam_height=2.0
            )
            print(f"Global rendering setup: ground_height={ground_height:.2f}, X={x_range}, Z={z_range}")

        use_bbox = self.cfg.demo.get("use_bbox", True)

        with torch.no_grad():
            for i in trange(0, total_frames, render_batch):
                end_idx = min(total_frames, i + render_batch)
                current_batch_size = end_idx - i

                verts_batch = verts_cam[i:end_idx]
                joints_batch = joints_cam[i:end_idx]

                # Render camera view
                fg_imgs_batch = renderer(verts_batch, joints_batch)
                
                # Render global view if enabled
                if render_global:
                    verts_world_batch = verts_world[i:end_idx]
                    global_imgs_batch = renderer.render_global(
                        verts_world_batch,
                        ground_mesh,
                        R_global,
                        T_global
                    )

                for b in range(current_batch_size):
                    frame_idx = i + b

                    # Load background image
                    img_path = self.image_paths[frame_idx]
                    if os.path.exists(img_path):
                        bg_img = cv2.imread(img_path)

                        # Draw bounding box if enabled
                        if use_bbox:
                            bg_img = self.draw_bbox_on_image(bg_img, frame_idx % F, track_idx)
                    else:
                        bg_img = np.zeros((self.H, self.W, 3), dtype=np.uint8)

                    fg_img = fg_imgs_batch[b]

                    # Composite images
                    bg_img = pil_img.fromarray(bg_img[..., ::-1])
                    fg_img = pil_img.fromarray(fg_img)

                    render_img = bg_img.copy()
                    fg_img_rgb = fg_img.convert("RGB")
                    fg_img_alpha = fg_img.split()[-1]
                    render_img.paste(fg_img_rgb, (0, 0), mask=fg_img_alpha)
                    
                    # Combine with global view if enabled
                    if render_global:
                        global_img = global_imgs_batch[b]
                        global_img_pil = pil_img.fromarray(global_img)
                        
                        combined_width = render_img.width + global_img_pil.width
                        combined_img = pil_img.new('RGB', (combined_width, render_img.height))
                        combined_img.paste(render_img, (0, 0))
                        combined_img.paste(global_img_pil, (render_img.width, 0))
                        combined_img.save(os.path.join(render_save_dir, f"{frame_idx:06d}.jpg"))
                    else:
                        render_img.save(os.path.join(render_save_dir, f"{frame_idx:06d}.jpg"))

        # Generate video
        video_path = os.path.join(track_result_dir, "result.mp4")
        fps = 30 
        ffmpeg_cmd = f'ffmpeg -y -pattern_type glob -i "{render_save_dir}/*.jpg" -r {fps} -c:v libx264 {video_path}'
        os.system(ffmpeg_cmd)
        print(f"Video for track {track_idx} saved to {video_path}")

    def run(self):
        self.preprocess_video()

        # Step 1: inference (transformer on GPU, no body models loaded)
        self.predict_results()

        # Step 2: init body models on GPU now that transformer is freed
        self.init_body_models()

        npz_only = self.cfg.demo.get("npz_only", False)
        for track_idx in trange(self.num_tracks, desc="Processing tracks"):
            if npz_only:
                self.save_npz(track_idx)
            else:
                self.render_results(track_idx)
                self.save_npz(track_idx)


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.register_new_resolver("eval", eval)

    moro = MoRoDemo(cfg)
    moro.run()


if __name__ == "__main__":
    main()