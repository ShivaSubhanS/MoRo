import torch
import numpy as np
import cv2
import os
from tqdm import tqdm


def read_video(video_path, output_fps=30):
    """
    Read video frames from a video file or image directory.
    
    Args:
        video_path: Path to video file or image directory
        output_fps: Target fps for video extraction (default 30)
    
    Returns:
        Tuple of (image_dir, frame_names) where frame_names is a sorted list of image filenames
    """
    if os.path.isdir(video_path):
        # Already an image folder
        image_dir = video_path
        frame_names = sorted([
            os.path.join(video_path, f) for f in os.listdir(video_path)
            if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']
        ])
    else:
        # Video file - extract frames
        base_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        image_dir = os.path.join(base_dir, f"{video_name}_frames")
        os.makedirs(image_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, round(source_fps / output_fps))
        
        frame_names = []
        frame_idx = 0
        saved_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame_name = f"{saved_idx:06d}.jpg"
                frame_path = os.path.join(image_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_names.append(frame_path)
                saved_idx += 1
            
            frame_idx += 1
        
        cap.release()
    
    return image_dir, frame_names


def get_video_lwh(image_paths):
    image = cv2.imread(image_paths[0])
    h, w, _ = image.shape
    return len(image_paths), w, h


def read_video_np(image_paths):
    """
    Read images from paths and return as numpy array.
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        numpy array of shape (L, H, W, 3) in RGB format
    """
    images = []
    for img_path in tqdm(image_paths, desc="Reading frames"):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return np.stack(images, axis=0)


def draw_bbx_xys_on_images(bbx_xys, images):
    """
    Draw bounding boxes on images.
    
    Args:
        bbx_xys: tensor of shape (N, L, 3) with [center_x, center_y, scale] format for N tracks
        images: numpy array of shape (L, H, W, 3) in RGB format
    
    Returns:
        numpy array of same shape with boxes drawn
    """
    images_with_boxes = images.copy()
    bbx_xys = bbx_xys.cpu().numpy()  # (N, L, 3)
    
    N, L, _ = bbx_xys.shape
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for frame_idx in range(L):
        for track_idx in range(N):
            cx, cy, scale = bbx_xys[track_idx, frame_idx]
            half_size = scale * 100.0 
            x1 = int(cx - half_size)
            y1 = int(cy - half_size)
            x2 = int(cx + half_size)
            y2 = int(cy + half_size)
            
            color = colors[track_idx % len(colors)]
            cv2.rectangle(images_with_boxes[frame_idx], (x1, y1), (x2, y2), color, 2)
            # Add track ID label
            cv2.putText(images_with_boxes[frame_idx], f"ID{track_idx}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return images_with_boxes


def save_video(frames, output_path, fps=30):
    """
    Save a video from a batch of frames.
    
    Args:
        frames: numpy array of shape (L, H, W, 3) in RGB format
        output_path: path to save the video
        fps: frames per second
    """
    L, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()

def fit_ground_plane(verts):
    """
    Fit ground plane and transform vertices to y-up world coordinates.
    
    Args:
        verts: torch tensor of shape (L, V, 3) - vertices in camera frame
    
    Returns:
        verts_world: torch tensor of shape (L, V, 3) - vertices in world frame (y-up)
    """
    device = verts.device
    L, V, _ = verts.shape
    
    # SMPL foot vertex indices (left and right foot bottom vertices)
    # These are approximate indices for SMPL mesh
    # left_foot_indices = [3387, 3365, 3386, 3383]  # Left foot bottom
    # right_foot_indices = [6728, 6706, 6727, 6724]  # Right foot bottom
    # foot_indices = left_foot_indices + right_foot_indices
    foot_indices = [3216, 3387, 6617, 6787]
    
    # Extract foot vertices
    foot_verts = verts[:, foot_indices, :]  # (L, 8, 3)
    
    # Compute velocity (frame-to-frame difference)
    foot_vel = torch.zeros_like(foot_verts)
    foot_vel[:-1] = foot_verts[1:] - foot_verts[:-1]
    foot_vel_norm = torch.norm(foot_vel, dim=-1)  # (L, 8)
    
    # Detect contact: velocity below threshold
    vel_threshold = 0.01  # 1cm per frame, for 30 fps ~0.3 m/s
    contact_mask = foot_vel_norm < vel_threshold  # (L, 8)
    
    contact_points = foot_verts[contact_mask].cpu().numpy()  # (N_contacts, 3)
    
    if len(contact_points) < 3:
        # Not enough contact points, use all foot vertices
        contact_points = foot_verts.reshape(-1, 3).cpu().numpy()
    else:
        contact_points = np.array(contact_points)
    
    # RANSAC to fit ground plane
    from sklearn.linear_model import RANSACRegressor
    
    # Fit plane: ax + by + cz + d = 0, we solve for z = mx + ny + c
    X = contact_points[:, :2]  # x, y coordinates
    y = contact_points[:, 2]    # z coordinate
    
    ransac = RANSACRegressor(random_state=0, min_samples=3, residual_threshold=0.05)
    ransac.fit(X, y)
    
    # Get plane normal (in camera frame)
    # Plane equation: z = m*x + n*y + c  =>  m*x + n*y - z + c = 0
    # Normal vector: [m, n, -1]
    m, n = ransac.estimator_.coef_
    plane_normal_cam = np.array([m, n, -1.0])
    plane_normal_cam = plane_normal_cam / np.linalg.norm(plane_normal_cam)
    
    # Target normal is [0, 1, 0] (y-up in world frame)
    target_normal = np.array([0.0, 1.0, 0.0])
    
    # Compute rotation to align plane_normal_cam to target_normal
    # Using Rodrigues' rotation formula
    v = np.cross(plane_normal_cam, target_normal)
    s = np.linalg.norm(v)
    c = np.dot(plane_normal_cam, target_normal)
    
    if s < 1e-6:  # Already aligned or opposite
        if c > 0:
            R = np.eye(3)
        else:
            # 180 degree rotation around any perpendicular axis
            perp = np.array([1.0, 0.0, 0.0]) if abs(plane_normal_cam[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            v = np.cross(plane_normal_cam, perp)
            v = v / np.linalg.norm(v)
            R = 2 * np.outer(v, v) - np.eye(3)
    else:
        # Skew-symmetric cross-product matrix
        vx = np.array([[0, -v[2], v[1]], 
                       [v[2], 0, -v[0]], 
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    
    R = torch.from_numpy(R).float().to(device)
    
    # Apply rotation to all vertices
    verts_rotated = verts @ R.T  # (L, V, 3)
    
    # Find lowest point after rotation and translate to ground (y=0)
    lowest_y = verts_rotated[:, :, 1].min()
    translation = torch.tensor([0.0, -lowest_y, 0.0], device=device)
    
    verts_world = verts_rotated + translation

    print(f"Ground plane fitted: normal={plane_normal_cam}, contact_points={len(contact_points)}")
    print(f"Rotation applied, translated by {translation.cpu().numpy()}")
    
    return verts_world