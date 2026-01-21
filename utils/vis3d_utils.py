import os
import numpy as np
import colorsys
import torch
import time
import trimesh
import pyrender
import tkinter as tk
from typing import Dict, List, Optional, Tuple

color_dict = {
    "skin": colorsys.hsv_to_rgb(0.6, 0.5, 1.0),
    "blue": (0.2, 0.3, 1.0),
    "red": (0.8, 0.2, 0.2),
    # "blue": (0.2, 0.4, 0.8),
    "green": (0.2, 0.7, 0.3),
    "yellow": (0.9, 0.8, 0.2),
    "purple": (0.6, 0.2, 0.8),
    "orange": (0.9, 0.5, 0.1),
    "pink": (0.9, 0.4, 0.6),
    "cyan": (0.2, 0.7, 0.8),
    "lime": (0.5, 0.9, 0.2),
    "magenta": (0.8, 0.2, 0.6),
    "teal": (0.2, 0.6, 0.6),
    "white": (1.0, 1.0, 1.0),
}

class Visualizer3D:
    """3D visualization utility for human mesh sequences."""
    
    def __init__(self, fps: int = 30, viewport_size: Tuple[int, int] = (1200, 800), ground_axis: str = 'y'):
        self.fps = fps
        self.viewport_size = viewport_size
        self.ground_axis = ground_axis  # 'y' for Y-up (XZ ground), 'z' for Z-up (XY ground)
        
        # Set up axis indices based on ground_axis
        if ground_axis == 'y':
            self.height_axis = 1  # y-axis is up
            self.horizontal_axes = [0, 2]  # x, z are horizontal
            self.world_up = np.array([0, 1, 0])
        elif ground_axis == 'z':
            self.height_axis = 2  # z-axis is up
            self.horizontal_axes = [0, 1]  # x, y are horizontal
            self.world_up = np.array([0, 0, 1])
        else:
            raise ValueError(f"Invalid ground_axis: {ground_axis}. Must be 'y' or 'z'")
        
    def prepare_meshes_from_clips(self, pred_verts_cam: np.ndarray, 
                                 frame_name_list: np.ndarray, 
                                 bm_faces: np.ndarray) -> Tuple[Dict, List]:
        """
        Prepare 3D meshes from overlapping clips, using latter batch results for overlaps.
        
        Args:
            pred_verts_cam: [B, F, V, 3] predicted vertices in camera coordinates
            frame_name_list: [B, F] frame names for each clip
            bm_faces: body model faces
        
        Returns:
            dict: frame_name -> vertices mapping
            list: ordered frame names
        """
        frame_to_verts = {}
        
        # Process clips and handle overlaps (use latter batch result)
        for batch_idx in range(pred_verts_cam.shape[0]):
            for frame_idx in range(pred_verts_cam.shape[1]):
                frame_name = frame_name_list[batch_idx, frame_idx]
                frame_to_verts[frame_name] = pred_verts_cam[batch_idx, frame_idx]
        
        # Get ordered frame names
        ordered_frames = sorted(frame_to_verts.keys())
        
        return frame_to_verts, ordered_frames

    def check_display_available(self) -> bool:
        """Check if display is available for 3D visualization."""
        try:
            # Check if we're in a headless environment
            display = os.environ.get('DISPLAY')
            if not display:
                return False
            
            # Try to import and initialize a simple GUI toolkit
            root = tk.Tk()
            root.withdraw()  # Hide the window
            root.destroy()
            return True
        except:
            return False

    def setup_camera(self, human_center: np.ndarray, cam2world: torch.Tensor = None) -> np.ndarray:
        """Setup camera position for optimal viewing based on ground_axis."""
        cam_distance = 4.0
        
        if self.ground_axis == 'y':
            # Y-up: place camera above and to the side
            cam_height = human_center[1] + 2.0  # 2m above human center
            cam_pos = np.array([
                human_center[0] - cam_distance * 0.7,  # Slightly to the side
                cam_height,
                human_center[2] - cam_distance * 0.7   # Back and to the side
            ])
        elif self.ground_axis == 'z':
            # Z-up: place camera in front and above in Z direction
            cam_height = human_center[2] + 2.0  # 2m above human center in Z
            cam_pos = np.array([
                human_center[0] - cam_distance * 0.7,  # Slightly to the side
                human_center[1] - cam_distance * 0.7,  # Slightly forward
                cam_height
            ])
        
        # Point camera at human center
        target = human_center
        
        # Camera coordinate system (OpenGL convention)
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Use world up vector for reference based on ground_axis
        right = np.cross(forward, self.world_up)
        right = right / np.linalg.norm(right)
        
        # Recompute up to ensure orthogonality
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create camera matrix (camera-to-world transform)
        cam_matrix = np.eye(4)
        cam_matrix[:3, 0] = right
        cam_matrix[:3, 1] = up
        cam_matrix[:3, 2] = -forward  # Negative because OpenGL convention
        cam_matrix[:3, 3] = cam_pos
        
        return cam_matrix

    def create_checkerboard_ground(self, ground_height: float, cam_trans: Optional[np.ndarray] = None, ground_center: Optional[np.ndarray] = None):
        """
        Create a checkerboard ground floor mesh with thickness.
        
        Args:
            ground_height: Height of the ground floor
            cam_trans: Camera transformation matrix to apply (optional)
            ground_center: [3] XYZ center position for the ground plane (optional)
        
        Returns:
            pyrender.Mesh: Checkerboard ground mesh
        """
        # Checkerboard parameters
        # color0 = [0.8, 0.9, 0.9]
        color0 = [0.95, 0.95, 0.95]
        # color1 = [0.6, 0.7, 0.7]
        color1 = [0.5, 0.5, 0.5]
        alpha = 1.0
        tile_width = 0.5
        ground_thickness = 0.02  # Add thickness to make ground visible from above
        color0 = np.array(color0 + [alpha])
        color1 = np.array(color1 + [alpha])
        
        # Make checkerboard
        length = 25.0
        radius = length / 2.0
        num_rows = num_cols = int(length / tile_width)
        vertices = []
        faces = []
        face_colors = []
        
        # Calculate ground center offset
        if ground_center is not None:
            center_offset_x = ground_center[0]
            center_offset_z = ground_center[2]
        else:
            center_offset_x = 0.0
            center_offset_z = 0.0
        
        for i in range(num_rows):
            for j in range(num_cols):
                start_loc = [-radius + j * tile_width + center_offset_x, 
                           radius - i * tile_width + center_offset_z]
                
                if self.ground_axis == 'y':
                    # Y-up: ground is XZ plane, create a box with thickness
                    cur_verts = np.array([
                        # Top face
                        [start_loc[0], ground_height, start_loc[1]],
                        [start_loc[0], ground_height, start_loc[1] - tile_width],
                        [start_loc[0] + tile_width, ground_height, start_loc[1] - tile_width],
                        [start_loc[0] + tile_width, ground_height, start_loc[1]],
                        # Bottom face
                        [start_loc[0], ground_height - ground_thickness, start_loc[1]],
                        [start_loc[0], ground_height - ground_thickness, start_loc[1] - tile_width],
                        [start_loc[0] + tile_width, ground_height - ground_thickness, start_loc[1] - tile_width],
                        [start_loc[0] + tile_width, ground_height - ground_thickness, start_loc[1]],
                    ])
                elif self.ground_axis == 'z':
                    # Z-up: ground is XY plane, create a box with thickness
                    cur_verts = np.array([
                        # Top face
                        [start_loc[0], start_loc[1], ground_height],
                        [start_loc[0], start_loc[1] - tile_width, ground_height],
                        [start_loc[0] + tile_width, start_loc[1] - tile_width, ground_height],
                        [start_loc[0] + tile_width, start_loc[1], ground_height],
                        # Bottom face
                        [start_loc[0], start_loc[1], ground_height - ground_thickness],
                        [start_loc[0], start_loc[1] - tile_width, ground_height - ground_thickness],
                        [start_loc[0] + tile_width, start_loc[1] - tile_width, ground_height - ground_thickness],
                        [start_loc[0] + tile_width, start_loc[1], ground_height - ground_thickness],
                    ])
                
                # Create faces for the box (top, bottom, and sides)
                cur_faces = np.array([
                    # Top face
                    [0, 1, 3], [1, 2, 3],
                    # Bottom face  
                    [4, 7, 5], [5, 7, 6],
                    # Side faces
                    [0, 4, 1], [1, 4, 5],  # Front
                    [2, 6, 3], [3, 6, 7],  # Back
                    [0, 3, 4], [3, 7, 4],  # Left
                    [1, 5, 2], [2, 5, 6],  # Right
                ], dtype=int)
                
                # Offset face indices for current tile
                cur_faces += 8 * (i * num_cols + j)  # 8 vertices per tile
                
                # Determine checkerboard color
                use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
                cur_color = color0 if use_color0 else color1
                
                # Create face colors for all faces (12 faces per tile)
                cur_face_colors = np.array([cur_color] * 12)
                
                vertices.append(cur_verts)
                faces.append(cur_faces)
                face_colors.append(cur_face_colors)
        
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        face_colors = np.concatenate(face_colors, axis=0)
        
        ground_tri = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors, process=False)
        
        # Apply camera transformation if provided
        if cam_trans is not None:
            ground_tri.apply_transform(np.linalg.inv(cam_trans))
        
        ground_mesh = pyrender.Mesh.from_trimesh(ground_tri, smooth=False)
        return ground_mesh

    def apply_displacement(self, vertices: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """
        Apply spatial displacement to vertices.
        
        Args:
            vertices: [F, V, 3] or [V, 3] vertex coordinates
            displacement: [3] displacement vector
        
        Returns:
            np.ndarray: Displaced vertices
        """
        return vertices + displacement

    def create_multi_source_static_visualizer(self, multi_source_data: List[Dict], 
                                            ground_height: float, cam2world: torch.Tensor, ground_center: Optional[np.ndarray] = None):
        """
        Static 3D visualization showing multiple frames from multiple sources together.
        
        Args:
            multi_source_data: List of dictionaries, each containing:
                - 'frame_to_verts': Dictionary mapping frame names to vertices
                - 'ordered_frames': List of ordered frame names
                - 'bm_faces': Body model faces
                - 'color': Color specification for this source
                - 'name': Name/identifier for this source
                - 'displacement': [3] spatial displacement vector
            ground_height: Height of the ground plane
            cam2world: Camera to world transformation
            ground_center: [3] XYZ center position for the ground plane (optional)
        """
        print(f"Creating multi-source static 3D scene (ground_axis: {self.ground_axis})...")
        scene = pyrender.Scene()
        
        # Create checkerboard ground with specified center
        ground_mesh = self.create_checkerboard_ground(ground_height, ground_center=ground_center)
        ground_node = pyrender.Node(mesh=ground_mesh, name='checkerboard_ground')
        scene.add_node(ground_node)
        
        # Process each source
        all_vertices = []  # For camera positioning
        
        for source_idx, source_data in enumerate(multi_source_data):
            frame_to_verts = source_data['frame_to_verts']
            ordered_frames = source_data['ordered_frames']
            bm_faces = source_data['bm_faces']
            source_color = source_data['color']
            source_name = source_data['name']
            displacement = source_data['displacement']
            
            total_frames = len(ordered_frames)
            # Show all frames if less than 10
            frame_indices = list(range(total_frames))
            
            selected_frames = [ordered_frames[i] for i in frame_indices]
            print(f"Source '{source_name}': Visualizing {len(selected_frames)} frames out of {total_frames} total frames")
            print(f"  Selected frame indices: {frame_indices}")
            print(f"  Displacement: {displacement}")
            
            # Generate color variations for frames within this source
            base_color = np.array(source_color[:3])  # RGB part
            frame_colors = []
            for i in range(len(selected_frames)):
                # Create slight variations in transparency and brightness
                # alpha = max(200, 255 - i * 10)  # More opaque, decreasing transparency
                # brightness_factor = 1.0 - i * 0.05  # Slight brightness variation
                alpha = 255
                brightness_factor = 1.0
                frame_color = (base_color * brightness_factor).astype(int)
                frame_color = np.append(frame_color, alpha)
                frame_colors.append(frame_color)
            
            # Add meshes for selected frames from this source
            for i, frame_name in enumerate(selected_frames):
                vertices = frame_to_verts[frame_name]
                
                # Apply spatial displacement
                displaced_vertices = self.apply_displacement(vertices, displacement)
                all_vertices.append(displaced_vertices)
                
                # Create mesh with source-specific color
                trimesh_obj = trimesh.Trimesh(
                    vertices=displaced_vertices, 
                    faces=bm_faces, 
                    process=False
                )
                trimesh_obj.visual.vertex_colors = frame_colors[i]
                mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
                node = pyrender.Node(mesh=mesh, name=f'{source_name}_frame_{i}')
                scene.add_node(node)
        
        # Set up camera view based on all vertices and aligned with checkerboard axis
        if all_vertices:
            all_vertices = np.concatenate(all_vertices, axis=0)
            human_center = np.mean(all_vertices, axis=0)
        else:
            human_center = np.array([0, 0, 0])
        
        # Setup camera aligned with checkerboard axis
        cam_distance = 4.0
        cam_height = 6.0
        
        if self.ground_axis == 'y':
            # Y-up: View along Z-axis (parallel to checkerboard rows/columns)
            # Place camera behind and above, looking forward along +Z direction
            cam_pos = np.array([
                human_center[0],  # Aligned with human center on X
                human_center[1] + cam_height,  # 2m above human center
                human_center[2] - cam_distance  # Behind in -Z direction
            ])
        elif self.ground_axis == 'z':
            # Z-up: View along Y-axis (parallel to checkerboard rows/columns)
            # Place camera behind and above, looking forward along +Y direction
            cam_pos = np.array([
                human_center[0],  # Aligned with human center on X
                human_center[1] - cam_distance,  # Behind in -Y direction
                human_center[2] + cam_height  # 2m above human center in Z
            ])
        
        # Point camera at human center
        target = human_center
        
        # Camera coordinate system (OpenGL convention)
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Use world up vector for reference based on ground_axis
        right = np.cross(forward, self.world_up)
        right = right / np.linalg.norm(right)
        
        # Recompute up to ensure orthogonality
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create camera matrix (camera-to-world transform)
        cam_matrix = np.eye(4)
        cam_matrix[:3, 0] = right
        cam_matrix[:3, 1] = up
        cam_matrix[:3, 2] = -forward  # Negative because OpenGL convention
        cam_matrix[:3, 3] = cam_pos
        
        # Create camera with our desired pose
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.5)
        camera_node = pyrender.Node(camera=camera, matrix=cam_matrix)
        scene.add_node(camera_node)
        
        # Create viewer
        viewer = pyrender.Viewer(
            scene, 
            use_raymond_lighting=True, 
            run_in_thread=False,  # Blocking mode for static visualization
            record=False,
            viewport_size=self.viewport_size
        )
        
        # Print visualization info
        axis_info = {
            'y': "Y-axis up (XZ ground plane)",
            'z': "Z-axis up (XY ground plane)"
        }
        print(f"Multi-Source Static 3D Visualization")
        print(f"Coordinate system: {axis_info[self.ground_axis]}")
        print(f"Showing {len(multi_source_data)} sources with spatial displacement")
        for i, source_data in enumerate(multi_source_data):
            print(f"  {i+1}. {source_data['name']}: {source_data['color'][:3]} RGB, displacement {source_data['displacement']}")
        print("Controls:")
        print("- Mouse: Rotate camera around target")
        print("- Scroll: Zoom in/out")
        print("- Middle mouse + drag: Pan camera")
        print("- Close window to continue")

    def visualize_multi_source_static(self, multi_source_data: List[Dict], 
                                    ground_height: float, cam2world: torch.Tensor, ground_center: Optional[np.ndarray] = None):
        """
        Entry point for multi-source static 3D visualization.
        
        Args:
            multi_source_data: List of source data dictionaries
            ground_height: Height of the ground plane
            cam2world: Camera to world transformation
            ground_center: [3] XYZ center position for the ground plane (optional)
        """
        # Check if display is available
        if not self.check_display_available():
            print("No display available for static 3D visualization. Skipping...")
            return
        
        self.create_multi_source_static_visualizer(
            multi_source_data, ground_height, cam2world, ground_center
        )

    def create_static_visualizer(self, frame_to_verts: Dict, ordered_frames: List, 
                               bm_faces: np.ndarray, ground_height: float, 
                               cam2world: torch.Tensor, frame_to_verts_gt: Optional[Dict] = None, 
                               vis_mode: int = 3):
        """
        Static 3D visualization showing multiple frames together in one scene.
        
        Args:
            frame_to_verts: Dictionary mapping frame names to vertices
            ordered_frames: List of ordered frame names
            bm_faces: Body model faces
            ground_height: Height of the ground plane
            cam2world: Camera to world transformation
            frame_to_verts_gt: Ground truth vertices (optional)
            vis_mode: Visualization mode (1=pred only, 2=gt only, 3=both)
        """
        print(f"Creating static 3D scene (ground_axis: {self.ground_axis})...")
        scene = pyrender.Scene()
        
        # Show all frames if less than 10
        total_frames = len(ordered_frames)
        frame_indices = list(range(total_frames))
        
        selected_frames = [ordered_frames[i] for i in frame_indices]
        print(f"Visualizing {len(selected_frames)} frames out of {total_frames} total frames")
        print(f"Selected frame indices: {frame_indices}")
        
        # Create checkerboard ground
        ground_mesh = self.create_checkerboard_ground(ground_height)
        ground_node = pyrender.Node(mesh=ground_mesh, name='checkerboard_ground')
        scene.add_node(ground_node)
        
        # Color scheme for multiple frames
        # Use different shades/transparencies for different frames - made more opaque
        pred_colors = [
            [66, 149, 245, 220],   # Light blue - more opaque
            [66, 190, 245, 200],   # Lighter blue
            [100, 149, 245, 180],  # Purple-blue
            [40, 149, 245, 160],   # Darker blue
            [66, 149, 200, 140],   # Muted blue
            [120, 149, 245, 220],  # Bright blue
            [66, 120, 245, 200],   # Deep blue
            [80, 170, 245, 180],   # Sky blue
            [50, 149, 220, 160],   # Steel blue
            [90, 149, 255, 140],   # Electric blue
        ]
        
        gt_colors = [
            [245, 66, 66, 220],    # Light red - more opaque
            [245, 100, 66, 200],   # Orange-red
            [245, 66, 100, 180],   # Pink-red
            [220, 66, 66, 160],    # Dark red
            [245, 120, 120, 140],  # Muted red
            [255, 80, 80, 220],    # Bright red
            [200, 66, 66, 200],    # Deep red
            [245, 90, 90, 180],    # Rose red
            [230, 66, 66, 160],    # Crimson
            [255, 100, 100, 140],  # Light crimson
        ]
        
        # Add meshes for selected frames
        for i, frame_name in enumerate(selected_frames):
            vertices_pred = frame_to_verts[frame_name]
            vertices_gt = frame_to_verts_gt[frame_name] if frame_to_verts_gt is not None else None
            
            color_idx = i % len(pred_colors)
            
            # Add prediction mesh
            if vis_mode in [1, 3]:
                pred_trimesh = trimesh.Trimesh(
                    vertices=vertices_pred, 
                    faces=bm_faces, 
                    process=False
                )
                pred_trimesh.visual.vertex_colors = pred_colors[color_idx]
                pred_mesh = pyrender.Mesh.from_trimesh(pred_trimesh)
                pred_node = pyrender.Node(mesh=pred_mesh, name=f'body_pred_frame_{i}')
                scene.add_node(pred_node)
            
            # Add ground truth mesh if available
            if vis_mode in [2, 3] and vertices_gt is not None:
                gt_trimesh = trimesh.Trimesh(
                    vertices=vertices_gt, 
                    faces=bm_faces, 
                    process=False
                )
                gt_trimesh.visual.vertex_colors = gt_colors[color_idx]
                gt_mesh = pyrender.Mesh.from_trimesh(gt_trimesh)
                gt_node = pyrender.Node(mesh=gt_mesh, name=f'body_gt_frame_{i}')
                scene.add_node(gt_node)
        
        # Set up camera view
        first_frame = selected_frames[0]
        first_verts = frame_to_verts[first_frame]
        human_center = np.mean(first_verts, axis=0)
        cam_matrix = self.setup_camera(human_center, cam2world)
        
        # Create camera with our desired pose
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.5)
        camera_node = pyrender.Node(camera=camera, matrix=cam_matrix)
        scene.add_node(camera_node)
        
        # Create viewer
        viewer = pyrender.Viewer(
            scene, 
            use_raymond_lighting=True, 
            run_in_thread=False,  # Blocking mode for static visualization
            record=False,
            viewport_size=self.viewport_size
        )
        
        # Print visualization info
        mode_names = {1: "prediction only (blue)", 2: "ground truth only (red)", 3: "both"}
        axis_info = {
            'y': "Y-axis up (XZ ground plane)",
            'z': "Z-axis up (XY ground plane)"
        }
        print(f"Static 3D Visualization - Mode: {mode_names.get(vis_mode, 'unknown')}")
        print(f"Coordinate system: {axis_info[self.ground_axis]}")
        print(f"Showing {len(selected_frames)} frames with different colors/transparencies")
        print("Controls:")
        print("- Mouse: Rotate camera around target")
        print("- Scroll: Zoom in/out")
        print("- Middle mouse + drag: Pan camera")
        print("- Close window to continue")

    def compute_trajectory_camera(self, multi_source_data: List[Dict], cam2world: torch.Tensor) -> np.ndarray:
        """
        Compute optimal camera position based on pelvis trajectory for good viewing of the sequence.
        
        Args:
            multi_source_data: List of source data dictionaries
            cam2world: Original camera to world transformation
            
        Returns:
            np.ndarray: 4x4 camera transformation matrix
        """
        # Collect all pelvis trajectories
        all_pelvis_positions = []
        for source_data in multi_source_data:
            if 'pelvis_trajectory' in source_data:
                pelvis_positions = source_data['pelvis_trajectory']  # [F, 3]
                displacement = source_data['displacement']
                displaced_pelvis = pelvis_positions + displacement
                all_pelvis_positions.append(displaced_pelvis)
        
        if not all_pelvis_positions:
            # Fallback to original camera setup
            return self.setup_camera(np.array([0, 0, 0]), cam2world)
        
        # Combine all trajectories
        all_pelvis = np.concatenate(all_pelvis_positions, axis=0)  # [N, 3]
        
        # Compute trajectory statistics
        trajectory_center = np.mean(all_pelvis, axis=0)  # [3] - center of all motion
        trajectory_min = np.min(all_pelvis, axis=0)      # [3] - minimum bounds
        trajectory_max = np.max(all_pelvis, axis=0)      # [3] - maximum bounds
        trajectory_span = trajectory_max - trajectory_min # [3] - size of motion area
        
        print(f"Trajectory center: {trajectory_center}")
        print(f"Trajectory span: {trajectory_span}")
        
        # Determine camera positioning based on ground axis
        if self.ground_axis == 'y':
            # Y-up coordinate system
            # Camera should be above the trajectory looking down
            
            # Horizontal distance from trajectory center (much closer for better frame filling)
            horizontal_span = max(trajectory_span[0], trajectory_span[2])  # max of X, Z span
            cam_distance_horizontal = max(2.5, horizontal_span * 0.6)  # Reduced from 2.5 and 0.6x to 1.5 and 0.4x
            
            # Height above trajectory (reduced for closer view)
            trajectory_height_span = trajectory_span[1]  # Y span (height variation)
            trajectory_max_height = trajectory_max[1]    # Highest point in trajectory
            cam_height = trajectory_max_height + max(3.0, trajectory_height_span * 1.6 + 2.5)  # Reduced from 3.5 and 1.8x+3.0 to 2.5 and 1.5x+2.0
            
            # Position camera parallel to X-axis (looking along Z direction)
            # This makes the camera plane parallel to the XY plane
            cam_pos = np.array([
                trajectory_center[0],  # Centered on X
                cam_height,
                trajectory_center[2] - cam_distance_horizontal  # Behind the trajectory (negative Z direction)
            ])
            
            # Look at trajectory center (slightly above ground)
            target = trajectory_center.copy()
            target[1] = trajectory_center[1]
            
        elif self.ground_axis == 'z':
            # Z-up coordinate system
            # Camera should be above the trajectory looking down
            
            # Horizontal distance from trajectory center (much closer for better frame filling)
            horizontal_span = max(trajectory_span[0], trajectory_span[1])  # max of X, Y span
            cam_distance_horizontal = max(2.5, horizontal_span * 0.6)  # Reduced from 2.5 and 0.6x
            
            # Height above trajectory (reduced for closer view)
            trajectory_height_span = trajectory_span[2]  # Z span (height variation)
            trajectory_max_height = trajectory_max[2]    # Highest point in trajectory
            cam_height = trajectory_max_height + max(3.0, trajectory_height_span * 1.6 + 2.5)  # Reduced from 3.5 and 1.8x+3.0 to 2.5 and 1.5x+2.0
            
            # Position camera parallel to X-axis (looking along Y direction)
            # This makes the camera plane parallel to the XZ plane
            cam_pos = np.array([
                trajectory_center[0],  # Centered on X
                trajectory_center[1] - cam_distance_horizontal,  # Behind the trajectory (negative Y direction)
                cam_height
            ])
            
            # Look at trajectory center
            target = trajectory_center.copy()
            target[2] = trajectory_center[2]
        
        print(f"Camera position: {cam_pos}")
        print(f"Camera target: {target}")
        
        # Compute camera orientation
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Use world up vector
        right = np.cross(forward, self.world_up)
        right = right / np.linalg.norm(right)
        
        # Recompute up to ensure orthogonality
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create camera matrix (camera-to-world transform)
        cam_matrix = np.eye(4)
        cam_matrix[:3, 0] = right
        cam_matrix[:3, 1] = up
        cam_matrix[:3, 2] = -forward  # Negative because OpenGL convention
        cam_matrix[:3, 3] = cam_pos
        
        return cam_matrix

    def create_multi_source_animated_visualizer(self, multi_source_data: List[Dict], 
                                              ground_height: float, cam2world: torch.Tensor, 
                                              ground_center: Optional[np.ndarray] = None,
                                              save_frames: bool = False, 
                                              save_dir: Optional[str] = None):
        """
        Animated 3D visualization showing frame-by-frame playback from multiple sources.
        
        Args:
            multi_source_data: List of dictionaries, each containing:
                - 'frame_to_verts': Dictionary mapping frame names to vertices
                - 'ordered_frames': List of ordered frame names
                - 'bm_faces': Body model faces
                - 'color': Color specification for this source
                - 'name': Name/identifier for this source
                - 'displacement': [3] spatial displacement vector
                - 'label': Label name (includes _cam suffix if applicable)
            ground_height: Height of the ground plane
            cam2world: Camera to world transformation
            ground_center: [3] XYZ center position for the ground plane (optional)
            save_frames: Whether to save individual frames as images
            save_dir: Directory to save frames (required if save_frames=True)
        """
        print(f"Creating multi-source animated 3D scene (ground_axis: {self.ground_axis})...")
        
        # Get all unique frame names and sort them
        all_frames = set()
        for source_data in multi_source_data:
            all_frames.update(source_data['ordered_frames'])
        sorted_frames = sorted(list(all_frames))
        
        print(f"Animation will play {len(sorted_frames)} frames at {self.fps} FPS")
        print(f"Sources: {[source_data['name'] for source_data in multi_source_data]}")
        
        # Set up frame saving if requested
        if save_frames:
            if save_dir is None:
                save_dir = "3d_vis_frames"
            os.makedirs(save_dir, exist_ok=True)
            print(f"Will save frames to: {save_dir}")
            
            # Create offscreen renderer for saving frames
            offscreen_renderer = pyrender.OffscreenRenderer(
                viewport_width=self.viewport_size[0],
                viewport_height=self.viewport_size[1]
            )
        
        # Initialize scene
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])  # Increased from [0.1, 0.1, 0.1] to [0.3, 0.3, 0.3]
        
        # Create checkerboard ground with specified center
        ground_mesh = self.create_checkerboard_ground(ground_height, ground_center=ground_center)
        ground_node = pyrender.Node(mesh=ground_mesh, name='checkerboard_ground')
        scene.add_node(ground_node)
        
        # Compute optimal camera position based on trajectory
        print("Computing optimal camera position based on trajectory...")
        cam_matrix = self.compute_trajectory_camera(multi_source_data, cam2world)
        
        # Create camera with optimal pose
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.5)
        camera_node = pyrender.Node(camera=camera, matrix=cam_matrix)
        scene.add_node(camera_node)
        
        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)  # Increased from 1.0 to 2.5
        light_node = pyrender.Node(light=light, matrix=cam_matrix)
        scene.add_node(light_node)
        
        # Create persistent pelvis trajectory spheres for each source (subsample for performance)
        print("Creating pelvis trajectory spheres...")
        for source_idx, source_data in enumerate(multi_source_data):
            source_color = source_data['color']
            source_name = source_data['name']
            displacement = source_data['displacement']
            source_label = source_data.get('label', '')
            
            # Skip if no pelvis data available
            if 'pelvis_trajectory' not in source_data:
                continue
                
            pelvis_positions = source_data['pelvis_trajectory']  # [F, 3] array of pelvis positions
            
            # Subsample pelvis trajectory for performance (every 10th frame)
            subsample_step = 5
            subsampled_pelvis = pelvis_positions[::subsample_step]
            
            # Create small spheres for each pelvis position
            sphere_radius = 0.02
            for frame_idx, pelvis_pos in enumerate(subsampled_pelvis):
                # Apply displacement to pelvis position
                displaced_pelvis = pelvis_pos + displacement
                
                # Create sphere mesh
                sphere = trimesh.creation.icosphere(radius=sphere_radius, subdivisions=1)
                sphere.vertices += displaced_pelvis
                
                # Determine alpha based on label name
                if source_label.endswith('_cam'):
                    alpha = 180  # Lower alpha for _cam methods
                elif source_label == 'ours':
                    alpha = 255  # Full opacity for our method (emphasized)
                elif source_label == 'gt':
                    alpha = 180  # Full opacity for ground truth
                else:
                    alpha = 180  # Lower alpha for comparison methods to de-emphasize them
                
                # Use same color as mesh but with transparency
                sphere_color = np.array(source_color[:3] + [alpha], dtype=np.uint8)
                vertex_colors = np.tile(sphere_color, (len(sphere.vertices), 1))
                sphere.visual.vertex_colors = vertex_colors
                
                # Create pyrender mesh and add to scene
                sphere_mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
                sphere_node = pyrender.Node(mesh=sphere_mesh, name=f'{source_name}_pelvis_{frame_idx}')
                scene.add_node(sphere_node)
        
        if save_frames:
            # Save frames mode: iterate through all frames once
            print("Saving frames for one complete pass...")
            
            for frame_idx in range(len(sorted_frames)):
                current_frame_name = sorted_frames[frame_idx]
                current_frame_nodes = []
                
                # Create and add meshes for current frame
                for source_idx, source_data in enumerate(multi_source_data):
                    frame_to_verts = source_data['frame_to_verts']
                    bm_faces = source_data['bm_faces']
                    source_color = source_data['color']
                    source_name = source_data['name']
                    displacement = source_data['displacement']
                    source_label = source_data.get('label', '')
                    
                    # Check if this source has data for current frame
                    if current_frame_name in frame_to_verts:
                        vertices = frame_to_verts[current_frame_name]
                        
                        # Apply spatial displacement
                        displaced_vertices = self.apply_displacement(vertices, displacement)
                        
                        # Create mesh with source-specific color
                        trimesh_obj = trimesh.Trimesh(
                            vertices=displaced_vertices, 
                            faces=bm_faces, 
                            process=False
                        )
                        
                        # Determine alpha based on label name
                        if source_label.endswith('_cam'):
                            alpha = 180  # Lower alpha for _cam methods
                        elif source_label == 'ours':
                            alpha = 255  # Full opacity for our method (emphasized)
                        elif source_label == 'gt':
                            alpha = 180  # Full opacity for ground truth
                        else:
                            alpha = 180  # Lower alpha for comparison methods to de-emphasize them
                        
                        frame_color = np.array(source_color[:3] + [alpha], dtype=np.uint8)
                        vertex_colors = np.tile(frame_color, (len(displaced_vertices), 1))
                        trimesh_obj.visual.vertex_colors = vertex_colors
                        
                        # Create pyrender mesh and add to scene
                        mesh = pyrender.Mesh.from_trimesh(trimesh_obj, smooth=False)
                        node = pyrender.Node(mesh=mesh, name=f'{source_name}_frame_{frame_idx}')
                        scene.add_node(node)
                        current_frame_nodes.append(node)
                
                # Render and save frame
                try:
                    color, depth = offscreen_renderer.render(scene)
                    frame_filename = f"frame_{frame_idx:05d}.png"
                    frame_path = os.path.join(save_dir, frame_filename)
                    
                    # Convert to PIL Image and save
                    import PIL.Image as pil_img
                    img = pil_img.fromarray(color)
                    img.save(frame_path)
                    
                    if frame_idx % 30 == 0:  # Print progress every 30 frames
                        print(f"Saved frame {frame_idx+1}/{len(sorted_frames)}: {frame_filename}")
                except Exception as e:
                    print(f"Error saving frame {frame_idx}: {e}")
                
                # Remove frame meshes for next iteration
                for node in current_frame_nodes:
                    if node in scene.nodes:
                        scene.remove_node(node)
            
            # Clean up offscreen renderer
            offscreen_renderer.delete()
            print(f"All {len(sorted_frames)} frames saved to: {save_dir}")
            
            # Generate video using ffmpeg
            self._generate_video(save_dir, len(sorted_frames))
            
        else:
            # Interactive viewer mode: continuous loop
            # Create viewer
            viewer = pyrender.Viewer(
                scene, 
                use_raymond_lighting=True, 
                run_in_thread=True,  # Non-blocking mode for animation
                record=False,
                viewport_size=self.viewport_size
            )
            
            # Print visualization info
            axis_info = {
                'y': "Y-axis up (XZ ground plane)",
                'z': "Z-axis up (XY ground plane)"
            }
            print(f"Multi-Source Animated 3D Visualization")
            print(f"Coordinate system: {axis_info[self.ground_axis]}")
            print(f"Showing {len(multi_source_data)} sources with spatial displacement")
            for i, source_data in enumerate(multi_source_data):
                source_label = source_data.get('label', '')
                alpha_info = " (reduced alpha)" if source_label.endswith('_cam') else ""
                print(f"  {i+1}. {source_data['name']}: {source_data['color'][:3]} RGB{alpha_info}, displacement {source_data['displacement']}")
            print("Controls:")
            print("- Mouse: Rotate camera around target")
            print("- Scroll: Zoom in/out")
            print("- Middle mouse + drag: Pan camera")
            print("- Space: Pause/Resume animation")
            print("- Close window to continue")
            
            # Animation loop with on-the-fly mesh creation
            frame_duration = 1.0 / self.fps
            current_frame_nodes = []  # Track current frame's mesh nodes
            
            frame_idx = 0
            paused = False
            last_frame_time = time.time()
            
            # Give viewer time to initialize properly
            time.sleep(0.5)
            
            print(f"Starting animation loop with {len(sorted_frames)} frames...")
            
            while viewer.is_active:
                current_time = time.time()
                
                # More robust frame timing - only update if enough time has passed
                if not paused and (current_time - last_frame_time) >= frame_duration:
                    
                    # CRITICAL: Use render lock to prevent race conditions
                    with viewer.render_lock:
                        # Remove previous frame's mesh nodes
                        nodes_to_remove = list(current_frame_nodes)  # Copy the list
                        
                        for node in nodes_to_remove:
                            if node in scene.nodes:
                                scene.remove_node(node)
                        current_frame_nodes.clear()
                        
                        # Create and add meshes for current frame ON-THE-FLY
                        current_frame_name = sorted_frames[frame_idx]
                        
                        for source_idx, source_data in enumerate(multi_source_data):
                            frame_to_verts = source_data['frame_to_verts']
                            bm_faces = source_data['bm_faces']
                            source_color = source_data['color']
                            source_name = source_data['name']
                            displacement = source_data['displacement']
                            source_label = source_data.get('label', '')
                            
                            # Check if this source has data for current frame
                            if current_frame_name in frame_to_verts:
                                vertices = frame_to_verts[current_frame_name]
                                
                                # Apply spatial displacement
                                displaced_vertices = self.apply_displacement(vertices, displacement)
                                
                                # Create mesh with source-specific color ON-THE-FLY
                                trimesh_obj = trimesh.Trimesh(
                                    vertices=displaced_vertices, 
                                    faces=bm_faces, 
                                    process=False
                                )
                                
                                # Determine alpha based on label name
                                if source_label.endswith('_cam'):
                                    alpha = 180  # Lower alpha for _cam methods
                                elif source_label == 'ours':
                                    alpha = 255  # Full opacity for our method (emphasized)
                                elif source_label == 'gt':
                                    alpha = 180  # Full opacity for ground truth
                                else:
                                    alpha = 180  # Lower alpha for comparison methods to de-emphasize them
                                
                                frame_color = np.array(source_color[:3] + [alpha], dtype=np.uint8)
                                vertex_colors = np.tile(frame_color, (len(displaced_vertices), 1))
                                trimesh_obj.visual.vertex_colors = vertex_colors
                                
                                # Create pyrender mesh and add to scene
                                mesh = pyrender.Mesh.from_trimesh(trimesh_obj, smooth=False)
                                node = pyrender.Node(mesh=mesh, name=f'{source_name}_frame_{frame_idx}')
                                scene.add_node(node)
                                current_frame_nodes.append(node)
                    
                    # Update frame index and timing
                    frame_idx = (frame_idx + 1) % len(sorted_frames)
                    last_frame_time = current_time
                
                # Check viewer status more frequently
                if not viewer.is_active:
                    print("Viewer became inactive, exiting animation loop")
                    break
                
                # Consistent sleep to prevent excessive CPU usage
                time.sleep(0.016)  # 60fps for responsiveness
            
            print("Animation loop completed")

    def _generate_video(self, save_dir: str, total_frames: int):
        """Generate video from saved frames using ffmpeg."""
        video_path = os.path.join(save_dir, "animation.mp4")
        
        # ffmpeg command to create video from PNG sequence
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # Overwrite output file if it exists
            '-r', str(self.fps),  # Input frame rate
            '-i', os.path.join(save_dir, 'frame_%05d.png'),  # Input pattern
            '-c:v', 'libx264',  # Video codec
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-crf', '18',  # Quality setting (lower = higher quality)
            video_path
        ]
        
        try:
            import subprocess
            print(f"Generating video with {total_frames} frames...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print(f"Video successfully created: {video_path}")
            
            # Print video info
            duration = total_frames / self.fps
            print(f"Video duration: {duration:.2f} seconds at {self.fps} FPS")
            
        except subprocess.CalledProcessError as e:
            print(f"Error generating video: {e}")
            print(f"ffmpeg stderr: {e.stderr}")
            print(f"Manual command: {' '.join(ffmpeg_cmd)}")
        except FileNotFoundError:
            print("ffmpeg not found. Please install ffmpeg to generate video.")
            print(f"Manual command: {' '.join(ffmpeg_cmd)}")

    def visualize_sequence_static(self, frame_to_verts: Dict, ordered_frames: List, 
                                bm_faces: np.ndarray, ground_height: float, 
                                cam2world: torch.Tensor, frame_to_verts_gt: Optional[Dict] = None, 
                                vis_mode: int = 3):
        """
        Entry point for static 3D visualization.
        
        Args:
            frame_to_verts: Dictionary mapping frame names to vertices
            ordered_frames: List of ordered frame names
            bm_faces: Body model faces
            ground_height: Height of the ground plane
            cam2world: Camera to world transformation
            frame_to_verts_gt: Ground truth vertices (optional)
            vis_mode: Visualization mode (1=pred only, 2=gt only, 3=both)
        """
        # Check if display is available
        if not self.check_display_available():
            print("No display available for static 3D visualization. Skipping...")
            return
        
        self.create_static_visualizer(
            frame_to_verts, ordered_frames, bm_faces, ground_height,
            cam2world, frame_to_verts_gt, vis_mode
        )

    def visualize_multi_source_animated(self, multi_source_data: List[Dict], 
                                       ground_height: float, cam2world: torch.Tensor, 
                                       ground_center: Optional[np.ndarray] = None,
                                       save_frames: bool = False, 
                                       save_dir: Optional[str] = None):
        """
        Entry point for multi-source animated 3D visualization.
        
        Args:
            multi_source_data: List of source data dictionaries
            ground_height: Height of the ground plane
            cam2world: Camera to world transformation
            ground_center: [3] XYZ center position for the ground plane (optional)
            save_frames: Whether to save individual frames as images
            save_dir: Directory to save frames (required if save_frames=True)
        """
        # Check if display is available
        if not self.check_display_available():
            print("No display available for animated 3D visualization. Skipping...")
            return
        
        self.create_multi_source_animated_visualizer(
            multi_source_data, ground_height, cam2world, ground_center,
            save_frames=save_frames, save_dir=save_dir
        )