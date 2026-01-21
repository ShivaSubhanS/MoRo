import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch3d.renderer.cameras import look_at_view_transform

from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras,
    DirectionalLights,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex,
)

from utils.other_utils import LIMBS_BODY_SMPL
from utils.vis3d_utils import color_dict
from PIL import Image


def get_colors_from_diff_pc(diff_pc, min_error, max_error, color="coolwarm"):
    """
    Adapted from https://github.com/papagina/MeshConvolution/blob/master/code/GraphAE/graphAE_test.py
    Gives the color of each vertex given its positional error.
    """
    b, n = diff_pc.shape
    colors = np.zeros((b, n, 3))
    mix = (diff_pc - min_error) / (max_error - min_error)
    mix = np.clip(mix, 0, 1)  # point_num
    cmap = plt.cm.get_cmap(color)
    colors = cmap(mix)[:, :, 0:3]
    return colors


def get_colors_from_meshes(groundtruth_mesh, prediction, min_error, max_error):
    """
    Gives the color of each vertex given the prediction and the groundtruth mesh.
    """
    groundtruth_mesh = groundtruth_mesh - torch.mean(
        groundtruth_mesh, axis=1, keepdims=True
    )
    prediction = prediction - torch.mean(prediction, axis=1, keepdims=True)
    diff_mesh = (
        1000 * torch.norm(groundtruth_mesh - prediction, dim=-1).detach().cpu().numpy()
    )
    colored_mesh = get_colors_from_diff_pc(
        diff_mesh,
        min_error,
        max_error,
    )
    return colored_mesh


def create_ground_mesh(ground_height, x_range, z_range, tile_size=0.5, padding=0.2):
    """
    Create a checkerboard ground mesh from XZ range.
    
    Args:
        ground_height: Height of the ground plane (Y coordinate)
        x_range: (x_min, x_max) tuple
        z_range: (z_min, z_max) tuple
        tile_size: Size of each checkerboard tile
        padding: Padding factor (e.g., 0.2 = 20% padding on each side)
        
    Returns:
        trimesh.Trimesh: Ground mesh with checkerboard pattern
    """
    # Compute ground size and center from range
    x_min, x_max = x_range
    z_min, z_max = z_range
    
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2
    
    x_span = x_max - x_min
    z_span = z_max - z_min
    max_span = max(x_span, z_span)
    
    # Add padding
    ground_size = max_span * (1 + 2 * padding)
    half_size = ground_size / 2.0
    
    # Checkerboard colors
    color0 = np.array([0.85, 0.85, 0.85, 1.0])  # light gray
    color1 = np.array([0.4, 0.4, 0.4, 1.0])  # Dark gray
    
    # Calculate grid
    num_tiles = int(np.ceil(ground_size / tile_size))
    
    vertices = []
    faces = []
    vertex_colors = []
    
    for i in range(num_tiles):
        for j in range(num_tiles):
            # Calculate tile position (centered at x_center, z_center)
            x_start = x_center - half_size + j * tile_size
            z_start = z_center + half_size - i * tile_size
            
            # Create 4 vertices for this tile
            tile_verts = np.array([
                [x_start, ground_height, z_start],
                [x_start, ground_height, z_start - tile_size],
                [x_start + tile_size, ground_height, z_start - tile_size],
                [x_start + tile_size, ground_height, z_start],
            ])
            
            # Determine color (checkerboard pattern)
            use_color0 = (i + j) % 2 == 0
            tile_color = color0 if use_color0 else color1
            
            # Add vertices and colors
            base_idx = len(vertices)
            vertices.extend(tile_verts)
            vertex_colors.extend([tile_color] * 4)
            
            # Add two triangles for this tile
            faces.extend([
                [base_idx, base_idx + 1, base_idx + 3],
                [base_idx + 1, base_idx + 2, base_idx + 3]
            ])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    vertex_colors = (np.array(vertex_colors) * 255).astype(np.uint8)
    
    ground_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False
    )
    
    return ground_mesh


def setup_global_camera(human_center, cam_distance=4.0, cam_height=2.0):
    """
    Setup camera for global view using PyTorch3D's look_at_view_transform.
    
    Args:
        human_center: [3] center position of human in world space
        cam_distance: Horizontal distance from human center
        cam_height: Height above human center
        
    Returns:
        tuple: (R_global, T_global) rotation matrix and translation vector as numpy arrays
    """
    # Camera position: behind and above the human
    cam_pos = np.array([
        human_center[0],                          # Centered on X
        human_center[1] + cam_height,              # Above human
        human_center[2] + cam_distance             # Behind human (+Z)
    ])
    
    # Use PyTorch3D's look_at_view_transform
    # This function returns R and T such that: p_cam = p_world @ R + T
    R, T = look_at_view_transform(
        eye=[cam_pos],
        at=[human_center],
        up=[[0, 1, 0]],  # Y-up
    )
    
    # Convert to numpy
    R_global = R[0].cpu().numpy()  # [3, 3]
    T_global = T[0].cpu().numpy()  # [3]
    
    return R_global, T_global


class BatchRenderer(nn.Module):
    def __init__(self, K, img_w=512, img_h=512, faces=None, mesh_color="skin"):
        super(BatchRenderer, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.faces = faces
        self.K = K
        self.max_batch_size = K.shape[0]
        self.mesh_color = mesh_color

        self.raster_settings = RasterizationSettings(
            image_size=(img_h, img_w), 
            blur_radius=0.0, 
            bin_size=0, 
        )

        self.lights = DirectionalLights(
            direction=[[0, 0, -1]],
            ambient_color=[[0.3, 0.3, 0.3]],
            diffuse_color=[[0.7, 0.7, 0.7]],
            specular_color=[[0.0, 0.0, 0.0]],
        )

        self.lights_global = DirectionalLights(
            direction=[[0, -1, 0]],  # Light from above
            ambient_color=[[0.7, 0.7, 0.7]],
            diffuse_color=[[0.3, 0.3, 0.3]],
            specular_color=[[0.0, 0.0, 0.0]],
        )

        # One shared camera
        cameras = self._build_camera()

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                cameras=cameras, lights=self.lights
            ),
        )

        # Color definitions
        self.color_dict = color_dict 
        
        self.color_skel_vis = (90 / 255, 135 / 255, 247 / 255)

    def _get_mesh_color(self):
        """Get RGB color tuple for the specified mesh color."""
        return self.color_dict.get(self.mesh_color, self.color_dict["skin"])

    def to(self, device):
        self.renderer.to(device)
        return self 

    def _build_camera(self):
        fx = self.K[:, 0, 0]
        fy = self.K[:, 1, 1]
        cx = self.K[:, 0, 2]
        cy = self.K[:, 1, 2]

        return PerspectiveCameras(
            focal_length=torch.stack((fx, fy), dim=1),
            principal_point=torch.stack((cx, cy), dim=1),
            image_size=((self.img_h, self.img_w),) * self.K.shape[0],
            in_ndc=False,
            device=self.K.device,
        )

    def _create_skel(self, joints):
        device = joints.device
        joints = joints.cpu().numpy()
        skeleton_mesh_list = []
        for j in range(22):
            sphere = trimesh.creation.icosphere(radius=0.025)
            transformation = np.identity(4)
            transformation[:3, 3] = joints[j]
            sphere.apply_transform(transformation)
            color = self.color_skel_vis
            color = torch.tensor(color, dtype=torch.float32, device=device)
            color = color.unsqueeze(0).expand(sphere.vertices.shape[0], 3)  # (V, 3)
            textures = TexturesVertex(verts_features=color.unsqueeze(0))
            sphere_mesh = Meshes(
                verts=torch.tensor(sphere.vertices, dtype=torch.float32, device=device).unsqueeze(0),
                faces=torch.tensor(sphere.faces, dtype=torch.int32, device=device).unsqueeze(0),
                textures=textures,
            )
            skeleton_mesh_list.append(sphere_mesh)

        for index_pair in LIMBS_BODY_SMPL:
            p1 = joints[index_pair[0]]
            p2 = joints[index_pair[1]]
            segment = np.array([p1, p2])
            cyl = trimesh.creation.cylinder(0.01, height=None, segment=segment)
            color = self.color_skel_vis
            color = torch.tensor(color, dtype=torch.float32, device=device)
            color = color.unsqueeze(0).expand(cyl.vertices.shape[0], 3)  # (V, 3)
            textures = TexturesVertex(verts_features=color.unsqueeze(0))
            cyl_mesh = Meshes(
                verts=torch.tensor(cyl.vertices, dtype=torch.float32, device=device).unsqueeze(0),
                faces=torch.tensor(cyl.faces, dtype=torch.int32, device=device).unsqueeze(0),
                textures=textures,
            )
            skeleton_mesh_list.append(cyl_mesh)
        return skeleton_mesh_list

    def render_global(self, verts_world, ground_mesh, R_global, T_global):
        """
        Render global view with pre-computed ground mesh and camera transform.
        
        Args:
            verts_world: [N, V, 3] body vertices in world space
            ground_mesh: trimesh.Trimesh pre-computed ground plane mesh
            R_global: [3, 3] rotation matrix (numpy or tensor)
            T_global: [3] translation vector (numpy or tensor)
            
        Returns:
            np.ndarray: Rendered images [N, H, W, 4]
        """
        device = verts_world.device
        N = verts_world.shape[0]
        
        # Convert camera matrices to tensors if needed
        if isinstance(R_global, np.ndarray):
            R_global = torch.from_numpy(R_global).float().to(device)
        if isinstance(T_global, np.ndarray):
            T_global = torch.from_numpy(T_global).float().to(device)
        
        # Transform body verts to camera space: p_cam = p_world @ R + T
        verts_cam = verts_world @ R_global.T + T_global
        
        # Transform ground mesh to camera space
        ground_verts_world = torch.from_numpy(ground_mesh.vertices).float().to(device)
        ground_verts_cam = ground_verts_world @ R_global.T + T_global
        
        # Expand ground for batch
        ground_verts_batch = ground_verts_cam.unsqueeze(0).expand(N, -1, -1)
        
        # Combine body and ground
        combined_verts = torch.cat([verts_cam, ground_verts_batch], dim=1)
        
        # Combine faces
        n_body_verts = verts_cam.shape[1]
        body_faces = self.faces
        ground_faces = torch.from_numpy(ground_mesh.faces).long().to(device) + n_body_verts
        combined_faces = torch.cat([body_faces, ground_faces], dim=0)
        
        # Get colors - body uses mesh_color, ground uses its vertex colors
        body_color = self._get_mesh_color()
        body_colors = torch.tensor(body_color, dtype=torch.float32, device=device).view(1, 1, 3).expand(N, n_body_verts, 3)
        
        # Extract ground colors from trimesh (normalize to [0, 1])
        ground_vertex_colors = torch.from_numpy(ground_mesh.visual.vertex_colors[:, :3] / 255.0).float().to(device)
        ground_colors_batch = ground_vertex_colors.unsqueeze(0).expand(N, -1, -1)
        
        # Combine colors
        combined_colors = torch.cat([body_colors, ground_colors_batch], dim=1)
        
        # Render with combined mesh and colors
        imgs = self._render_with_colors(combined_verts, combined_colors, combined_faces)
        
        return imgs

    def _render_with_colors(self, verts, vertex_colors, faces):
        """Render with per-vertex colors and custom faces."""
        device = verts.device
        N = verts.shape[0]
        
        # Handle variable batch sizes
        if N != self.max_batch_size:
            K_subset = self.K[:N]
            cameras = PerspectiveCameras(
                focal_length=torch.stack((K_subset[:, 0, 0], K_subset[:, 1, 1]), dim=1),
                principal_point=torch.stack((K_subset[:, 0, 2], K_subset[:, 1, 2]), dim=1),
                image_size=((self.img_h, self.img_w),) * N,
                in_ndc=False,
                device=device,
            )
            
            temp_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, raster_settings=self.raster_settings
                ),
                shader=HardPhongShader(
                    cameras=cameras, lights=self.lights_global
                ),
            )
            temp_renderer.to(device)
            current_renderer = temp_renderer
        else:
            current_renderer = self.renderer

        # Use provided colors
        textures = TexturesVertex(verts_features=vertex_colors)

        # Create per-mesh objects with custom faces
        mesh_list = [
            Meshes(verts=[verts[i]], faces=[faces], textures=textures[i : i + 1])
            for i in range(N)
        ]
        scene_mesh = join_meshes_as_batch(mesh_list)
        
        # Override lighting with global lights
        mesh_img = current_renderer(scene_mesh, lights=self.lights_global.to(device))

        mesh_img = mesh_img.detach().cpu().numpy()
        mesh_img = (mesh_img * 255).astype(np.uint8)
        return mesh_img

    def forward(self, verts, joints=None, bg_img_rgb=None, render_skel=False):
        """
        verts: Tensor of shape (N, V, 3)
        bg_img_rgb: Optional numpy array of shape (H, W, 3)
        """
        device = verts.device
        N = verts.shape[0]
        
        # Handle variable batch sizes by using only the required cameras
        if N != self.max_batch_size:
            # Create a temporary renderer with the correct batch size
            K_subset = self.K[:N]
            cameras = PerspectiveCameras(
                focal_length=torch.stack((K_subset[:, 0, 0], K_subset[:, 1, 1]), dim=1),
                principal_point=torch.stack((K_subset[:, 0, 2], K_subset[:, 1, 2]), dim=1),
                image_size=((self.img_h, self.img_w),) * N,
                in_ndc=False,
                device=device,
            )
            
            temp_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, raster_settings=self.raster_settings
                ),
                shader=HardPhongShader(
                    cameras=cameras, lights=self.lights
                ),
            )
            temp_renderer.to(device)
            current_renderer = temp_renderer
        else:
            current_renderer = self.renderer

        # transform to pytorch3d coordinate system
        R = torch.tensor(
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=verts.dtype, device=verts.device
        )
        verts = verts @ R.T

        mesh_color_rgb = self._get_mesh_color()
        vertex_colors = torch.tensor(
            mesh_color_rgb, dtype=torch.float32, device=verts.device)
        vertex_colors = vertex_colors.view(1, 1, 3).expand(N, verts.shape[1], 3)
        textures = TexturesVertex(verts_features=vertex_colors)

        # Create per-mesh objects
        mesh_list = [
            Meshes(verts=[verts[i]], faces=[self.faces], textures=textures[i : i + 1])
            for i in range(N)
        ]
        scene_mesh = join_meshes_as_batch(mesh_list)
        # Combine into a single scene
        mesh_img = current_renderer(scene_mesh) 

        # if render_skel:
        #     assert joints is not None, "Joints must be provided if render_skel is True"
        #     # Create skeleton meshes
        #     joints = joints @ R.T
        #     skel_list = []
        #     for i in range(N):
        #         skel_list.extend(self._create_skel(joints[i]))
        #     scene_skel = join_meshes_as_scene(skel_list)
        #     skel_img = self._render_img(self.renderer, scene_skel, alpha=2.0)

        mesh_img = mesh_img.detach().cpu().numpy()
        mesh_img = (mesh_img * 255).astype(np.uint8)
        return mesh_img
