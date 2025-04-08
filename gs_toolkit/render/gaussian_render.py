import torch
import math

import numpy as np

from typing import Tuple

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from taichi_3d_ellipsoid import EllipsoidRasterizationRenderer, EllipsoidRenderer
from ..model.gaussian_model import GaussianModel
from ..model.camera_model import Camera
from ..utils.graphics_utils import sh_features_to_colors, quaternion_to_rotation_matrix
from ..utils.general_utils import convert_gs_camera_to_cv2_camera

def render(viewpoint_camera : Camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out

def build_ellipsoid_renderer(viewpoint_camera : Camera, pc : GaussianModel, bg_color : torch.Tensor):
    """
    Build an ellipsoid renderer for the scene.
    """
    resolution = (viewpoint_camera.image_width, viewpoint_camera.image_height) # (width, height)
    fov = viewpoint_camera.FoVx * 180.0 / math.pi  # radian to degree
    centers = pc.get_xyz
    radii = pc.get_scaling
    radii = radii * 2.0  # axis-length of each ellipsoid. (reference: https://blog.42yeah.is/rendering/opengl/2023/12/20/rasterizing-splats.html)
    colors = sh_features_to_colors(pc.get_features_dc).squeeze(1)
    rotations = quaternion_to_rotation_matrix(pc.get_rotation)
    opacities = pc.get_opacity.squeeze(1)
    background_color = list(bg_color.tolist())

    renderer = EllipsoidRasterizationRenderer(
        centers=centers,
        radii=radii,
        colors=colors,
        rotations=rotations,
        opacities=opacities,
        arr_type="torch", 
        res=resolution,
        fov=fov,
        background_color=background_color,
        opacity_limit=0.2,
        specular_strength=0.0,
        headless=True,
        device="cuda:0",
    )

    return renderer

def render_ellipsoid(viewpoint_camera : Camera, renderer: EllipsoidRenderer):
    c2w, _ = convert_gs_camera_to_cv2_camera(viewpoint_camera)
    # camera position in the world coordinate system
    camera_pos = c2w[:3, 3]

    # the forward direction is the z direction in OpenCV camera, 
    forward_dir = c2w[:3, 2]
    camera_lookat = camera_pos + forward_dir * 1.0

    # the up direction is the -y direction in OpenCV camera
    # camera_up is the up vector of the camera in the world coordinate system
    camera_up = -c2w[:3, 1]

    result = renderer.render_image(
        output_path=None,
        camera_pos=camera_pos,
        camera_lookat=camera_lookat,
        camera_up=camera_up  
    ) # [W, H, C]
    result = result.permute(2, 1, 0) # [C, H, W]
    # we need to flip the image vertically to deal with the difference between taichi and torchvision
    result = result.flip(1)
    return result
