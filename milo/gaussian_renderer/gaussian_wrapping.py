from typing import Optional
import math
import torch
from diff_gaussian_rasterization_ours import GaussianRasterizationSettings, GaussianRasterizer
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel


def sample_depth_gaussian_wrapping(points3D, viewpoint_camera: Camera, pc : GaussianModel, pipe : torch.Tensor, kernel_size : float, scaling_modifier = 1.0):

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    active_sg_degree = 0

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth, inside = rasterizer.sample_depth(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"sampled_depth": depth,
            "inside": inside}
