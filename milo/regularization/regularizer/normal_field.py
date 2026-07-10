from typing import Dict, Any, Tuple, Optional, List, Callable
import numpy as np
import torch
from argparse import Namespace
from arguments import PipelineParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.geometry_utils import depth_to_normal, depth_to_normal_with_mask
from utils.general_utils import build_scaling_rotation, build_rotation
from gaussian_renderer import (
    render_depth,
    render_simp,
)


def initialize_normal_field(
    scene,
) -> Dict[str, Any]:
    normal_field_state = {}
    return normal_field_state


def compute_normal_field_regularization(
    viewpoint_cam: Camera, 
    gaussians: GaussianModel,
    median_depth: torch.Tensor,
    oriented_normal: torch.Tensor,
):
    view_to_world_transform = viewpoint_cam.world_view_transform[:3,:3].permute(-1, -2)
    normal_field_alignment_loss = torch.zeros(size=(), device=gaussians._xyz.device)
    median_depth_normal = depth_to_normal(
        viewpoint_cam,
        median_depth,
        None
    )
    median_depth_normal = (
        median_depth_normal.permute(1, 2, 0) @ view_to_world_transform
    ).permute(2, 0, 1)  # (3, H, W)
    normal_field_alignment_loss = normal_field_alignment_loss + (
        1. - (oriented_normal * median_depth_normal).sum(dim=0)
    ).mean() * 0.6
    normal_field_alignment_loss = normal_field_alignment_loss * 0.05
    return normal_field_alignment_loss


def get_gaussian_std_in_direction(
    directions: torch.Tensor,
    gaussians: Optional[GaussianModel]=None,
    gaussian_scaling: Optional[torch.Tensor]=None,
    gaussian_rotation: Optional[torch.Tensor]=None,
    normalize_directions: bool = True,
) -> torch.Tensor:
    """
    Get the standard deviation of the Gaussian in the given directions.
    
    If gaussians is provided, the scaling and rotation are extracted from the Gaussians.
    If gaussian_scaling and gaussian_rotation are provided, they are used instead of the Gaussians.
    
    Args:
        directions (torch.Tensor): A vector of shape (N_gaussians, n_directions, 3).
        gaussians (GaussianModel): The Gaussian model, with N_gaussians Gaussians.
        gaussian_scaling (torch.Tensor): The scaling of the Gaussians. Has shape (N_gaussians, 3).
        gaussian_rotation (torch.Tensor): The rotation of the Gaussians. Has shape (N_gaussians, 3, 3).
        normalize_directions (bool): Whether to normalize the directions.

    Returns:
        torch.Tensor: The standard deviation of the Gaussian in the direction of the given vector, of shape (N_gaussians, n_directions).
    """
    assert (
        gaussians is not None
        or (gaussian_scaling is not None and gaussian_rotation is not None)
    )
    
    # Get transposed scaled rotation
    if gaussian_scaling is None:
        gaussian_scaling = gaussians.get_scaling_with_3D_filter.detach()
    if gaussian_rotation is None:
        gaussian_rotation = gaussians._rotation.detach()
    transposed_scaled_rotation = build_scaling_rotation(
        s=gaussian_scaling,  # (N_gaussians, 3)
        r=gaussian_rotation,  # (N_gaussians, 3, 3)
    ).transpose(-1, -2)  # (N_gaussians, 3, 3)
    
    if normalize_directions:
        directions_to_use = torch.nn.functional.normalize(directions, dim=-1)
    else:
        directions_to_use = directions
    
    scaled_directions = torch.bmm(
        transposed_scaled_rotation,  # (N_gaussians, 3, 3)
        directions_to_use.permute(0, 2, 1),  # (N_gaussians, 3, n_directions)
    ).permute(0, 2, 1)  # (N_gaussians, n_directions, 3)
    
    direction_stds = scaled_directions.norm(dim=-1)  # (N_gaussians, n_directions)
    return direction_stds  # (N_gaussians, n_directions)


@torch.no_grad()
def compute_normal_error(
    gaussians: GaussianModel,
    cameras: List[Camera],
    render_func: Callable,
    pipe: PipelineParams,
    background: torch.Tensor = torch.zeros(3, device="cuda"),
    method: str ="area",  # "count" or "area" or "none"
    normal_to_use: str ="expected_depth",  # "rendered" or "median_depth" or "expected_depth"
    average_method_over_cameras: str ="all",  # "all" or "visible"
    mask_error_at_zero_depth: bool = True,
) -> torch.Tensor:
    """
    Compute, for each Gaussian, the average normal error between the Gaussian rendering and the Normal Field rendering.

    Args:
        gaussians (GaussianModel): The Gaussian model.
        cameras (List[Camera]): The cameras.
        render_func (Callable): The rendering function.
        pipe (PipelineParams): The pipeline parameters.
        background (torch.Tensor, optional): The background color. Defaults to torch.zeros(3, device="cuda").
        method (str, optional): The method to use for normalization. Defaults to "count".
        normal_to_use (str, optional): The normal to use for the error computation. Defaults to "median_depth".

    Raises:
        ValueError: If the method is not "count", "area" or "none".
        ValueError: If the normal_to_use is not "rendered", "median_depth" or "expected_depth".

    Returns:
        torch.Tensor: The average normal error between the Gaussian rendering and the Normal Field rendering. Has shape (N_gaussians,).
    """
    
    assert normal_to_use in ["rendered", "median_depth", "expected_depth"], "Invalid normal to use"
    assert method in ["count", "area", "none"], "Invalid method"
    assert average_method_over_cameras in ["all", "visible"], "Invalid average method over cameras"
    
    gaussian_errors = torch.zeros_like(gaussians._xyz[:, 0])
    # Number of visible cameras for each Gaussian
    if average_method_over_cameras == "visible":
        gaussian_visible_cameras = torch.zeros_like(gaussian_errors)
    
    for i_img in range(len(cameras)):
        # Get Gaussian idx
        msv2_render_pkg = render_depth(
            viewpoint_camera=cameras[i_img], 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background,
            culling=None
        )
        msv2_idx = msv2_render_pkg["gidx"]
        
        # Get projected Gaussian areas
        msv2_render_pkg_simp = render_simp(
            viewpoint_camera=cameras[i_img], 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background,
            culling=None
        )
        gaussians_proj_area = msv2_render_pkg_simp['area_proj']  # (N_gaussians,)
        
        # Get Gaussian Splatting rendering and rendered Normal Field
        gaussian_render_pkg = render_func(
            viewpoint_camera=cameras[i_img],
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            require_coord=False,
            require_depth=True,
            render_normal_field=True
        )
        normal_field_render = gaussian_render_pkg["normal_field"]  # (3, H, W)
        view_to_world_transform = cameras[i_img].world_view_transform[:3, :3].permute(-1, -2)
        
        # Select a normal map to compare the Normal Field rendering to
        if normal_to_use == "rendered":
            rendered_normal = gaussian_render_pkg["normal"]  # (3, H, W)
            error_mask = None
            rendered_normal = (rendered_normal.permute(1, 2, 0) @ view_to_world_transform).permute(2, 0, 1)  # (3, H, W)
            normal_render_to_use = rendered_normal  # (3, H, W)

        elif normal_to_use == "median_depth":
            median_depth = gaussian_render_pkg["median_depth"]  # (1, H, W)
            if mask_error_at_zero_depth:
                median_depth_normal, error_mask = depth_to_normal_with_mask(cameras[i_img], median_depth)  # (3, H, W) and (1, H, W)
                error_mask = error_mask.squeeze(0)  # (H, W)
            else:
                median_depth_normal = depth_to_normal(cameras[i_img], median_depth, None)  # (3, H, W)
                error_mask = None
            median_depth_normal = (median_depth_normal.permute(1, 2, 0) @ view_to_world_transform).permute(2, 0, 1)  # (3, H, W)
            normal_render_to_use = median_depth_normal  # (3, H, W)

        elif normal_to_use == "expected_depth":
            expected_depth = gaussian_render_pkg["expected_depth"]  # (1, H, W)
            if mask_error_at_zero_depth:
                expected_depth_normal, error_mask = depth_to_normal_with_mask(cameras[i_img], expected_depth)  # (3, H, W) and (1, H, W)
                error_mask = error_mask.squeeze(0)  # (H, W)
            else:
                expected_depth_normal = depth_to_normal(cameras[i_img], expected_depth, None)  # (3, H, W)
                error_mask = None
            expected_depth_normal = (expected_depth_normal.permute(1, 2, 0) @ view_to_world_transform).permute(2, 0, 1)  # (3, H, W)
            normal_render_to_use = expected_depth_normal  # (3, H, W)

        else:
            raise ValueError(f"Invalid normal to use: {normal_to_use}")

        # Compute normal error
        normal_error = 1. - (normal_field_render * normal_render_to_use).sum(dim=0)  # (H, W)
        if mask_error_at_zero_depth and (error_mask is not None):
            if i_img == 0:
                print("[INFO] Masking error at zero depth.")
            normal_error = torch.where(error_mask, normal_error, torch.zeros_like(normal_error))
        
        # Compute per-Gaussian error
        gaussian_errors_i = torch.zeros_like(gaussians_proj_area, dtype=torch.float32)  # (N_gaussians,)
        gaussian_errors_i.index_add_(0, msv2_idx.flatten(), normal_error.flatten())
        
        # If count, we normalize by the number of pixels in which the Gaussian is visible
        if method == "count":
            gaussian_count = torch.zeros_like(gaussians_proj_area, dtype=torch.float32)
            gaussian_count.index_add_(0, msv2_idx.flatten(), torch.ones_like(normal_error.flatten(), dtype=torch.float32))
            
            valid_mask = gaussian_count > 0
            gaussian_errors_i = torch.where(valid_mask, gaussian_errors_i / gaussian_count, torch.zeros_like(gaussian_errors_i))
        
        # If area, we normalize by the area of the projected Gaussian splat
        elif method == "area":
            valid_area_mask = gaussians_proj_area > 0
            gaussian_errors_i = torch.where(valid_area_mask, gaussian_errors_i / gaussians_proj_area, torch.zeros_like(gaussian_errors_i))
        
        # If none, we don't normalize
        elif method == "none":
            pass
        
        else:
            raise ValueError(f"Invalid method: {method}")
        
        gaussian_errors = gaussian_errors + gaussian_errors_i
        
        if average_method_over_cameras == "visible":
            visible_gaussian_idx = msv2_idx.unique()
            gaussian_visible_cameras.index_add_(0, visible_gaussian_idx, torch.ones_like(visible_gaussian_idx, dtype=torch.float32))
    
    if average_method_over_cameras == "all":
        gaussian_errors = gaussian_errors / len(cameras)
    elif average_method_over_cameras == "visible":
        gaussian_errors = torch.where(gaussian_visible_cameras > 0, gaussian_errors / gaussian_visible_cameras, torch.zeros_like(gaussian_errors))
    else:
        raise ValueError(f"Invalid average method over cameras: {average_method_over_cameras}")
    
    return gaussian_errors


@torch.no_grad()
def densify_normal_field(
    gaussians: GaussianModel, 
    cameras: List[Camera],
    pipe: PipelineParams, 
    background: torch.Tensor, 
    render_func: Callable, 
    args: Namespace,
    maintain_constant_volume: bool = True,
):
    # Get Gaussian normals
    gaussian_normals = gaussians.convert_features_to_normals()  # (N_gaussians, 3)
    gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)  # (N_gaussians, 3)
    
    # Compute normal errors
    normal_errors = compute_normal_error(
        gaussians=gaussians,
        cameras=cameras,
        render_func=render_func,
        pipe=pipe,
        background=background,
        method="count",                # "count" or "area" or "none"
        normal_to_use="median_depth",  # "rendered" or "median_depth" or "expected_depth"
    )  # (N_gaussians,)
    
    # Compute normal errors quantile
    densification_normal_errors_quantile = 0.05 # Tried 0.05. 0.1 works well. Percentage of the highest normal errors to use for densification
    normal_errors_quantile = torch.quantile(normal_errors, q=1. - densification_normal_errors_quantile)
    
    # Densification mask
    densification_mask = normal_errors > normal_errors_quantile  # (N_gaussians,)

    # If N_max_gaussians is set, cap the number of new Gaussians
    if getattr(args, 'N_max_gaussians', None) is not None:
        n_current = gaussians._xyz.shape[0]
        n_allowed = args.N_max_gaussians - n_current
        if n_allowed <= 0:
            print("[WARNING] Maximum Number of Gaussians reached. Skipping Densification.")
            return  # Already at or above cap, skip densification entirely
        n_selected = densification_mask.sum().item()
        if n_selected > n_allowed:
            # Keep only the top n_allowed Gaussians by normal error
            candidate_indices = densification_mask.nonzero(as_tuple=True)[0]
            top_indices = candidate_indices[normal_errors[candidate_indices].topk(n_allowed).indices]
            densification_mask = torch.zeros_like(densification_mask)
            densification_mask[top_indices] = True
            print(f"[WARNING] Capping the number of gaussians to {args.N_max_gaussians}.")

    # Adjust scale of Gaussians to be densified. The idea is to divide the volume of the densified Gaussian by 2,
    # while taking into account the direction of the normal.
    if maintain_constant_volume:
        #   > First, we compute the local basis of the Gaussian
        local_basis = build_rotation(
            r=gaussians._rotation[densification_mask]  # (N_gaussians_to_densify, 3, n_vectors_in_basis)
        ).transpose(-1, -2)  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
        
        #   > Then, we compute the projections of the normals on the local basis
        projections_on_local_basis = (
            gaussian_normals[densification_mask].unsqueeze(1)  # (N_gaussians_to_densify, 1, 3)
            * local_basis  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
        ).sum(dim=-1)  # (N_gaussians_to_densify, n_vectors_in_basis)
        
        #   > We compute the logarithm of the adjustment factors
        log_adjustment_factors = np.log(1. / 2.) * projections_on_local_basis ** 2
        
        #   > Adjust the scaling of the Gaussians
        gaussians._scaling[densification_mask] = gaussians._scaling[densification_mask] + log_adjustment_factors
    
    # Compute xyz of cloned Gaussians as same xyz minus a small multiple of the normal
    new_xyz = gaussians._xyz[densification_mask]  # (N_new_gaussians, 3)
    new_normals = - gaussian_normals[densification_mask]  # (N_new_gaussians, 3)
    normal_stds = get_gaussian_std_in_direction(
        directions=new_normals.unsqueeze(1),  # (N_new_gaussians, 1, 3)
        gaussian_scaling=gaussians.get_scaling_with_3D_filter[densification_mask].detach(), 
        gaussian_rotation=gaussians._rotation[densification_mask].detach(),
        normalize_directions=False,
    )  # (N_gaussians, 1)
    # FIXME: What is the best factor to use here?
    delta = 0.1
    # delta = 1.0
    # delta = np.sqrt(3.)
    # new_xyz = new_xyz + 0.01 * normal_stds * new_normals
    # new_xyz = new_xyz + 1. * normal_stds * new_normals
    # new_xyz = new_xyz + 3. * normal_stds * new_normals
    # new_xyz = new_xyz + 0.05 * normal_stds * new_normals  # best so far?
    new_xyz = new_xyz + delta * normal_stds * new_normals
    
    # Compute normal features of cloned Gaussians to obtain the opposite normal
    new_gaussian_features = gaussians._gaussian_features[densification_mask]  # (N_new_gaussians, n_features)
    new_gaussian_features[:, -1:] = -new_gaussian_features[:, -1:]
    
    # Update xyz of densified Gaussians to be xyz plus a small multiple of the normal
    gaussians._xyz[densification_mask] = (
        gaussians._xyz[densification_mask]
        + delta * normal_stds * gaussian_normals[densification_mask]
    )
    
    # Densify Gaussians
    gaussians.densify_and_clone_from_mask(
        selected_pts_mask=densification_mask,
        new_xyz=new_xyz,
        new_gaussian_features=new_gaussian_features,
    )

    if gaussians.use_mip_filter:
        gaussians.compute_3D_filter(cameras)

