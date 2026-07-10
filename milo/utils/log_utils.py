import os
from typing import List, Union, Dict, Any
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.geometry_utils import depth_to_normal
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, L1_loss_appearance
from lpipsPyTorch import lpips
try:
    import wandb
except ImportError:
    pass


def fix_normal_map(view, normal, normal_in_view_space=True):
    """_summary_

    Args:
        view (_type_): _description_
        normal (_type_): 

    Returns:
        _type_: _description_
    """
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = (intrins_inv @ points).reshape(3, H, W)
    
    if normal_in_view_space:
        normal_view = normal
    else:
        normal_view = normal.clone()
        if normal.shape[0] == 3:
            normal_view = normal_view.permute(1, 2, 0)
        normal_view = normal_view @ view.world_view_transform[:3,:3]
        if normal.shape[0] == 3:
            normal_view = normal_view.permute(2, 0, 1)
    
    if normal_view.shape[0] != 3:
        rays_d = rays_d.permute(1, 2, 0)
        dim_to_sum = -1
    else:
        dim_to_sum = 0
        
    return torch.sign((-rays_d * normal_view).sum(dim=dim_to_sum, keepdim=True)) * normal_view


def make_log_figure(
    images:List[torch.Tensor],
    nrows:Union[int, None]=None,
    ncols:Union[int, None]=None,
    titles:Union[List[str], None]=None,
    figsize:int=30,
    cmap:Union[str, List[str]]="Spectral",
    show_plot:bool=False,
    save_plot:bool=False,
    save_path:Union[str, None]=None,
    return_log_images:bool=False,
):
    log_images = []
    for image in images:
        image_to_add = image
        if (image.ndim == 3) and (image.shape[0] == 1 or image.shape[0] == 3):
            image_to_add = image.permute(1, 2, 0)
        log_images.append(image_to_add.cpu())
    
    n_images = len(log_images)
    if nrows is None and ncols is None:
        raise ValueError("nrows and ncols cannot be None at the same time")
    elif nrows is None:
        nrows = round(np.ceil(float(n_images) / float(ncols)))
    elif ncols is None:
        ncols = round(np.ceil(float(n_images) / float(nrows)))
    
    height, width = log_images[0].shape[:2]    
    plt.figure(figsize=(figsize, figsize * height / width * nrows / ncols))
    
    for i, log_image in enumerate(log_images):
        plt.subplot(nrows, ncols, i+1)
        if log_image.shape[-1] == 3:
            plt.imshow(log_image.clamp(min=0, max=1))
        else:
            if type(cmap) == list:
                plt.imshow(log_image, cmap=cmap[i])
            else:
                if log_image.dtype == torch.bool:
                    plt.imshow(log_image, cmap="gray")
                else:
                    plt.imshow(log_image, cmap=cmap)
            plt.colorbar()
        if titles is not None:
            plt.title(titles[i])
    
    if show_plot:
        plt.show()
    if save_plot and (save_path is not None):
        plt.savefig(save_path)
    plt.close()
    
    if return_log_images:
        log_images_dict = {}
        for i, log_image in enumerate(log_images):
            if titles is not None:
                log_images_dict[titles[i]] = log_image
            else:
                log_images_dict[f"image_{i}"] = log_image
        return log_images_dict


def training_report(iteration, l1_loss, testing_iterations, scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)        

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))                    


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 

                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")

        torch.cuda.empty_cache()


# TODO: Logging function should be made clearer and more modular.
def log_training_progress(
    args, iteration:int, log_interval:int, progress_bar, run,
    # Objects
    scene, gaussians, pipe, opt, background,
    # Rendering results,
    viewpoint_idx, viewpoint_cam, render_pkg, mesh_render_pkg, do_supervision_depth,
    # Flags
    reg_kick_on:bool, mesh_kick_on:bool, depth_order_kick_on:bool,
    # Losses
    loss:torch.Tensor, depth_normal_loss:torch.Tensor, mesh_depth_loss:torch.Tensor,
    mesh_normal_loss:torch.Tensor, occupied_centers_loss:torch.Tensor, occupancy_labels_loss:torch.Tensor,
    depth_prior_loss:torch.Tensor,
    # Configs
    mesh_config:Dict[str, Any],
    # EMA losses for logging
    postfix_dict:Dict[str, Any], ema_loss_for_log:float, ema_semantic_loss_for_log:float, ema_depth_normal_loss_for_log:float,
    ema_mesh_depth_loss_for_log:float, ema_mesh_normal_loss_for_log:float,
    ema_occupied_centers_loss_for_log:float, ema_occupancy_labels_loss_for_log:float,
    ema_depth_order_loss_for_log:float,
    # Additional arguments
    testing_iterations:List[int], saving_iterations:List[int], render_imp, semantic_loss:torch.Tensor=None,
    monosdf_loss:torch.Tensor=None, scale_factor=None,
):
    WANDB_FOUND = run is not None
    
    # ---Progress bar---
    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    if reg_kick_on:
        ema_depth_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_depth_normal_loss_for_log
    if mesh_kick_on:
        ema_mesh_depth_loss_for_log = 0.4 * mesh_depth_loss.item() + 0.6 * ema_mesh_depth_loss_for_log
        ema_mesh_normal_loss_for_log = 0.4 * mesh_normal_loss.item() + 0.6 * ema_mesh_normal_loss_for_log
        if mesh_config["enforce_occupied_centers"]:
            ema_occupied_centers_loss_for_log = 0.4 * occupied_centers_loss.item() + 0.6 * ema_occupied_centers_loss_for_log
        if mesh_config["use_occupancy_labels_loss"]:
            ema_occupancy_labels_loss_for_log = 0.4 * occupancy_labels_loss.item() + 0.6 * ema_occupancy_labels_loss_for_log
    if depth_order_kick_on:
        ema_depth_order_loss_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_depth_order_loss_for_log

    # Update raw float values in postfix_dict for logging (e.g. to WandB)
    postfix_dict["Loss"] = ema_loss_for_log
    if reg_kick_on:
        postfix_dict["DNLoss"] = ema_depth_normal_loss_for_log
    else:
        postfix_dict.pop("DNLoss", None)
        
    if depth_order_kick_on:
        postfix_dict["DOLoss"] = ema_depth_order_loss_for_log
    else:
        postfix_dict.pop("DOLoss", None)
        
    if monosdf_loss is not None:
        ema_ms = postfix_dict.get("MSNLoss", 0.0)
        ema_ms = 0.4 * monosdf_loss.item() + 0.6 * ema_ms
        postfix_dict["MSNLoss"] = ema_ms
    else:
        postfix_dict.pop("MSNLoss", None)
        
    if scale_factor is not None:
        val = scale_factor.item() if isinstance(scale_factor, torch.Tensor) else scale_factor
        ema_sf = postfix_dict.get("ScaleFac", 0.0)
        ema_sf = 0.4 * val + 0.6 * ema_sf
        postfix_dict["ScaleFac"] = ema_sf
    else:
        postfix_dict.pop("ScaleFac", None)
        
    if mesh_kick_on:
        postfix_dict["MDLoss"] = ema_mesh_depth_loss_for_log
        postfix_dict["MNLoss"] = ema_mesh_normal_loss_for_log
        if mesh_config and mesh_config.get("enforce_occupied_centers", False):
            postfix_dict["OccLoss"] = ema_occupied_centers_loss_for_log
        else:
            postfix_dict.pop("OccLoss", None)
        if mesh_config and mesh_config.get("use_occupancy_labels_loss", False):
            postfix_dict["OccLabLoss"] = ema_occupancy_labels_loss_for_log
        else:
            postfix_dict.pop("OccLabLoss", None)
    else:
        postfix_dict.pop("MDLoss", None)
        postfix_dict.pop("MNLoss", None)
        postfix_dict.pop("OccLoss", None)
        postfix_dict.pop("OccLabLoss", None)
    
    if iteration % 10 == 0:
        postfix_dict_display = {}
        for key in ["Loss", "DNLoss", "DOLoss", "MSNLoss", "ScaleFac", "MDLoss", "MNLoss", "OccLoss", "OccLabLoss"]:
            if key in postfix_dict:
                postfix_dict_display[key] = f"{postfix_dict[key]:.{7}f}"
        postfix_dict_display["N_Gauss"] = f"{gaussians._xyz.shape[0]}"
        progress_bar.set_postfix(postfix_dict_display)
        progress_bar.update(10)
        
    if iteration == opt.iterations:
        progress_bar.close()
    
    # ---Logging---
    if (log_interval is not None) and (
        (iteration % log_interval == 0) or (mesh_kick_on and iteration == mesh_config["start_iter"])
    ):
        images_to_log, titles_to_log = [], []
        
        if depth_order_kick_on:
            images_to_log.append(viewpoint_cam.original_image.cuda())
            titles_to_log.append(f"GT RGB {viewpoint_idx}")
            
            images_to_log.append(do_supervision_depth)
            titles_to_log.append(f"Supervision Depth {viewpoint_idx}")
            
            images_to_log.append((1. - depth_to_normal(viewpoint_cam, do_supervision_depth)) / 2.)
            titles_to_log.append(f"Supervision Depth to Normal {viewpoint_idx}")
        
        images_to_log.append(render_pkg["render"])
        titles_to_log.append(f"Rendered RGB {viewpoint_idx}")
        
        if reg_kick_on or mesh_kick_on or depth_order_kick_on:
            images_to_log.append(render_pkg["median_depth"])
            titles_to_log.append(f"Rendered Depth {viewpoint_idx}")
        
        if reg_kick_on or mesh_kick_on:
            images_to_log.append((1. - render_pkg["normal"]) / 2.)
            titles_to_log.append(f"Rendered Normals {viewpoint_idx}")
            
        if "semantic" in render_pkg and render_pkg["semantic"].shape[0] > 0:
            images_to_log.append(render_pkg["semantic"].argmax(dim=0).float())
            titles_to_log.append(f"Rendered Semantic {viewpoint_idx}")
            if viewpoint_cam.semantic_mask is not None:
                images_to_log.append(viewpoint_cam.semantic_mask.cuda().float())
                titles_to_log.append(f"GT Semantic {viewpoint_idx}")
            
        if mesh_kick_on:
            images_to_log.append(torch.zeros_like(render_pkg["render"]))
            titles_to_log.append(f"Mesh RGB {viewpoint_idx}")

            images_to_log.append(
                torch.where(
                    mesh_render_pkg["depth"].detach() > 0,
                    mesh_render_pkg["depth"].detach(),
                    mesh_render_pkg["depth"].detach().max().item()
                )
            )
            titles_to_log.append(f"Mesh Depth {viewpoint_idx}")
            
            images_to_log.append(
                (1. - fix_normal_map(
                    viewpoint_cam, 
                    mesh_render_pkg["normals"].detach(),
                )) / 2.
            )
            titles_to_log.append(f"Mesh Normals {viewpoint_idx}")
        
        # Define visualization directory
        viz_dir = os.path.join(args.model_path, "visualizations", f"iter_{iteration}")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Save combined figure for quick overview
        make_log_figure(
            images=images_to_log, 
            titles=titles_to_log, 
            cmap='Spectral',
            ncols=3,
            figsize=30,
            show_plot=False,
            save_plot=True,
            save_path=os.path.join(viz_dir, "overview.png"),
            return_log_images=False,
        )

        # Save individual high-quality images for detailed inspection
        log_images_dict = make_log_figure(
            images=images_to_log, 
            titles=titles_to_log, 
            cmap='Spectral',
            ncols=3,
            figsize=30,
            show_plot=False,
            save_plot=False,
            return_log_images=True,
        )
        
        for title, img in log_images_dict.items():
            img_to_save = img.clone().detach().nan_to_num(0)
            img_path = os.path.join(viz_dir, f"{title.lower().replace(' ', '_')}.png")
            
            if "semantic" in title.lower():
                # Map to discrete colors using tab20
                max_c = gaussians.num_classes - 1 if gaussians.num_classes > 1 else 1
                normalized_img = (img_to_save.squeeze() / max_c).clamp(0, 1)
                plt.imsave(img_path, plt.get_cmap('tab20')(normalized_img.cpu().numpy()))
            elif "depth" in title.lower():
                # Map to Spectral colormap
                v_min, v_max = img_to_save.min(), img_to_save.max()
                normalized_img = ((img_to_save.squeeze() - v_min) / (v_max - v_min + 1e-5)).clamp(0, 1)
                plt.imsave(img_path, plt.get_cmap('Spectral')(normalized_img.cpu().numpy()))
            else:
                # RGB or Normals (already H,W,C from make_log_figure)
                plt.imsave(img_path, img_to_save.squeeze().clamp(0, 1).cpu().numpy())
        
        if WANDB_FOUND:
            # Log metrics
            wandb_log_dict = {}
            for key, value in postfix_dict.items():
                try:
                    wandb_log_dict[key] = float(value)
                except:
                    wandb_log_dict[key] = value
            run.log(wandb_log_dict, step=iteration)
            
            # Log images
            wandb_log_images_dict = {}
            for log_img_name, log_img in log_images_dict.items():
                wandb_img_to_log = log_img.clone().detach().squeeze()
                # If grayscale, scale to 0-1 and apply colormap
                if wandb_img_to_log.ndim < 3:
                    wandb_img_to_log = (wandb_img_to_log - wandb_img_to_log.min()) / (wandb_img_to_log.max() - wandb_img_to_log.min())
                    cmap = plt.get_cmap('Spectral')
                    wandb_img_to_log = cmap(wandb_img_to_log)[..., :3]
                    wandb_img_to_log = torch.from_numpy(wandb_img_to_log).to(torch.float32)
                # If float, scale to 0-255
                if wandb_img_to_log.dtype == torch.float32:
                    wandb_img_to_log = (wandb_img_to_log.clamp(0., 1.) * 255).to(torch.uint8)
                # Convert to wandb image
                wandb_img_to_log = wandb_img_to_log.cpu().numpy()
                wandb_log_images_dict[log_img_name.replace(str(viewpoint_idx), "")] = wandb.Image(
                    wandb_img_to_log, caption=log_img_name
                )
            run.log(wandb_log_images_dict, step=iteration)

    # ---Report---
    training_report(iteration, l1_loss, testing_iterations, scene, render_imp, (pipe, background))
    if (iteration in saving_iterations):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)

    return (
        postfix_dict,
        ema_loss_for_log, 
        ema_semantic_loss_for_log,
        ema_depth_normal_loss_for_log, 
        ema_mesh_depth_loss_for_log, ema_mesh_normal_loss_for_log, 
        ema_occupied_centers_loss_for_log, ema_occupancy_labels_loss_for_log, 
        ema_depth_order_loss_for_log
    )