import os
import sys
import torch
import torch.utils.cpp_extension
_original_load = torch.utils.cpp_extension.load
def _patched_load(*args, **kwargs):
    module = _original_load(*args, **kwargs)
    if module is not None:
        name = kwargs.get('name') or (args[0] if len(args) > 0 else None)
        if name:
            sys.modules[name] = module
    return module
torch.utils.cpp_extension.load = _patched_load

import torchvision
import trimesh
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from os import makedirs

import sys
sys.path.append(os.path.dirname(__file__))

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.mesh import Meshes, MeshRasterizer, ScalableMeshRenderer as MeshRenderer
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos

def render_set_static(output_path, name, views, mesh_obj, renderer):
    render_path = os.path.join(output_path, name, "renders")
    gts_path = os.path.join(output_path, name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if not views:
        return

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} progress")):
        with torch.no_grad():
            # Set max_triangles_in_batch to 1,000,000 to prevent nvdiffrast subtriangle count overflow
            rendered_pkg = renderer(
                mesh=mesh_obj,
                cameras=view,
                cam_idx=0,
                use_antialiasing=True,
                max_triangles_in_batch=1000000
            )
        rendering = rendered_pkg["rgb"].squeeze(0).permute(2, 0, 1) # [3, H, W]
        gt = view.original_image[0:3, :, :]
        
        # Save image
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def main():
    parser = ArgumentParser(description="Render mesh with static vertex colors")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ply_file", required=True, type=str)
    parser.add_argument("--output_path", type=str, default="")
    
    # ModelParams automatically adds --source_path / -s, --resolution, --data_device, --model_path / -m
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)

    # Resolve default values for any parameters that were not specified or loaded from config
    default_parser = ArgumentParser()
    ModelParams(default_parser, sentinel=False)
    PipelineParams(default_parser)
    default_args = default_parser.parse_args([])
    
    for k, v in vars(default_args).items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    # Validate source_path since it is required for dataset loading
    if not args.source_path:
        raise ValueError("Please specify the source path using --source_path or -s")

    output_path = args.output_path if args.output_path else os.path.join(os.path.dirname(args.ply_file), "mesh_static_nvs")
    print(f"[INFO] Rendering " + args.ply_file + " with static vertex colors")
    print(f"[INFO] Output path: {output_path}")

    # Instantiate scene
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, "images", True)
    else:
        raise ValueError("Only Colmap data sets are supported for now")

    print("Loading Training Cameras")
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
    print("Loading Test Cameras")
    test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, args)

    safe_state(args.quiet)

    # Load mesh
    print(f"[INFO] Loading mesh from {args.ply_file}...")
    mesh = trimesh.load(args.ply_file)
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        verts_colors = torch.from_numpy(mesh.visual.vertex_colors).float().cuda()[:, :3] / 255.0
    else:
        verts_colors = torch.ones_like(verts)
        
    mesh_obj = Meshes(verts=verts, faces=faces, verts_colors=verts_colors)
    
    # Initialize ScalableMeshRenderer
    rasterizer = MeshRasterizer(cameras=train_cameras[0] if train_cameras else test_cameras[0])
    renderer = MeshRenderer(rasterizer=rasterizer)

    if not args.skip_train:
         render_set_static(output_path, "train", train_cameras, mesh_obj, renderer)

    if not args.skip_test:
         render_set_static(output_path, "test", test_cameras, mesh_obj, renderer)

    print(f"[INFO] Static rendering complete. Results saved in {output_path}")

if __name__ == "__main__":
    main()
