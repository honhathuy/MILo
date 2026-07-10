import os
import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel
from scene.mesh import Meshes, MeshRasterizer, ScalableMeshRenderer
from scene.cameras import Camera
from utils.graphics_utils import getProjectionMatrix
from utils.video_utils import get_interpolate_render_path, get_spiral_render_path


class MiniCamera(Camera):
    def __init__(self, c2w, FoVx, FoVy, image_width, image_height, znear=0.01, zfar=100.0):
        # Initialize nn.Module without running Camera's full __init__
        super(Camera, self).__init__()
        
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height
        self.znear = znear
        self.zfar = zfar
        
        # Compute w2c
        w2c = np.linalg.inv(c2w)
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)


def main():
    parser = ArgumentParser(description="Render video from training trajectory for extracted mesh")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=18000, type=int)
    parser.add_argument("--mesh", default="", type=str, help="Path to the extracted .ply mesh file")
    parser.add_argument("--output_video", default="", type=str, help="Path to the output .mp4 video file")
    parser.add_argument("--fps", default=30, type=int, help="Frames per second of the output video")
    parser.add_argument("--n_views", default=240, type=int, help="Number of views/frames in the trajectory")
    parser.add_argument("--trajectory_type", default="exact", choices=["exact", "interpolate", "spiral"], type=str)
    
    args = get_combined_args(parser)
    
    # Load scene and Gaussian model to fetch training cameras
    print("[INFO] Loading scene and cameras...")
    gaussians = GaussianModel(args.sh_degree, num_classes=0, n_gaussian_features=4)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    import re
    def natural_sort_key(camera):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', camera.image_name)]
        
    train_cameras = sorted(scene.getTrainCameras(), key=natural_sort_key)
    
    if not train_cameras:
        raise ValueError("No training cameras found in the scene.")
        
    print(f"[INFO] Loaded {len(train_cameras)} training cameras.")
    
    # Auto-detect mesh path if not specified
    if not args.mesh:
        print("[INFO] Mesh path not specified, looking for mesh in model path...")
        mesh_files = [f for f in os.listdir(args.model_path) if f.startswith("mesh_") and f.endswith(".ply")]
        if not mesh_files:
            raise FileNotFoundError(f"No extracted meshes found in {args.model_path}. Please extract mesh first or specify --mesh parameter.")
        # Prioritize postprocessed mesh if available
        post_mesh_files = [f for f in mesh_files if "_post" in f]
        selected_mesh_name = post_mesh_files[0] if post_mesh_files else mesh_files[0]
        args.mesh = os.path.join(args.model_path, selected_mesh_name)
        
    print(f"[INFO] Using mesh file: {args.mesh}")
    
    # Auto-detect output video path if not specified
    if not args.output_video:
        mesh_dir = os.path.dirname(args.mesh)
        mesh_base = os.path.splitext(os.path.basename(args.mesh))[0]
        args.output_video = os.path.join(mesh_dir, f"{mesh_base}_{args.trajectory_type}.mp4")
        
    print(f"[INFO] Video output path: {args.output_video}")
    
    # Retrieve average camera properties for rendering
    avg_fov_x = np.mean([cam.FoVx for cam in train_cameras])
    avg_fov_y = np.mean([cam.FoVy for cam in train_cameras])
    width = train_cameras[0].image_width
    height = train_cameras[0].image_height
    znear = train_cameras[0].znear
    zfar = train_cameras[0].zfar
    
    # Compute C2Ws from training cameras
    c2ws = []
    for cam in train_cameras:
        w2c = cam.world_view_transform.T.cpu().numpy()
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)
    
    # Generate camera trajectory
    print(f"[INFO] Generating {args.trajectory_type} trajectory...")
    if args.trajectory_type == "exact":
        render_poses = c2ws
    elif args.trajectory_type == "interpolate":
        from scipy.spatial.transform import Rotation as R, Slerp
        from scipy.interpolate import CubicSpline
        
        # To make a seamless loop, append the first camera pose at the end
        c2ws_loop = np.concatenate([c2ws, c2ws[0:1]], axis=0)
        N = len(c2ws_loop)
        
        times_in = np.arange(N)
        times_out = np.linspace(0, N - 1, args.n_views)
        
        rotations = R.from_matrix(c2ws_loop[:, :3, :3])
        positions = c2ws_loop[:, :3, 3]
        
        # Smoothly interpolate rotations (using Quaternion SLERP)
        slerp = Slerp(times_in, rotations)
        interp_rotations = slerp(times_out)
        
        # Smoothly interpolate positions using a periodic Cubic Spline to avoid sagging
        cs = CubicSpline(times_in, positions, bc_type='periodic')
        interp_positions = cs(times_out)
            
        render_poses = []
        for r, p in zip(interp_rotations, interp_positions):
            pose = np.eye(4)
            pose[:3, :3] = r.as_matrix()
            pose[:3, 3] = p
            render_poses.append(pose)
        render_poses = np.stack(render_poses, axis=0)
    elif args.trajectory_type == "spiral":
        from utils.camera_utils import get_cameras_spatial_extent
        extent = get_cameras_spatial_extent(train_cameras)
        radius = extent['radius'].item()
        near_far = [radius * 0.1, radius * 10.0]
        render_poses = get_spiral_render_path(c2ws, near_far, rads_scale=0.5, N_views=args.n_views)
    else:
        raise ValueError(f"Unknown trajectory type: {args.trajectory_type}")
        
    # Load mesh
    print(f"[INFO] Loading mesh from {args.mesh}...")
    mesh = trimesh.load(args.mesh)
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        verts_colors = torch.from_numpy(mesh.visual.vertex_colors).float().cuda()[:, :3] / 255.0
    else:
        verts_colors = torch.ones_like(verts)
        
    mesh_obj = Meshes(verts=verts, faces=faces, verts_colors=verts_colors)
    
    # Initialize ScalableMeshRenderer
    rasterizer = MeshRasterizer(cameras=train_cameras[0])
    renderer = ScalableMeshRenderer(rasterizer=rasterizer)
    
    # Render video
    print(f"[INFO] Rendering video ({len(render_poses)} frames)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))
    
    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        # Get frame-specific camera properties
        if args.trajectory_type == "exact":
            render_cam = train_cameras[idx]
        else:
            # Construct mock camera for interpolated poses
            render_cam = MiniCamera(
                c2w=pose,
                FoVx=avg_fov_x,
                FoVy=avg_fov_y,
                image_width=width,
                image_height=height,
                znear=znear,
                zfar=zfar
            )
            
        with torch.no_grad():
            rendered_pkg = renderer(
                mesh=mesh_obj,
                cameras=render_cam,
                cam_idx=0,
                use_antialiasing=True
            )
            
        rgb = rendered_pkg["rgb"].squeeze(0).cpu().numpy()
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        
        # If the rendered size is different from the video size, resize it
        if rgb.shape[1] != width or rgb.shape[0] != height:
            rgb = cv2.resize(rgb, (width, height))
            
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_video.write(bgr)
        
    out_video.release()
    print(f"[INFO] Video successfully rendered and saved to: {args.output_video}")


if __name__ == "__main__":
    main()
