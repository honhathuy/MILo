import torch
import numpy as np
import maxflow
from tqdm import tqdm
import os
import argparse
import copy
from scene.mesh import Meshes, MeshRasterizer, MeshRenderer, ScalableMeshRenderer
from utils.geometry_utils import transform_points_world_to_view, transform_points_to_pixel_space
from regularization.sdf.depth_fusion import get_interpolated_value_from_pixel_coordinates
from plyfile import PlyData, PlyElement

def load_ply_mesh(path, device='cuda'):
    plydata = PlyData.read(path)
    v = plydata['vertex']
    verts = torch.stack([torch.from_numpy(v['x']), torch.from_numpy(v['y']), torch.from_numpy(v['z'])], dim=-1).to(device).float()
    
    colors = None
    if 'red' in v:
        colors = torch.stack([torch.from_numpy(v['red']), torch.from_numpy(v['green']), torch.from_numpy(v['blue'])], dim=-1).to(device).float() / 255.0
        
    faces = torch.from_numpy(np.vstack(plydata['face']['vertex_indices'])).to(device).long()
    return Meshes(verts=verts, faces=faces, verts_colors=colors)

def project_masks_to_mesh_vertices(mesh, cameras, target_class, depth_threshold=0.05):
    """
    Projects 2D semantic masks onto mesh vertices using the MeshRenderer to handle occlusions.
    """
    device = mesh.verts.device
    num_verts = mesh.verts.shape[0]
    vertex_scores = torch.zeros(num_verts, device=device)
    vertex_counts = torch.zeros(num_verts, device=device)
    
    # Initialize Renderer
    rasterizer = MeshRasterizer(cameras=cameras)
    renderer = ScalableMeshRenderer(rasterizer)
    
    print(f"[INFO] Projecting masks from {len(cameras)} views...")
    for i, cam in enumerate(tqdm(cameras)):
        if not hasattr(cam, 'semantic_mask') or cam.semantic_mask is None:
            continue
            
        # 1. Render mesh depth from this view
        # We use a smaller max_triangles_in_batch to avoid subtriangle count overflow
        render_pkg = renderer(mesh, cam_idx=i, return_depth=True, use_antialiasing=False, max_triangles_in_batch=1000000)
        rendered_depth = render_pkg["depth"].squeeze() # (H, W)
        
        # 2. Project vertices to pixel space
        view_points = transform_points_world_to_view(mesh.verts, [cam]).squeeze() # (N, 3)
        pix_coords = transform_points_to_pixel_space(view_points[None], [cam], points_are_already_in_view_space=True, keep_float=True)[0] # (N, 2)
        
        depths = view_points[:, 2]
        
        # 3. Visibility check (Frustum + Occlusion)
        h, w = cam.image_height, cam.image_width
        in_frustum = (pix_coords[:, 0] >= 0) & (pix_coords[:, 0] < w) & \
                     (pix_coords[:, 1] >= 0) & (pix_coords[:, 1] < h) & \
                     (depths > cam.znear)
        
        if not in_frustum.any():
            continue
            
        # Sample rendered depth at projected pixels
        # Use nearest neighbor for the occlusion check
        valid_pix = pix_coords[in_frustum]
        sampled_depth = get_interpolated_value_from_pixel_coordinates(rendered_depth.unsqueeze(-1), valid_pix, interpolation_mode='nearest').squeeze()
        
        # Vertex is visible if its depth matches the rendered depth
        is_visible = (depths[in_frustum] - sampled_depth).abs() < depth_threshold
        
        # 4. Sample semantic mask
        target_indices = torch.where(in_frustum)[0][is_visible]
        if len(target_indices) == 0:
            continue
            
        mask_2d = (cam.semantic_mask == target_class).float()

        # Penalize the enemy class by subtracting its probability
        # This helps remove "tentacles" from the enemy object that leaked into the target mask
        mask_2d = mask_2d - (cam.semantic_mask != target_class).float()

        sampled_mask = get_interpolated_value_from_pixel_coordinates(mask_2d.unsqueeze(-1), pix_coords[target_indices], interpolation_mode='nearest').squeeze()
        
        vertex_scores[target_indices] += sampled_mask
        vertex_counts[target_indices] += 1.0

    # Average scores. 
    # With the (mask == target) - (mask != target) logic, 
    # avg_scores will be in range [-1.0, 1.0]
    avg_scores = torch.where(vertex_counts > 0, vertex_scores / vertex_counts, torch.zeros_like(vertex_scores))
    
    # Remap from [-1, 1] to [0, 1] for MaxFlow compatibility
    # 1.0 (Target) -> 1.0 (Strong Source)
    # 0.0 (Unseen) -> 0.5 (Neutral)
    # -1.0 (Enemy) -> 0.0 (Strong Sink)
    final_scores = (avg_scores + 1.0) / 2.0
    
    return final_scores

def segment_mesh_with_graphcut(mesh, scores, user_weight=5.0, sig_col_neigh=30.0, spatial_radius=0.005):
    """
    Builds a graph based on mesh edges AND spatial neighbors, then performs max-flow/min-cut.
    """
    num_verts = mesh.verts.shape[0]
    verts_np = mesh.verts.cpu().numpy()
    colors_np = mesh.verts_colors.cpu().numpy() if mesh.verts_colors is not None else None
    
    # 1. Edge extraction (Combining Mesh Edges + Spatial Neighbors)
    print("[INFO] Extracting mesh edges...")
    faces = mesh.faces
    e_u = faces[:, [0, 1, 2]].flatten().cpu().numpy()
    e_v = faces[:, [1, 2, 0]].flatten().cpu().numpy()
    
    print(f"[INFO] Finding spatial neighbors (radius={spatial_radius})...")
    from scipy.spatial import KDTree
    tree = KDTree(verts_np)
    # Find up to 6 spatial neighbors within the radius for every vertex
    dist_s, idx_s = tree.query(verts_np, k=7, distance_upper_bound=spatial_radius)
    
    # Flatten spatial neighbors
    s_u = np.repeat(np.arange(num_verts), 7)
    s_v = idx_s.flatten()
    # Filter out self-loops and invalid neighbors
    s_mask = (s_v < num_verts) & (s_u != s_v)
    
    # Combine everything
    u_all = np.concatenate([e_u, s_u[s_mask]])
    v_all = np.concatenate([e_v, s_v[s_mask]])
    
    # No deduplication needed! Maxflow sums capacities for duplicate edges.
    # This saves a massive amount of memory and time by skipping the expensive unique() call.
    u_idx, v_idx = u_all, v_all
    
    g = maxflow.Graph[float]()
    g.add_nodes(num_verts)
    
    # 2. Terminal edges (Data term)
    print("[INFO] Building terminal edges...")
    scores_np = scores.cpu().numpy()
    source_caps = scores_np * user_weight
    sink_caps = (1.0 - scores_np) * user_weight
    for i in range(num_verts):
        g.add_tedge(i, source_caps[i], sink_caps[i])
        
    # 3. Neighbor edges
    print("[INFO] Pre-calculating edge weights...")
    dists = np.linalg.norm(verts_np[u_idx] - verts_np[v_idx], axis=1)
    weights = np.exp(-dists * 10.0)
    
    if colors_np is not None:
        col_dists = np.linalg.norm(colors_np[u_idx] - colors_np[v_idx], axis=1)
        # Using squared distance makes the penalty much sharper for large color differences
        # This helps separate the vase from the table at their contact line.
        weights *= np.exp(-(col_dists**2) * sig_col_neigh)
        
    print(f"[INFO] Building {len(u_idx)} graph edges...")
    # INCREASED MULTIPLIER: This makes the mesh much "stickier"
    weights *= 10
    for i in range(len(u_idx)):
        w = weights[i]
        g.add_edge(u_idx[i], v_idx[i], w, w)
        
    print("[INFO] Computing max flow...")
    g.maxflow()
    
    # Vectorized segment retrieval
    print("[INFO] Retrieving segments...")
    keep_mask = g.get_grid_segments(np.arange(num_verts)) == 0
    return torch.from_numpy(keep_mask).to(mesh.verts.device)

def remove_small_components(mesh, scores=None, min_size=500, score_threshold=0.8):
    """
    Removes disconnected mesh components that are either too small OR
    do not contain any vertices with a high semantic confidence score.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    
    num_verts = mesh.verts.shape[0]
    if num_verts == 0: return torch.zeros(0, device=mesh.verts.device, dtype=torch.bool)
    faces = mesh.faces.cpu().numpy()
    
    # Build adjacency matrix
    u = faces[:, [0, 1, 2]].flatten()
    v = faces[:, [1, 2, 0]].flatten()
    data = np.ones(len(u))
    adj = coo_matrix((data, (u, v)), shape=(num_verts, num_verts))
    
    print(f"[INFO] Finding connected components...")
    n_components, labels = connected_components(adj, directed=False)
    
    # Count sizes and check scores
    print(f"[INFO] Filtering {n_components} components...")
    keep_components = []
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    scores_np = scores.cpu().numpy() if scores is not None else None
    
    for i, label in enumerate(unique_labels):
        if counts[i] < min_size:
            continue
            
        if scores_np is not None:
            comp_mask = (labels == label)
            if scores_np[comp_mask].max() < score_threshold:
                continue
                
        keep_components.append(label)
    
    keep_mask = np.isin(labels, keep_components)
    print(f"[INFO] Kept {len(keep_components)} components, removed {n_components - len(keep_components)} islands/floaters.")
    
    return torch.from_numpy(keep_mask).to(mesh.verts.device)

def fill_mesh_holes(mesh, iterations=50, weight=0.5, seam_verts=None):
    """
    Ultra-robust hole filling using 2D Delaunay Triangulation.
    Uses DBSCAN to isolate the local hole from distant background boundaries.
    """
    if seam_verts is None or len(seam_verts) < 10:
        print("[INFO] Not enough seam vertices found to patch.")
        return mesh

    from scipy.spatial import Delaunay
    from sklearn.cluster import DBSCAN
    import numpy as np
    
    verts = mesh.verts.cpu().numpy()
    faces = list(mesh.faces.cpu().numpy())
    seam_idx_all = np.array(list(seam_verts))
    points_all = verts[seam_idx_all]
    
    # 0. CLUSTERING: Isolate the local hole from distant background boundaries
    print("[INFO] Clustering seam vertices to isolate the local hole...")
    # eps=0.2 means points within 20cm belong to the same cluster
    clustering = DBSCAN(eps=0.2, min_samples=10).fit(points_all)
    labels = clustering.labels_
    
    # Pick the cluster with the most points (this is your vase hole)
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        print("[INFO] No clear local hole cluster found.")
        return mesh
        
    best_label = max(unique_labels, key=lambda l: np.sum(labels == l))
    mask = labels == best_label
    seam_idx = seam_idx_all[mask]
    points = points_all[mask]
    
    print(f"[INFO] Patching local cluster with {len(points)} vertices...")
    
    # 1. SMOOTH SURROUNDING SURFACE (Laplacian Smoothing)
    # Use Laplacian smoothing to relax the "upward pointing tube" gracefully
    from scipy.spatial import KDTree
    from scipy.sparse import csr_matrix
    print("[INFO] Laplacian smoothing surrounding surface (5cm radius)...")
    
    tree = KDTree(points)
    dists, _ = tree.query(verts, distance_upper_bound=0.05)
    near_mask = dists < 0.05
    
    # Build fast sparse adjacency matrix
    faces_np = np.array(faces)
    row = np.concatenate([faces_np[:,0], faces_np[:,1], faces_np[:,2], faces_np[:,1], faces_np[:,2], faces_np[:,0]])
    col = np.concatenate([faces_np[:,1], faces_np[:,2], faces_np[:,0], faces_np[:,0], faces_np[:,1], faces_np[:,2]])
    A = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(verts), len(verts)))
    degree = np.array(A.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    inv_degree = 1.0 / degree
    
    # Smoothly interpolate the Laplacian weight based on distance
    w = 1.0 - (dists[near_mask] / 0.05)
    w = w * w # quadratic falloff for smoother transition
    
    # Aggressively relax the mesh (100 iterations)
    for _ in range(200):
        laplacian = A.dot(verts) * inv_degree[:, None]
        verts[near_mask] = (1.0 - w[:, None]) * verts[near_mask] + w[:, None] * laplacian[near_mask]
        
    # Update seam points after smoothing
    points = verts[seam_idx]

    # 2. Project points to their best-fit 2D plane (PCA-style)
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, Vh = np.linalg.svd(centered)
    points_2d = centered @ Vh[:2].T
    
    # 2. Dense Grid Generation (Manual Poisson-style reconstruction)
    # We add a 1cm grid of points to make the patch solid and high-resolution
    x_min, x_max = points_2d[:,0].min(), points_2d[:,0].max()
    y_min, y_max = points_2d[:,1].min(), points_2d[:,1].max()
    gx, gy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_2d = np.c_[gx.ravel(), gy.ravel()]
    
    # Only keep grid points that are inside the hole's footprint
    center_2d = np.mean(points_2d, axis=0)
    dist_to_center = np.linalg.norm(grid_2d - center_2d, axis=1)
    max_radius = np.max(np.linalg.norm(points_2d - center_2d, axis=1))
    grid_2d = grid_2d[dist_to_center < max_radius * 0.9]
    
    # 3. Combine original rim points with our new high-res grid
    # Map back to 3D and rebuild mesh
    new_verts_list = list(verts)
    new_colors_list = list(mesh.verts_colors.cpu().numpy()) if mesh.verts_colors is not None else None
    
    # Add grid points to the mesh
    grid_3d = mean + grid_2d @ Vh[:2]
    grid_start_idx = len(new_verts_list)
    new_verts_list.extend(grid_3d)
    if new_colors_list is not None:
        avg_color = np.mean(np.array(new_colors_list)[seam_idx], axis=0)
        new_colors_list.extend([avg_color] * len(grid_2d))
    
    # Create mapping for triangulation
    # indices 0...len(seam_idx)-1 are the rim points
    # indices len(seam_idx)... are the grid points
    all_points_2d = np.vstack([points_2d, grid_2d])
    tri = Delaunay(all_points_2d)
    
    # 4. Add the new triangles
    added_count = 0
    for f_idx in tri.simplices:
        # Map Delaunay indices back to global vertex indices
        global_f = []
        for local_idx in f_idx:
            if local_idx < len(seam_idx):
                global_f.append(seam_idx[local_idx])
            else:
                global_f.append(grid_start_idx + (local_idx - len(seam_idx)))
        
        # Check edge lengths to avoid "jumping" across the whole scene
        p_tri = all_points_2d[f_idx]
        edge_len = np.max([np.linalg.norm(p_tri[0]-p_tri[1]), 
                           np.linalg.norm(p_tri[1]-p_tri[2]), 
                           np.linalg.norm(p_tri[2]-p_tri[0])])
        if edge_len < 0.1: # 10cm limit
            faces.append(global_f)
            added_count += 1
            
    print(f"[INFO] Added {added_count} triangles to bridge the messy area.")
    
    # 5. Local Smoothing to blend the edges
    from scene.mesh import Meshes
    out_mesh = Meshes(
        verts=torch.from_numpy(np.array(new_verts_list)).to(mesh.verts.device).float(),
        faces=torch.from_numpy(np.array(faces)).to(mesh.verts.device).long(),
        verts_colors=torch.from_numpy(np.array(new_colors_list)).to(mesh.verts.device).float() if new_colors_list is not None else None
    )
    return out_mesh

def save_mesh_ply(mesh, path):
    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    
    v_list = []
    has_colors = mesh.verts_colors is not None
    colors = (mesh.verts_colors.detach().cpu().numpy() * 255).astype(np.uint8) if has_colors else None
    
    for i in range(len(verts)):
        if has_colors:
            v_list.append((verts[i, 0], verts[i, 1], verts[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))
        else:
            v_list.append((verts[i, 0], verts[i, 1], verts[i, 2]))
            
    if has_colors:
        v_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    else:
        v_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        
    v_el = PlyElement.describe(np.array(v_list, dtype=v_dtype), 'vertex')
    
    f_list = [(faces[i],) for i in range(len(faces))]
    f_el = PlyElement.describe(np.array(f_list, dtype=[('vertex_indices', 'i4', (3,))]), 'face')
    
    PlyData([v_el, f_el]).write(path)

if __name__ == "__main__":
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos

    input_path = 'E:\\Learning\\hcmus\\3d_reconstructions\\MILo\\milo\\output\\garden2\\mesh_exact_computation_9pivots_transmittance_threshold_0.7_searched.ply'
    output_path = 'E:\\Learning\\hcmus\\3d_reconstructions\\MILo\\milo\\output\\garden2\\mesh_segmented0.ply'
    data_path = 'E:\\Learning\\hcmus\\3d_reconstructions\\MILo\\milo\\data\\garden\\undistorted'
    target_class = 0
    user_weight = 2 # Lower trust in 2D masks to favor mesh color/connectivity
    sig_col_neigh = 100.0 # High sensitivity to color boundaries

    class MockArgs:
        resolution = 1
        data_device = "cuda"

    # 1. Load Mesh
    print(f"[INFO] Loading mesh from {input_path}...")
    mesh = load_ply_mesh(input_path)

    # 2. Load Scene Info and Cameras
    print(f"[INFO] Loading cameras from {data_path}...")
    scene_info = readColmapSceneInfo(path=data_path, images="images", eval=False, no_depth_prior=True)
    view_points = cameraList_from_camInfos(scene_info.train_cameras, 1.0, MockArgs())

    # 3. Project Masks
    vertex_scores = project_masks_to_mesh_vertices(mesh, view_points, target_class)

    # 4. Graph Cut
    keep_mask = segment_mesh_with_graphcut(mesh, vertex_scores, user_weight=user_weight, sig_col_neigh=sig_col_neigh)

    # 5. Identify Seam Vertices (Vertices on the edge of the cut)
    print("[INFO] Identifying segmentation seam...")
    faces_np = mesh.faces.cpu().numpy()
    keep_mask_np = keep_mask.cpu().numpy()
    
    # Identify Seam Vertices on the boundary
    seam_verts_orig = set()
    for f in faces_np:
        ins = [keep_mask_np[f[0]], keep_mask_np[f[1]], keep_mask_np[f[2]]]
        if any(ins) and not all(ins):
            for i in range(3):
                if ins[i]: seam_verts_orig.add(f[i])
    
    # 6. Extract Segmented Mesh
    segmented_mesh = mesh.submesh(vert_mask=torch.from_numpy(keep_mask_np).to(keep_mask.device))
    
    # Map seam vertices to the new indices in the submesh
    old_to_new = np.full(len(keep_mask_np), -1)
    old_to_new[keep_mask_np] = np.arange(keep_mask_np.sum())
    seam_verts_new = [old_to_new[v] for v in seam_verts_orig if old_to_new[v] != -1]

    # 7. Remove Small Islands
    island_mask = remove_small_components(segmented_mesh, scores=vertex_scores[keep_mask_np], min_size=5000)
    final_mesh = segmented_mesh.submesh(vert_mask=island_mask)
    
    # Map seam vertices one more time for the island removal
    old_to_new_island = np.full(len(island_mask), -1)
    island_mask_np = island_mask.cpu().numpy()
    old_to_new_island[island_mask_np] = np.arange(island_mask_np.sum())
    seam_verts_final = [old_to_new_island[v] for v in seam_verts_new if v < len(old_to_new_island) and old_to_new_island[v] != -1]

    # 8. Optional Seam-Aware Hole Filling
    fill_holes = True
    if fill_holes:
        final_mesh = fill_mesh_holes(final_mesh, seam_verts=seam_verts_final)

    print(f"[INFO] Saving result to {output_path}...")
    save_mesh_ply(final_mesh, output_path)
    print("[INFO] Done!")
