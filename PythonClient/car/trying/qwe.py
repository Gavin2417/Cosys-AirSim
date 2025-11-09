import os, math, time, heapq, json, argparse
import numpy as np
import open3d as o3d
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
import cosysairsim as airsim
from linefit import ground_seg
from function5 import *
import casadi as ca
from skimage.graph import route_through_array
base = os.path.dirname(__file__)        
project_root = base
print(project_root)
rand_dir = os.path.join(project_root, "rand")
os.chdir(rand_dir)
from predict1 import RandlaGroundSegmentor
from scipy.ndimage import convolve
from scipy.interpolate import CubicSpline
import numpy as np

def euler_from_R(R):
    """
    Returns roll, pitch, yaw in radians from a 3x3 rotation matrix.
    Assumes R maps body->world (consistent with your yaw = atan2(R[1,0], R[0,0])).
    """
    roll  = math.atan2(R[2,1], R[2,2])
    pitch = -math.asin(max(-1.0, min(1.0, R[2,0])))
    yaw   = math.atan2(R[1,0], R[0,0])
    return roll, pitch, yaw

def is_flipped(R, up_z_threshold=0.3, angle_deg_threshold=85.0):
    """
    Flip if the vehicle's body 'up' axis points too little toward world +Z
    OR if roll/pitch exceed a large angle threshold.
    """
    # world-up alignment of body-Z axis:
    up_world = R[:, 2]         # body z-axis expressed in world frame
    if float(up_world[2]) < up_z_threshold:
        return True

    roll, pitch, _ = euler_from_R(R)
    return (abs(math.degrees(roll))  > angle_deg_threshold or
            abs(math.degrees(pitch)) > angle_deg_threshold)

# --- add these small helpers near your other utils ---
def remap_mask(old_mask, old_x_mid, old_y_mid, new_x_edges, new_y_edges, new_shape):
    """
    Map a boolean mask defined on (old_x_mid, old_y_mid) to the new grid defined by new_x_edges/new_y_edges.
    Grid convention: indexing='ij' -> axis 0 is x, axis 1 is y.
    """
    if old_mask is None:
        return np.zeros(new_shape, dtype=bool)

    ii, jj = np.nonzero(old_mask)
    if ii.size == 0:
        return np.zeros(new_shape, dtype=bool)

    xs = old_x_mid[ii]
    ys = old_y_mid[jj]

    ni = np.clip(np.digitize(xs, new_x_edges) - 1, 0, new_shape[0]-1)
    nj = np.clip(np.digitize(ys, new_y_edges) - 1, 0, new_shape[1]-1)

    new_mask = np.zeros(new_shape, dtype=bool)
    new_mask[ni, nj] = True
    return new_mask

def remap_values(old_vals, old_x_mid, old_y_mid, new_x_edges, new_y_edges, new_shape, reducer=np.nanmax):
    """
    Map a value grid (e.g., prev_risk_grid) to the new grid.
    Multiple old cells may land in one new cell -> reduce with `reducer` (nanmax by default).
    """
    if old_vals is None:
        return None

    ii, jj = np.where(np.isfinite(old_vals))
    if ii.size == 0:
        return np.full(new_shape, np.nan, dtype=float)

    xs = old_x_mid[ii]
    ys = old_y_mid[jj]
    ni = np.clip(np.digitize(xs, new_x_edges) - 1, 0, new_shape[0]-1)
    nj = np.clip(np.digitize(ys, new_y_edges) - 1, 0, new_shape[1]-1)

    new_vals = np.full(new_shape, np.nan, dtype=float)
    for k in range(ni.size):
        i, j = ni[k], nj[k]
        v = old_vals[ii[k], jj[k]]
        if np.isnan(new_vals[i, j]):
            new_vals[i, j] = v
        else:
            new_vals[i, j] = reducer([new_vals[i, j], v])
    return new_vals

def build_arc_length_path(raw_xy: np.ndarray, ds=0.15):
    """
    raw_xy: (M,2) points from A* (in world {x,y}, not grid indices)
    returns:
      S:   (K,) arc-length samples
      XY:  (K,2) smoothed, arc-length sampled path
      PSI: (K,) heading along the path (rad)
      KAP: (K,) curvature (1/m)
    """
    if len(raw_xy) < 3:
        XY = raw_xy.copy()
        s  = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(XY, axis=0), axis=1))])
        S  = np.arange(0, s[-1]+1e-9, max(ds, 1e-3))
        psi = np.zeros_like(S)
        kap = np.zeros_like(S)
        return S, np.interp(S, s, XY[:,0])[:,None].repeat(2,1), psi, kap

    seg = np.diff(raw_xy, axis=0)
    s   = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    S   = np.arange(0.0, max(s[-1], ds)+1e-9, ds)

    sx = CubicSpline(s, raw_xy[:,0], bc_type='clamped')
    sy = CubicSpline(s, raw_xy[:,1], bc_type='clamped')

    x  = sx(S);  y  = sy(S)
    dx = sx(S,1); dy = sy(S,1)
    ddx= sx(S,2); ddy= sy(S,2)

    psi   = np.arctan2(dy, dx)
    denom = np.maximum((dx*dx + dy*dy)**1.5, 1e-6)
    kap   = (dx*ddy - dy*ddx)/denom
    XY = np.column_stack([x,y])
    return S, XY, psi, kap
def curvature_speed(kappa, v_max=0.8, a_lat_max=1.0, v_min=0.15):
    # v_curv = sqrt(a_lat_max / |kappa|) clipped by v_max
    v_curv = np.sqrt(np.maximum(a_lat_max / np.maximum(np.abs(kappa), 1e-6), 0.0))
    v = np.minimum(v_curv, v_max)
    return np.clip(v, v_min, v_max)

def risk_scaled_speed(xy, risk_grid, X_mesh, Y_mesh, base_v, max_risk=50.0, scale=0.5):
    """
    Reduces speed near risky cells up to `scale` fraction.
    """
    # nearest-cell lookup
    centers = np.column_stack((X_mesh.ravel(), Y_mesh.ravel()))
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    idx  = tree.query(xy, k=1)[1]
    local_risk = risk_grid.ravel()[idx]
    factor = 1.0 - scale*np.clip(local_risk/max_risk, 0, 1)
    return np.clip(base_v * factor, 0.1, np.max(base_v))
def pick_reference_window(xy_path, psi_path, v_path, ego_xy, N, lead_idx=1):
    # closest arc-length sample to the car
    d = np.linalg.norm(xy_path - ego_xy, axis=1)
    i0 = int(np.argmin(d))
    j  = np.clip(i0 + lead_idx + np.arange(N), 0, len(xy_path)-1)
    return xy_path[j], psi_path[j], v_path[j], i0

def fuse_geom_edge_preserving(step_risk, slope_risk,
                              tau=0.35,        # edge threshold
                              beta=1.0):       # how much slope you allow in flat areas
    """
    Edge-preserving fusion:
      - Where step_risk >= tau: keep step (sharp).
      - Else: allow slope to contribute (scaled by beta).
    """
    step = np.asarray(step_risk, dtype=float)
    slope = np.asarray(slope_risk, dtype=float)

    # valid mask
    m = np.isfinite(step) | np.isfinite(slope)

    # gate: 1 where step is small, 0 where it's large (edge)
    gate = np.clip((tau - step) / max(tau, 1e-6), 0.0, 1.0)
    # fused = max(step, beta * gate * slope)
    fused = np.where(m, np.maximum(step, beta * gate * slope), np.nan)
    return fused

def softmax_entropy(p, eps=1e-8):
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log(p), axis=1)
    Hmax = np.log(p.shape[1])
    return H / Hmax  # 0..1  (0=confident, 1=uncertain)
def step_risk_confidence(Z_mean_grid, N_count_grid, grid_resolution,
                         radius_m=0.5, z_noise_std=0.02,
                         w_cov=0.4, w_sup=0.3, w_cons=0.3, k_sup=6.0):
    """
    Confidence in [0,1] from local coverage, support, and height std.
    """
    # neighborhood radius in cells
    r = max(1, int(np.round(radius_m / grid_resolution)))
    k = 2*r + 1
    K = np.ones((k, k), dtype=float)

    valid = (~np.isnan(Z_mean_grid)).astype(float)
    Z0 = np.nan_to_num(Z_mean_grid, nan=0.0)

    sumZ   = convolve(Z0, K, mode='constant', cval=0.0)
    sumZ2  = convolve(Z0*Z0, K, mode='constant', cval=0.0)
    count  = convolve(valid, K, mode='constant', cval=0.0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(count > 0, sumZ / count, np.nan)
        mean2 = np.where(count > 0, sumZ2 / count, np.nan)
        var = np.maximum(mean2 - mean**2, 0.0)
        std = np.sqrt(var)

    coverage = np.clip(count / (k*k), 0.0, 1.0)
    support  = np.tanh(N_count_grid / max(k_sup, 1e-6))
    consistency = 1.0 / (1.0 + (std / max(z_noise_std, 1e-6)))

    eps = 1e-6
    log_conf = (w_cov*np.log(coverage + eps) +
                w_sup*np.log(support  + eps) +
                w_cons*np.log(consistency + eps))
    conf = np.exp(log_conf)
    conf[np.isnan(Z_mean_grid)] = np.nan
    return conf
class GridMap:
    def __init__(self, resolution):
        self.resolution = float(resolution)
        self.grid = {}

    def get_grid_cell(self, x, y):
        ix = int(np.floor(x / self.resolution))
        iy = int(np.floor(y / self.resolution))
        return (ix, iy)

    def add_point(self, x, y, z, sem_risk, sem_conf):
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = [float(z), float(sem_risk), float(sem_conf), 1]
        else:
            g = self.grid[cell]
            g[0] += float(z); g[1] += float(sem_risk); g[2] += float(sem_conf); g[3] += 1

    def get_estimates(self):
        h, r, c, n = [], [], [], []
        for (ix, iy), (zsum, rsum, csum, cnt) in self.grid.items():
            cx = (ix + 0.5) * self.resolution
            cy = (iy + 0.5) * self.resolution
            inv = 1.0 / cnt
            h.append([cx, cy, zsum*inv])
            r.append([cx, cy, rsum*inv])
            c.append([cx, cy, csum*inv])
            n.append([cx, cy, cnt])
        return np.array(h), np.array(r), np.array(c), np.array(n)


    def prune_far(self, cx, cy, max_radius_cells):
        to_del = []
        for (gx, gy) in list(self.grid.keys()):
            if (gx - cx)**2 + (gy - cy)**2 > max_radius_cells**2:
                to_del.append((gx, gy))
        for k in to_del:
            del self.grid[k]
class TempGoalGuard:
    def __init__(self):
        self.last_goal_ij = None
        self.last_goal_xy = None
        self.last_switch_t = 0.0
        self.flip_times = []  # timestamps of flips for rate limiting
        self.last_metric = None  # (dist_to_final, goal_risk)

    def allow_switch(self, now, candidate_ij, candidate_xy, candidate_risk, final_xy):
        # 1) cooldown
        if (now - self.last_switch_t) < STEP_config['flip_cooldown_s']:
            return False

        # 2) rate limit
        window = 60.0
        self.flip_times = [t for t in self.flip_times if now - t < window]
        if len(self.flip_times) >= STEP_config['max_flip_per_min']:
            return False

        # 3) improvement gate
        cand_dist = float(np.linalg.norm(np.array(candidate_xy) - np.array(final_xy)))
        cand_risk = float(candidate_risk)
        if self.last_metric is None:
            return True  # first ever set is fine
        prev_dist, prev_risk = self.last_metric

        dist_gain = prev_dist - cand_dist
        risk_gain = prev_risk - cand_risk
        if (dist_gain >= STEP_config['min_improve_dist_m']) or (risk_gain >= STEP_config['min_improve_risk']):
            return True
        return False

    def commit(self, now, ij, xy, risk, final_xy):
        self.last_goal_ij = ij
        self.last_goal_xy = xy
        self.last_switch_t = now
        self.flip_times.append(now)
        self.last_metric = (
            float(np.linalg.norm(np.array(xy) - np.array(final_xy))),
            float(risk)
        )
def path_cost(path_idx, cost_map):
    if path_idx is None:
        return float('inf')
    rows, cols = cost_map.shape
    total = 0.0
    for (r, c) in path_idx:
        if 0 <= r < rows and 0 <= c < cols:
            total += float(cost_map[r, c])
        else:
            total += 1e6  # off-grid = very expensive
    return total

    
STEP_config ={
    # MAP
    'grid_margin': 8,
    'grid_resolution': 0.1,
    'radius_filter': 12,

    # RISK
    'max_height_diff': 0.25, 
    'max_slope_degrees': 70.0,
    'risk_radius': 0.1,

    'step_weight': 2.0,
    'slope_weight': 2.0,
    'z_norm_weight': 2.0,
    'interpolate_radius': 1.5,
    'cvar_a': 0.5,
    'cvar_radius': 4.0,
    'distance_ignored': 9.0,

    # a star
    'distance_to_temp': 5.0,
    'distance_to_goal': 1,
    # NMPC
    'N-npmc': 20,
    'Vmax-nmpc': 0.8,
    'delta-nmpc': 25,

    # others for replan:
    'HIGH_RISK': 0.6,
    'MAX_RTSK_VALUE': 50,
    'visualize': True,
    'Capturing': False,
    'MAX_ITER': 350,
    'flip_cooldown_s': 2.0,           # lock temp goal for this long after a switch
    'min_improve_dist_m': 0.8,        # must get at least this much closer to final goal
    'min_improve_risk': 5.0,          # and reduce risk by this much to justify switching
    'max_flip_per_min': 6,            # anti-oscillation rate limit
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xgoal', type=float, default=17.0, help='X coordinate of the goal point')
    parser.add_argument('--ygoal', type=float, default=-7.0, help='Y coordinate of the goal point')
    parser.add_argument('--name', type=str, default='step', help='Name of the experiment')
    parser.add_argument('--maxiter', type=int, default=700, help='Maximum number of iterations')
    args = parser.parse_args()
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(False, 'CPHusky')
    seg = RandlaGroundSegmentor(device=None, subsample_grid=0.1)
    STEP_config['MAX_ITER'] = int(args.maxiter)
    
    # Map setup:
    pos, _ = lidar_test.get_vehicle_pose()
    start_point = pos[:2]
    destination_point = np.array([args.xgoal, args.ygoal])
    x_edges, y_edges, x_mid, y_mid = get_map_setting(start_point, destination_point, margin=STEP_config['grid_margin'], grid_resolution=STEP_config['grid_resolution'])
    X, Y = np.meshgrid(x_mid, y_mid, indexing='ij')
    grid_map_ground = GridMap(resolution=STEP_config['grid_resolution'])


    # Setup NMPC and plot
    nmpc = NMPCController(horizon=STEP_config['N-npmc'],
                          wheelbase=0.25,
                          V_max=STEP_config['Vmax-nmpc'],
                          delta_max=np.deg2rad(STEP_config['delta-nmpc']))
    ctr = airsim.CarControls()
    temp_guard = TempGoalGuard()
    if STEP_config['visualize']:
        colors = [
            (0.5, 0.5, 0.5),  # gray
            (1.0, 1.0, 0.0),  # yellow
            (1.0, 0.5, 0.0),  # orange
            (1.0, 0.0, 0.0),  # red
            (0.0, 0.0, 0.0),  # black
        ]
        cmap = LinearSegmentedColormap.from_list(
            "gray_yellow_orange_red_black", colors, N=50
        )
        fig, ax = plt.subplots(); plt.ion(); 
        colorbar = None
    prev_grid = None
    prev_path = None
    temp_goal_idx = None     
    temp_dest_xy  = None  
    i0_prev = 0 
    last_viz_t = 0.0
    struct = generate_binary_structure(2,1)
    prev_t = time.time()
    stats_dict ={
        'count': 0,
        'collision_count':0,
        'total_length':[],
        'dist_to_goal': None,
        'reach_goal': False,
        'current_pos': None,
        'collision_info': []
    }
    dt_filt = None
    LAT = 0.15  
    distance_last = np.linalg.norm(destination_point - np.array([pos[0], pos[1]]))
    stats_dict['dist_to_goal'] = distance_last
    last_pos = start_point.copy()
    # Persistent annulus mask across iterations (clears where support appears)
    persist_annulus_mask = None
    # Before the loop:
    persist_annulus_mask = None
    prev_risk_grid = None
    x_mid_prev = x_mid
    y_mid_prev = y_mid

    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is None:
                continue
            
            # Process point cloud.
            points = np.asarray(point_cloud_data[:, :3])
            points = points[np.linalg.norm(points, axis=1) > 0.6]
            pos, R = lidar_test.get_vehicle_pose()
            vehicle_x, vehicle_y = pos[0], pos[1]
            veh_xy = np.array([vehicle_x, vehicle_y]) 
            if needs_recentering(veh_xy, destination_point, x_edges, y_edges,
                                buffer=STEP_config['grid_resolution']*10):
                ### NEW: stash old grid state (before you overwrite edges/mids)
                old_x_mid = x_mid.copy()
                old_y_mid = y_mid.copy()
                old_persist_mask = persist_annulus_mask.copy() if persist_annulus_mask is not None else None
                old_prev_risk = prev_risk_grid.copy() if 'prev_risk_grid' in locals() else None

                # your existing recentering
                x_edges, y_edges, x_mid, y_mid = get_map_setting(
                    veh_xy, destination_point,
                    margin=STEP_config['grid_margin'],
                    grid_resolution=STEP_config['grid_resolution']
                )
                X, Y = np.meshgrid(x_mid, y_mid, indexing='ij')

                # indices from the old grid are invalid; force fresh target + plan
                prev_path = None
                temp_goal_idx = None
                trigger_temp_dest = True

                ### NEW: remap old mask/values into the new grid coordinates
                new_shape = (len(x_mid), len(y_mid))
                remapped_mask = remap_mask(
                    old_persist_mask, old_x_mid, old_y_mid,
                    x_edges, y_edges, new_shape
                )
                remapped_prev_risk = remap_values(
                    old_prev_risk, old_x_mid, old_y_mid,
                    x_edges, y_edges, new_shape, reducer=np.nanmax
                )

                # keep them available after recenter; will be merged below with ring/support
                persist_annulus_mask = remapped_mask
                prev_risk_grid = remapped_prev_risk

            # Record stats
            distance_travelled = np.linalg.norm(last_pos - np.array([vehicle_x, vehicle_y]))
            stats_dict['total_length'].append(distance_travelled)
            last_pos = veh_xy.copy() 

            points_world = lidar_test.transform_to_world(points, pos.astype(points.dtype), R)
            points_world[:, 2] = -points_world[:, 2]
            # Use network outputs
            labels, all_probs = seg.segment(points_world)   # labels: [M], all_probs: [M, C]
            p = all_probs                                   # already softmaxed
            sem_risk_point = labels.astype(float)           # shape [M], scalar per point

            # --- Confidence from entropy (0..1; 0=confident, 1=uncertain) ---
            sem_conf_point = 1.0 - softmax_entropy(p)       # shape [M], scalar per point

            # Add to grid (expects scalars)
            for (x, y, z), r, c in zip(points_world, sem_risk_point, sem_conf_point):
                grid_map_ground.add_point(x, y, z, r, c)

           # vehicle index in grid coordinates
            veh_gx = int(np.floor(vehicle_x / STEP_config['grid_resolution']))
            veh_gy = int(np.floor(vehicle_y / STEP_config['grid_resolution']))
            max_radius_cells = int(STEP_config['radius_filter'] / STEP_config['grid_resolution'])
            # grid_map_ground.prune_far(veh_gx, veh_gy, max_radius_cells)

            ground_points, semrisk_points, semconf_points, count_points = grid_map_ground.get_estimates()

            if ground_points.size == 0: continue
            Z_ground, _, _, _ = binned_statistic_2d(
            ground_points[:,0], ground_points[:,1], ground_points[:,2],
                statistic='mean', bins=[x_edges, y_edges]
            )

            # Semantic risk (already 0..50)
            sem_risk_grid, _, _, _ = binned_statistic_2d(
                semrisk_points[:,0], semrisk_points[:,1], semrisk_points[:,2],
                statistic='mean', bins=[x_edges, y_edges]
            )

            # Semantic confidence (0..1) + support
            sem_conf_mean, _, _, _ = binned_statistic_2d(
                semconf_points[:,0], semconf_points[:,1], semconf_points[:,2],
                statistic='mean', bins=[x_edges, y_edges]
            )
            sem_count, _, _, _ = binned_statistic_2d(
                count_points[:,0], count_points[:,1], count_points[:,2],
                statistic='mean', bins=[x_edges, y_edges]
            )
            # sem_conf_grid = np.nan_to_num(sem_conf_mean) * np.tanh(np.nan_to_num(sem_count) / 8.0)
            sem_conf_grid = np.nan_to_num(sem_conf_mean)
            # Calculate risk grids.
            non_nan_indices = np.argwhere(~np.isnan(Z_ground))
            step_risk_grid, slope_risk_grid = calculate_combined_risks(
                Z_ground, non_nan_indices, max_height_diff=STEP_config['max_height_diff'], max_slope_degrees=STEP_config['max_slope_degrees'], radius=STEP_config['risk_radius']
            )
            geom_risk01 = fuse_geom_edge_preserving(step_risk_grid, slope_risk_grid, tau=0.35, beta=1.0)
            geom_risk_grid = np.clip(geom_risk01 * STEP_config['MAX_RTSK_VALUE'], 0, STEP_config['MAX_RTSK_VALUE'])

            # geometric confidence from coverage/consistency (your function)
            N_ground, _, _, _ = binned_statistic_2d(
                ground_points[:,0], ground_points[:,1], ground_points[:,2],
                statistic='count', bins=[x_edges, y_edges]
            )
            geom_conf_grid = step_risk_confidence(
                Z_mean_grid=Z_ground,
                N_count_grid=N_ground,
                grid_resolution=STEP_config['grid_resolution'],
                radius_m=STEP_config['risk_radius']+0.4,
                z_noise_std=0.02,
                w_cov=0.4, w_sup=0.3, w_cons=0.3, k_sup=8.0
            )
            eps = 1e-8

            # 1) Normal confidence-weighted average (like before)
            num = (geom_conf_grid * np.nan_to_num(geom_risk_grid) +
                sem_conf_grid  * np.nan_to_num(sem_risk_grid))
            den = np.maximum(geom_conf_grid + sem_conf_grid, eps)
            fused_avg = num / den

            # 2) Risk inflation penalty (uncertainty → push toward worst case)
            # use the *highest confidence* among the two as the "trust level"
            max_conf = np.nanmax(np.stack([geom_conf_grid, sem_conf_grid]), axis=0)
            penalty = (1.0 - max_conf) * STEP_config['MAX_RTSK_VALUE'] * 0.1  # 0.3 = inflation factor

            # 3) Final fused risk
            total_risk_grid = fused_avg + penalty
            total_risk_grid = np.clip(total_risk_grid, 0, STEP_config['MAX_RTSK_VALUE'])

            # 4) Same interpolation and CVaR as before
            total_risk_grid = interpolate_in_radius(total_risk_grid, STEP_config['interpolate_radius'])
            masked_total_risk_grid = ma.masked_invalid(total_risk_grid)
            risk_grid = compute_cvar_cellwise(masked_total_risk_grid,
                                            alpha=STEP_config['cvar_a'],
                                            radius=STEP_config['cvar_radius'])
            # capture where CVaR produced NaNs before any filling
            nan_mask_initial = np.isnan(risk_grid)


            # Mask cells far from the vehicle.
            distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)
            # Ensure mask shape matches grid shape after any recentering
            if distance_from_vehicle.shape != risk_grid.shape:
                if distance_from_vehicle.T.shape == risk_grid.shape:
                    distance_from_vehicle = distance_from_vehicle.T
                else:
                    X, Y = np.meshgrid(x_mid, y_mid, indexing='ij')
                    distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)
            
            # Exclude cells with lidar support from the annulus
            # ...
            # Exclude cells with lidar support from the annulus
            support_mask = np.nan_to_num(N_ground) > 0
            ring_mask = (distance_from_vehicle >= 1.7) & (distance_from_vehicle <= 3.475)

            # Ensure shape correctness (already handled above when rebuilding X,Y with indexing='ij')
            if (persist_annulus_mask is None) or (persist_annulus_mask.shape != risk_grid.shape):
                # start from remapped (if any) OR zeros
                base_mask = persist_annulus_mask if (persist_annulus_mask is not None and persist_annulus_mask.shape == risk_grid.shape) else np.zeros_like(risk_grid, dtype=bool)
                persist_annulus_mask = (base_mask | ring_mask) & (~support_mask)
            else:
                persist_annulus_mask = (persist_annulus_mask | ring_mask) & (~support_mask)


            # Carry over previous values inside persistent annulus into current NaNs (shape-safe)
            if 'prev_risk_grid' in locals() and prev_risk_grid is not None and prev_risk_grid.shape == risk_grid.shape:
                carry_mask = nan_mask_initial & persist_annulus_mask
                risk_grid[carry_mask] = prev_risk_grid[carry_mask]

            # Fill remaining NaNs in persistent annulus to MAX_RTSK_VALUE
            risk_grid[nan_mask_initial & persist_annulus_mask] = STEP_config['MAX_RTSK_VALUE']
            prev_risk_grid = risk_grid.copy()
            risk_grid = np.nan_to_num(risk_grid, nan=25.0)
           
            with np.errstate(invalid='ignore'):
                max_val  = STEP_config['MAX_RTSK_VALUE']

                # robust high-risk definition (treat non-finite as high too if desired)
                high_thr = 0.9 * np.nanmax(risk_grid)
                finite   = np.isfinite(risk_grid)
                high_mask = finite & (risk_grid >= high_thr)
                if STEP_config.get('prox_treat_nan_as_high', False):
                    high_mask |= ~finite  # optional: NaN areas behave as "high" blobs

                outside = (~high_mask) & finite

                # distance from boundary, in meters (0 exactly at the first outside cell):
                dist_cells = distance_transform_edt(~high_mask)   # 0 on boundary outside
                dist_m = dist_cells * STEP_config['grid_resolution']

                # parameters
                radius_m   = float(STEP_config.get('inflation_radius', 0.8))   # halo width
                boundary_p = float(STEP_config.get('halo_boundary_level', 0.7))  # 0..1 of MAX at boundary
                profile    = STEP_config.get('inflation_profile', 'smoothstep')   # 'smoothstep'|'gaussian'|'poly'

                # set boundary level explicitly vs current local value
                boundary_p = float(STEP_config.get('halo_boundary_level', 0.7))
                max_val = STEP_config['MAX_RTSK_VALUE']
                boundary_target = boundary_p * max_val

                # Keep fades tied to boundary, not absolute zero, so holes remain "repulsive"
                t = np.clip(dist_m / max(radius_m, 1e-6), 0.0, 1.0)
                s = t*t*(3.0 - 2.0*t)  # smoothstep
                w = 1.0 - s            # 1 at boundary, 0 by radius
                want = np.clip(boundary_target - risk_grid, 0.0, max_val)
                add_cost = w * want
                risk_grid[outside] = np.clip(risk_grid[outside] + add_cost[outside], 0.0, max_val)


            trigger_temp_dest = False
            trigger_temp_dest = False
            valid = np.argwhere(~np.isnan(risk_grid))

            # --- Select temp goal periodically or when close to previous one ---
            if (temp_goal_idx is None or
                (temp_dest_xy is not None and
                np.hypot(vehicle_x - temp_dest_xy[0],
                        vehicle_y - temp_dest_xy[1]) < STEP_config['distance_to_temp']) or
                stats_dict['count'] % 20 == 0):

                if valid.size > 0:
                    centers = np.column_stack((x_mid[valid[:, 0]], y_mid[valid[:, 1]]))
                    dists_to_goal = np.linalg.norm(centers - destination_point, axis=1)
                    best = valid[np.argmin(dists_to_goal)]
                    temp_goal_idx = (int(best[0]), int(best[1]))
                    temp_dest_xy = (float(x_mid[temp_goal_idx[0]]),
                                    float(y_mid[temp_goal_idx[1]]))
                    trigger_temp_dest = True

            # --- Define start index ---
            rows, cols = risk_grid.shape
            raw_si = np.digitize(vehicle_x, x_edges) - 1
            raw_sj = np.digitize(vehicle_y, y_edges) - 1
            start_idx = (int(np.clip(raw_si, 0, rows - 1)),
                        int(np.clip(raw_sj, 0, cols - 1)))

            # --- Default goal ---
            # --- Default goal ---
            if temp_goal_idx is None:
                goal_idx = start_idx
            else:
                gi, gj = temp_goal_idx
                goal_val = risk_grid[gi, gj]

                # --- if temp goal is NOT high-risk -> find closest safe cell near the final goal ---
                if goal_val < STEP_config['MAX_RTSK_VALUE'] * 0.95:
                    safe_mask = (~np.isnan(risk_grid)) & (risk_grid < STEP_config['MAX_RTSK_VALUE'] * 0.95)
                    safe_cells = np.argwhere(safe_mask)
                    if safe_cells.size > 0:
                        centers = np.column_stack((x_mid[safe_cells[:, 0]], y_mid[safe_cells[:, 1]]))
                        dists_to_final = np.linalg.norm(centers - destination_point, axis=1)
                        nearest_safe = safe_cells[np.argmin(dists_to_final)]
                        gi, gj = int(nearest_safe[0]), int(nearest_safe[1])
                        # (DON'T assign temp_goal_idx/temp_dest_xy here yet)

                # --- if temp goal IS at moderate/high risk (~50%) -> trigger flip-around behavior ---
                # --- if temp goal IS very high risk -> pick a "flip-around" cell at ~MAX-1 risk ---
                # --- if temp goal IS very high risk -> pick an outside rim point behind the blob ---
                elif goal_val >= STEP_config['MAX_RTSK_VALUE'] * 0.95:
                    MAX = float(STEP_config['MAX_RTSK_VALUE'])
                    safe_thr = 0.95 * MAX

                    # --- find the high-risk component that blocks the way (the one holding (gi, gj)) ---
                    import numpy as _np
                    from scipy.ndimage import label as _label, binary_dilation as _bd, generate_binary_structure as _gbs
                    struct = _gbs(2, 1)

                    high_mask = _np.isfinite(risk_grid) & (risk_grid >= safe_thr)

                    # If our current goal cell isn't actually high (rare race), snap to nearest high to final goal
                    if not high_mask[gi, gj]:
                        # nearest high to final goal
                        hi, hj = _np.nonzero(high_mask)
                        if hi.size > 0:
                            centers = _np.column_stack((x_mid[hi], y_mid[hj]))
                            gxy = _np.asarray(destination_point, dtype=float)
                            best = _np.argmin(_np.linalg.norm(centers - gxy, axis=1))
                            gi, gj = int(hi[best]), int(hj[best])

                    labeled, ncc = _label(high_mask, structure=struct)
                    if ncc > 0:
                        comp_id = labeled[gi, gj] if labeled[gi, gj] != 0 else 0
                    else:
                        comp_id = 0

                    if comp_id != 0:
                        comp_mask = (labeled == comp_id)
                    else:
                        # fallback: treat the single cell as component
                        comp_mask = _np.zeros_like(high_mask, dtype=bool)
                        comp_mask[gi, gj] = True

                    # --- build an outside rim (1 cell thick) ---
                    rim = _bd(comp_mask, structure=struct) & (~comp_mask)

                    # --- flipped half-plane w.r.t. goal -> vehicle direction ---
                    gxy = _np.asarray(destination_point, dtype=float)
                    vxy = _np.asarray([vehicle_x, vehicle_y], dtype=float)
                    g_to_v = vxy - gxy
                    g_to_v /= (_np.linalg.norm(g_to_v) + 1e-9)

                    ii, jj = _np.indices(risk_grid.shape)
                    centers = _np.column_stack((x_mid[ii.ravel()], y_mid[jj.ravel()]))
                    dirs = centers - gxy
                    dirs /= (_np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
                    dots = (dirs @ g_to_v).reshape(risk_grid.shape)
                    flipped_mask = (dots <= 0.0)

                    # --- candidate rim cells: outside, safe, flipped side ---
                    cand_mask = rim & _np.isfinite(risk_grid) & (risk_grid < safe_thr) & flipped_mask
                    cand_cells = _np.argwhere(cand_mask)

                    # If none, loosen by dilating rim once more
                    if cand_cells.size == 0:
                        rim2 = _bd(rim, structure=struct) & (~comp_mask)
                        cand_mask = rim2 & _np.isfinite(risk_grid) & (risk_grid < safe_thr) & flipped_mask
                        cand_cells = _np.argwhere(cand_mask)

                    if cand_cells.size > 0:
                        # Prefer candidates with the largest clearance from the high blob, then lower risk
                        from scipy.ndimage import distance_transform_edt as _dt
                        # distance measured OUTSIDE the blob
                        dist_from_blob = _dt(~comp_mask)
                        scores = dist_from_blob[cand_cells[:, 0], cand_cells[:, 1]]  # larger is better
                        # tie-breaker: prefer lower risk
                        risks = risk_grid[cand_cells[:, 0], cand_cells[:, 1]]
                        order = _np.lexsort((risks, -scores))  # maximize clearance, then minimize risk
                        best = cand_cells[order[0]]
                        gi, gj = int(best[0]), int(best[1])
                    # else: keep (gi, gj) as-is (fallback)

                # ---- direct assignment (no temp_guard) ----
                gi = int(np.clip(gi, 0, rows - 1))
                gj = int(np.clip(gj, 0, cols - 1))
                goal_idx = (gi, gj)
                temp_goal_idx = goal_idx
                temp_dest_xy  = (float(x_mid[gi]), float(y_mid[gj]))
                trigger_temp_dest = True




            # ------------ plan ------------
            planner = AStarPlanner(risk_grid)

            # --- Hysteresis + stickiness on replans ---
            max_risk_value = np.nanmax(risk_grid)
            hr = (risk_grid >= STEP_config['HIGH_RISK'] * max_risk_value)
            hr_dilated = binary_dilation(hr, structure=struct, iterations=3)

            NEED_BLOCKED = False
            if prev_path is not None:
                for (r, c) in prev_path:
                    if 0 <= r < hr_dilated.shape[0] and 0 <= c < hr_dilated.shape[1]:
                        if hr_dilated[r, c]:
                            NEED_BLOCKED = True
                            break

            # Cost of keeping the old path
            OLD_COST = path_cost(prev_path, planner.cost_map) if prev_path is not None else float('inf')

            # Candidate new path and its cost
            NEW_PATH = planner.plan(start_idx, goal_idx, STEP_config['MAX_RTSK_VALUE'])
            NEW_COST = path_cost(NEW_PATH, planner.cost_map)

            # Only switch if new path is clearly better (15% by default), or we must
            IMPROVES_ENOUGH = (OLD_COST - NEW_COST) > 0.15 * OLD_COST

            if (prev_path is None) or NEED_BLOCKED or IMPROVES_ENOUGH or trigger_temp_dest:
                path_idx = NEW_PATH
            else:
                path_idx = prev_path

            # Fallback if we ended up with no path
            if path_idx is None:
                try:
                    cost_map = np.copy(planner.cost_map)
                    high_risk_thresh = STEP_config['HIGH_RISK'] * np.nanmax(risk_grid)
                    cost_map[np.isinf(cost_map)] = 1e6
                    cost_map[risk_grid >= high_risk_thresh] *= 10
                    cost_map = np.clip(cost_map, 0, 1e6)
                    path, _ = route_through_array(cost_map, start_idx, goal_idx, fully_connected=True)
                    path_idx = path
                except Exception:
                    path_idx = [start_idx]

            # After you compute path_idx
            touches_border = any(r in (0, rows-1) or c in (0, cols-1) for r,c in path_idx[-min(10, len(path_idx)):])
            if touches_border:
                trigger_temp_dest = True
            prev_path = list(path_idx)
            raw_coords = np.array([[x_mid[r], y_mid[c]] for r, c in path_idx])
       

            def arc_length_parametrize(P, ds=0.1):
                seg = np.diff(P, axis=0)
                s   = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
                S   = np.arange(0, s[-1]+1e-9, ds)
                # cubic splines (C²)
                from scipy.interpolate import CubicSpline
                sx = CubicSpline(s, P[:,0], bc_type='clamped')
                sy = CubicSpline(s, P[:,1], bc_type='clamped')
                X  = np.column_stack([sx(S), sy(S)])
                # derivatives for heading/curvature
                dx, dy   = sx(S, 1), sy(S, 1)
                ddx, ddy = sx(S, 2), sy(S, 2)
                psi      = np.arctan2(dy, dx)
                kappa    = (dx*ddy - dy*ddx) / np.maximum((dx*dx + dy*dy)**1.5, 1e-6)
                return X, psi, kappa

            smoothed_path, path_psi, path_kappa = arc_length_parametrize(raw_coords, ds=0.15)

            # smoothed_path = smooth_path(raw_coords, window_size=5)
     
            # Compute dt
            dt_loop = max(min(time.time() - prev_t, 0.2), 0.05)
            prev_t = time.time()

            ## NMPC
            if len(smoothed_path) == 0:
                continue
            dists = np.linalg.norm(smoothed_path - pos[:2], axis=1)
            SEARCH_BACK, SEARCH_AHEAD = 2, 25
            s0 = max(i0_prev - SEARCH_BACK, 0)
            s1 = min(i0_prev + SEARCH_AHEAD, len(smoothed_path) - 1)
            window = dists[s0:s1+1]
            if window.size == 0 or not np.isfinite(window).any():
                i0 = i0_prev
            else:
                i0 = s0 + int(np.argmin(window))
            i0 = max(i0, i0_prev - SEARCH_BACK)   # prevent big backward jumps
            i0 = int(np.clip(i0, 0, len(smoothed_path) - 1))
            i0_prev = i0

            now = time.time()
            dt_meas = np.clip(now - prev_t, 0.05, 0.2)  # actual loop period
            prev_t = now                                 # move the baseline forward

            # low-pass to avoid jitter
            if dt_filt is None:
                dt_filt = dt_meas
            dt_filt = 0.8*dt_filt + 0.2*dt_meas

            # latency → index lead
            n_shift = int(round(LAT / max(dt_filt, 1e-3)))
            lead = 1 + n_shift

            # pass a constant dt_seq to the solver for this horizon
            dt_seq = [dt_meas]*nmpc.N

            # extract NMPC reference: next N waypoints
            ref_pts = smoothed_path[i0+1 : i0+1+nmpc.N]
            if len(ref_pts) < nmpc.N and len(ref_pts) > 0:
                ref_pts = np.vstack((ref_pts, np.tile(ref_pts[-1], (nmpc.N - len(ref_pts), 1))))
            elif len(ref_pts) == 0:
                ref_pts = np.tile(smoothed_path[-1], (nmpc.N, 1))

            # 2) solve NMPC
            psi0 = math.atan2(R[1,0], R[0,0])
            x0   = np.array([vehicle_x, vehicle_y, psi0])
            try:
                # After you have raw path indices -> convert to world coords raw_coords
                S, XY, PSI, KAP = build_arc_length_path(raw_coords, ds=0.15)

                # base curvature speed
                v_base = curvature_speed(KAP, v_max=STEP_config['Vmax-nmpc'], a_lat_max=1.0, v_min=0.15)

                # optional risk scaling
                v_prof = risk_scaled_speed(XY, risk_grid, X, Y, v_base, max_risk=STEP_config['MAX_RTSK_VALUE'], scale=0.5)

                # latency-aware index lead (see §4)
                n_shift = round(LAT / dt_filt )
                lead = 1 + n_shift   # compute n_shift in §4
                ref_xy, ref_psi, ref_v, i0 = pick_reference_window(XY, PSI, v_prof, pos[:2], nmpc.N, lead_idx=lead)

                x0 = np.array([vehicle_x, vehicle_y, psi0])
                dt_seq = [dt_loop]*nmpc.N
                v_cmd, d_cmd = nmpc.solve(x0, ref_xy, ref_psi, ref_v, dt_seq)

            except Exception:
                v_cmd, d_cmd = (0.0, 0.0)

            # 3) desired heading from path tangent (fix #4)
            LOOKAHEAD_STEPS = 4
            j = min(i0 + LOOKAHEAD_STEPS, len(smoothed_path) - 1)
            dx = smoothed_path[j,0] - smoothed_path[i0,0]
            dy = smoothed_path[j,1] - smoothed_path[i0,1]
            des_ψ = math.atan2(dy, dx)
            err_ψ = math.atan2(math.sin(des_ψ - psi0), math.cos(des_ψ - psi0))

            # your existing big-turn branch (unchanged)
            if abs(err_ψ) > np.deg2rad(20):
                ctr.steering = np.clip(err_ψ/np.deg2rad(20), -1, 1)
                ctr.throttle = 0.0
            else:
                ctr.steering = float(np.clip(d_cmd / nmpc.delta_max, -1, 1))
                scale        = 1 - 0.8*abs(err_ψ)/np.deg2rad(20)
                ctr.throttle = float(np.clip(v_cmd*scale / nmpc.V_max, 0, nmpc.V_max))

            lidar_test.client.setCarControls(ctr)

            
            now_viz = time.time()
            if STEP_config['visualize'] and (now_viz - last_viz_t) >= STEP_config.get('viz_dt', 0.25):
                ax.clear()
                c = ax.pcolormesh(Y, X, risk_grid, shading='auto',
                                  cmap=cmap, alpha=0.7,
                                  vmin=0, vmax=STEP_config['MAX_RTSK_VALUE'])
                if colorbar is None:
                    colorbar = fig.colorbar(c, ax=ax, label='Risk')
                else:
                    colorbar.update_normal(c)
                colorbar.set_ticks(np.linspace(0, STEP_config['MAX_RTSK_VALUE'], 6))  # 0,10,...,50
                ax.scatter(vehicle_y, vehicle_x, c='green', s=35, label='Vehicle')
                ax.scatter(destination_point[1], destination_point[0], c='red', s=50, label='Goal')
                # heading arrow (plot axes are Y on X-axis, X on Y-axis)
                arrow_len = 0.9
                dx = np.sin(psi0)
                dy = np.cos(psi0)
                dx, dy = (dx, dy) / np.hypot(dx, dy) * arrow_len  # normalize for consistency

                ax.quiver(vehicle_y, vehicle_x, dx, dy,
                        angles='xy', scale_units='xy', scale=1.0,
                        color='green', width=0.012,
                        pivot='tail', headwidth=5, headlength=5, headaxislength=5)

                ax.set_aspect('equal', adjustable='box')
                if temp_dest_xy is not None:
                    ax.scatter(temp_dest_xy[1], temp_dest_xy[0], c='black', s=30, marker='s',
                               linewidth=0.15, label='Temp Goal')
                ax.plot(smoothed_path[:,1], smoothed_path[:,0], color='blue', linewidth=2, label='Smoothed A* Path')
                ax.plot(ref_pts[:,1], ref_pts[:,0], 'r--', linewidth=1, label='Reference Trajectory')
                # ax.legend()
                plt.draw(); plt.pause(0.001)
                last_viz_t = now_viz
                if STEP_config['Capturing']:
                    path = os.path.join(base, "record/random_4", args.name)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    plt.savefig(os.path.join(path, f'{stats_dict["count"]}.png'))

                    # save the car state
                    car_state = lidar_test.client.getCarState()
                    car_state_filename = os.path.join(path, f'{stats_dict["count"]}_car_state.json')
                    car_state_dict = serialize(car_state)
                    with open(car_state_filename, "w") as f:
                        json.dump(car_state_dict, f, indent=2)
            pos, R = lidar_test.get_vehicle_pose()
            vehicle_x, vehicle_y = pos[0], pos[1]

            # flip check -> hard stop + classify as failed
            if is_flipped(R, up_z_threshold=0.3, angle_deg_threshold=85.0):
                try:
                    lidar_test.client.setCarControls(airsim.CarControls(throttle=0.0, steering=0.0), lidar_test.vehicleName)
                except Exception:
                    pass
                stats_dict['reach_goal'] = False
                stats_dict['current_pos'] = [vehicle_x, vehicle_y]
                stats_dict.setdefault('failure_reason', 'flipped_over')
                print("Detected flip-over → terminating and classifying as failed.")
                break
            # Record REST INFO
            stats_dict['count'] += 1
            if lidar_test.client.simGetCollisionInfo().has_collided:
                stats_dict['collision_count'] += 1
                ci = lidar_test.client.simGetCollisionInfo()
                stats_dict['collision_info'].append(serialize(ci))
            distance_last = np.linalg.norm(destination_point - np.array([vehicle_x, vehicle_y]))
            # stats_dict['dist_to_goal'].append(distance_last)
            if distance_last < STEP_config['distance_to_goal']:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0), lidar_test.vehicleName)
                stats_dict['reach_goal'] = True
                stats_dict['current_pos'] = [vehicle_x, vehicle_y]
                break
            elif stats_dict['count'] >= STEP_config['MAX_ITER']:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0), lidar_test.vehicleName)
                stats_dict['reach_goal'] = False
                stats_dict['current_pos'] = [vehicle_x, vehicle_y]
                break
    finally:
        print("-----------------------------------------------")
        print("Reached Destination")
        print("Count: ", stats_dict['count'])
        print("collision_count: ", stats_dict['collision_count'])
        print("dist_to_goal: ", stats_dict['dist_to_goal'])
        print("total_length: ", np.sum(stats_dict['total_length']))
        print("current_pos: ", stats_dict['current_pos'])
        # lidar_test.client.enableApiControl(False, lidar_test.vehicleName)
        print("--------------Done--------------")

    
    # path to your “master” stats file
    os.chdir(base)
    stats_file = os.path.join(base, "record/random_4.json")
    all_runs = []
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            all_runs = json.load(f)
    len_run = len(all_runs)
    # append this run’s stats
    all_runs.append({
        "num": len(all_runs) + 1,
        "reach_goal": stats_dict['reach_goal'],
        "start_point": start_point.tolist(),        # e.g. [x0, y0]
        "goal_point": destination_point.tolist(),   # e.g. [xg, yg]
        "count": stats_dict["count"],
        "collision_count": stats_dict["collision_count"],
        "total_length": stats_dict["total_length"],
        "dist_to_goal": stats_dict["dist_to_goal"],
        "current_pos": stats_dict["current_pos"],
        "collision_info": stats_dict["collision_info"]
    })

    if STEP_config['Capturing']:
        with open(stats_file, "w") as f:
            json.dump(all_runs, f, indent=2)

    print(f"Saved stats for this run")