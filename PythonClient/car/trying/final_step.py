import os, math, time, heapq, json, argparse
import numpy as np
import open3d as o3d
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
from scipy.ndimage import label as _label
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
import cosysairsim as airsim
from linefit import ground_seg
from function5 import *
import casadi as ca
from skimage.graph import route_through_array

class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        # Store (sum, count) per cell
        self.grid = {}

    def get_grid_cell(self, x, y):
        return (round(x / self.resolution, 1), round(y / self.resolution, 1))

    def add_point(self, x, y, z, timestamp):
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = [z, 1]
        else:
            self.grid[cell][0] += z
            self.grid[cell][1] += 1

    def get_height_estimate(self):
        estimates = []
        for (gx, gy), (z_sum, count) in self.grid.items():
            mean_z = float(z_sum/count)
            estimates.append([gx * self.resolution, gy * self.resolution, mean_z])
        return np.array(estimates)
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
    'Capturing': True,
    'MAX_ITER': 350,
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xgoal', type=float, default=17.0, help='X coordinate of the goal point')
    parser.add_argument('--ygoal', type=float, default=-7.0, help='Y coordinate of the goal point')
    parser.add_argument('--name', type=str, default='step', help='Name of the experiment')
    parser.add_argument('--maxiter', type=int, default=700, help='Maximum number of iterations')
    args = parser.parse_args()
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')
    STEP_config['MAX_ITER'] = int(args.maxiter)

    # Initialize ground segmentation.
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    config_path = os.path.join(base, "../assets/config.toml")
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default parameters")
        groundseg = ground_seg()
    else:
        groundseg = ground_seg(config_path)

    # Map setup:
    pos, _ = lidar_test.get_vehicle_pose()
    start_point = pos[:2]
    destination_point = np.array([args.xgoal, args.ygoal])
    x_edges, y_edges, x_mid, y_mid = get_map_setting(start_point, destination_point, margin=STEP_config['grid_margin'], grid_resolution=STEP_config['grid_resolution'])
    X, Y = np.meshgrid(x_mid, y_mid, indexing='ij')
    grid_map_ground = GridMap(resolution=STEP_config['grid_resolution'])
    grid_map_obstacle = GridMap(resolution=STEP_config['grid_resolution'])

    # Setup NMPC and plot
    nmpc = NMPCController(horizon=STEP_config['N-npmc'],
                          wheelbase=0.25,
                          V_max=STEP_config['Vmax-nmpc'],
                          delta_max=np.deg2rad(STEP_config['delta-nmpc']))
    ctr = airsim.CarControls()

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

    persist_annulus_mask = None
    prev_risk_grid = None
    prev_grid = None
    prev_path = None
    temp_goal_idx = None     
    temp_dest_xy  = None 
    dt_filt = None
    LAT = 0.15   
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
    distance_last = np.linalg.norm(destination_point - np.array([pos[0], pos[1]]))
    stats_dict['dist_to_goal'] = distance_last
    last_pos = start_point.copy()

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
                old_x_mid = x_mid.copy()
                old_y_mid = y_mid.copy()
                old_persist_mask = persist_annulus_mask.copy() if persist_annulus_mask is not None else None
                old_prev_risk = prev_risk_grid.copy() if 'prev_risk_grid' in locals() else None
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
                
                remapped_prev_risk = remap_values(
                    old_prev_risk, old_x_mid, old_y_mid,
                    x_edges, y_edges, new_shape, reducer=np.nanmax
                )
                prev_risk_grid = remapped_prev_risk

            # Record stats
            distance_travelled = np.linalg.norm(last_pos - np.array([vehicle_x, vehicle_y]))
            stats_dict['total_length'].append(distance_travelled)
            last_pos = veh_xy.copy() 

            points_world = lidar_test.transform_to_world(points, pos, R)
            points_world[:, 2] = -points_world[:, 2] 
            labels = np.array(groundseg.run(points_world))

            # Populate grid maps based on segmentation.
            for i, point in enumerate(points_world):
                x, y, z = point
                if labels[i] == 1:
                    grid_map_ground.add_point(x, y, z, timestamp)
                elif z > -pos[2]:
                    grid_map_obstacle.add_point(x, y, z, timestamp)
                else:
                    grid_map_ground.add_point(x, y, z, timestamp)
            ground_points = grid_map_ground.get_height_estimate()
            obstacle_points = grid_map_obstacle.get_height_estimate()
            # ground_points = filter_points_by_radius(ground_points, veh_xy, STEP_config['radius_filter'])
            if ground_points.size == 0: continue
            Z_ground, _, _, _ = binned_statistic_2d(
                ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], statistic='mean', bins=[x_edges, y_edges]
            )

            # Calculate risk grids.
            non_nan_indices = np.argwhere(~np.isnan(Z_ground))
            step_risk_grid, slope_risk_grid = calculate_combined_risks(
                Z_ground, non_nan_indices, max_height_diff=STEP_config['max_height_diff'], max_slope_degrees=STEP_config['max_slope_degrees'], radius=STEP_config['risk_radius']
            )
            combined_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
            masked_step_risk = np.ma.masked_array(step_risk_grid, mask=combined_mask) * STEP_config['step_weight']
            masked_slope_risk = np.ma.masked_array(slope_risk_grid, mask=combined_mask) * STEP_config['slope_weight']
            sum_grid = np.ma.filled(masked_step_risk, 0) + np.ma.filled(masked_slope_risk, 0)
            both_nan_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
            total_risk_grid = np.where(both_nan_mask, np.nan, sum_grid)

            # Incorporate obstacle risk.
            if obstacle_points.size != 0:
                # obstacle_points = filter_points_by_radius(obstacle_points, veh_xy, STEP_config['radius_filter'])
                if obstacle_points.size != 0:
                    obs_x_idx = np.clip(np.digitize(obstacle_points[:, 0], x_edges) - 1, 0, len(x_mid)-1)
                    obs_y_idx = np.clip(np.digitize(obstacle_points[:, 1], y_edges) - 1, 0, len(y_mid)-1)
                    total_risk_grid[obs_x_idx, obs_y_idx] = 3.0

            # Apply fading and transform risk values.
            total_risk_grid = fade_with_distance_transform(total_risk_grid,
                                                           high_threshold=0.65,
                                                           fade_scale=4.0,
                                                           sigma=3.0)
            max_risk = np.nanmax(total_risk_grid)
            threshold = 0.20 * max_risk
            mask = total_risk_grid > threshold
            total_risk_grid[mask] = np.exp(total_risk_grid[mask])
            total_risk_grid = interpolate_in_radius(total_risk_grid, STEP_config['interpolate_radius'])
            masked_total_risk_grid = ma.masked_invalid(total_risk_grid)
            risk_grid = compute_cvar_cellwise(masked_total_risk_grid, alpha=STEP_config['cvar_a'], radius=STEP_config['cvar_radius'])
            nan_mask_initial = np.isnan(risk_grid)

            # Calculate distance from vehicle and ensure mask shape matches grid shape after any recentering
            distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)
            if distance_from_vehicle.shape != risk_grid.shape:
                if distance_from_vehicle.T.shape == risk_grid.shape:
                    distance_from_vehicle = distance_from_vehicle.T
                else:
                    X, Y = np.meshgrid(x_mid, y_mid, indexing='ij')
                    distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)

            # Carry over previous values inside persistent annulus into current NaNs (shape-safe)
            if 'prev_risk_grid' in locals() and prev_risk_grid is not None and prev_risk_grid.shape == risk_grid.shape:
                carry_mask = nan_mask_initial 
                risk_grid[carry_mask] = prev_risk_grid[carry_mask]

            prev_risk_grid = risk_grid.copy()
            risk_grid = np.nan_to_num(risk_grid, nan=25.0)
            trigger_temp_dest = False
            valid = np.argwhere(~np.isnan(risk_grid))

            # --- Select temp goal periodically or when close to previous one ---
            if (temp_goal_idx is None or (temp_dest_xy is not None and
                np.hypot(vehicle_x - temp_dest_xy[0], vehicle_y - temp_dest_xy[1]) < STEP_config['distance_to_temp']) or
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
                        
                # --- if temp goal IS at moderate/high risk (~50%) -> trigger flip-around behavior ---
                # --- if temp goal IS very high risk -> pick a "flip-around" cell at ~MAX-1 risk ---
                # --- if temp goal IS very high risk -> pick an outside rim point behind the blob ---
                elif goal_val >= STEP_config['MAX_RTSK_VALUE'] * 0.95:
                    MAX = float(STEP_config['MAX_RTSK_VALUE'])
                    safe_thr = 0.95 * MAX
                    # --- find the high-risk component that blocks the way (the one holding (gi, gj)) ---
                    struct = generate_binary_structure(2, 1)
                    high_mask = np.isfinite(risk_grid) & (risk_grid >= safe_thr)

                    # If our current goal cell isn't actually high (rare race), snap to nearest high to final goal
                    if not high_mask[gi, gj]:
                        # nearest high to final goal
                        hi, hj = np.nonzero(high_mask)
                        if hi.size > 0:
                            centers = np.column_stack((x_mid[hi], y_mid[hj]))
                            gxy = np.asarray(destination_point, dtype=float)
                            best = np.argmin(np.linalg.norm(centers - gxy, axis=1))
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
                        comp_mask = np.zeros_like(high_mask, dtype=bool)
                        comp_mask[gi, gj] = True

                    # --- build an outside rim (1 cell thick) ---
                    rim = binary_dilation(comp_mask, structure=struct) & (~comp_mask)

                    # --- flipped half-plane w.r.t. goal -> vehicle direction ---
                    gxy = np.asarray(destination_point, dtype=float)
                    vxy = np.asarray([vehicle_x, vehicle_y], dtype=float)
                    g_to_v = vxy - gxy
                    g_to_v /= (np.linalg.norm(g_to_v) + 1e-9)

                    ii, jj = np.indices(risk_grid.shape)
                    centers = np.column_stack((x_mid[ii.ravel()], y_mid[jj.ravel()]))
                    dirs = centers - gxy
                    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
                    dots = (dirs @ g_to_v).reshape(risk_grid.shape)
                    flipped_mask = (dots <= 0.0)

                    # --- candidate rim cells: outside, safe, flipped side ---
                    cand_mask = rim & np.isfinite(risk_grid) & (risk_grid < safe_thr) & flipped_mask
                    cand_cells = np.argwhere(cand_mask)

                    # If none, loosen by dilating rim once more
                    if cand_cells.size == 0:
                        rim2 = binary_dilation(rim, structure=struct) & (~comp_mask)
                        cand_mask = rim2 & np.isfinite(risk_grid) & (risk_grid < safe_thr) & flipped_mask
                        cand_cells = np.argwhere(cand_mask)

                    if cand_cells.size > 0:
                        # Prefer candidates with the largest clearance from the high blob, then lower risk
                        # distance measured OUTSIDE the blob
                        dist_from_blob = distance_transform_edt(~comp_mask)
                        scores = dist_from_blob[cand_cells[:, 0], cand_cells[:, 1]]  # larger is better
                        # tie-breaker: prefer lower risk
                        risks = risk_grid[cand_cells[:, 0], cand_cells[:, 1]]
                        order = np.lexsort((risks, -scores))  # maximize clearance, then minimize risk
                        best = cand_cells[order[0]]
                        gi, gj = int(best[0]), int(best[1])

                gi = int(np.clip(gi, 0, rows - 1))
                gj = int(np.clip(gj, 0, cols - 1))
                goal_idx = (gi, gj)
                temp_goal_idx = goal_idx
                temp_dest_xy  = (float(x_mid[gi]), float(y_mid[gj]))
                trigger_temp_dest = True



            # ------------ plan (Hysteresis + stickiness on replans)------------
            planner = AStarPlanner(risk_grid)
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

            # Cost of keeping the old path and candidate new path and their cost
            OLD_COST = path_cost(prev_path, planner.cost_map) if prev_path is not None else float('inf')
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
            S, smoothed_path, path_psi, path_kappa = build_arc_length_path(raw_coords, ds=0.15)
     
            # Compute loop timing once, using a monotonic clock
            now = time.time()
            dt_meas = max(min(now - prev_t, 0.2), 0.05)  # clamp to [0.05, 0.2]
            prev_t = now
            # low-pass filter to reduce jitter
            if dt_filt is None:
                dt_filt = dt_meas
            dt_filt = 0.8*dt_filt + 0.2*dt_meas
            dt_seq = [dt_filt]*nmpc.N

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
                # use precomputed smoothed_path/path_psi/path_kappa
                v_base = curvature_speed(path_kappa, v_max=STEP_config['Vmax-nmpc'], a_lat_max=1.0, v_min=0.15)
                v_prof = risk_scaled_speed(smoothed_path, risk_grid, X, Y, v_base, max_risk=STEP_config['MAX_RTSK_VALUE'], scale=0.5)

                n_shift = int(round(LAT / max(dt_filt, 1e-3)))
                # n_shift = int(min(n_shift, max(0, nmpc.N - 2)))   # cap so 1+n_shift < N
                lead    = 1 + n_shift
                ref_xy, ref_psi, ref_v, i0 = pick_reference_window(smoothed_path, path_psi, v_prof, pos[:2], nmpc.N, lead_idx=lead)

                x0 = np.array([vehicle_x, vehicle_y, psi0])
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
                # plt.draw(); plt.pause(0.001)
                last_viz_t = now_viz
                if STEP_config['Capturing']:
                    path = os.path.join(base, "record/finalstep_3", args.name)
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
    stats_file = os.path.join(base, "record/finalstep_stats_3.json")
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