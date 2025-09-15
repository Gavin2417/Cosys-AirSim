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

    
class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        # Store (sum, count) per cell
        self.grid = {}

    def get_grid_cell(self, x, y):
        return (int(np.floor(x / self.resolution)), int(np.floor(y / self.resolution)))


    def add_point(self, x, y, z, label):
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = [z, label, 1]
        else:
            self.grid[cell][0] += z
            self.grid[cell][1] += label
            self.grid[cell][2] += 1

    def get_height_estimate(self):
        height_estimates = []
        label_estimates = []
        for (gx, gy), (z_sum, label_sum, count) in self.grid.items():
            mean_z = float(z_sum/count)
            mean_label = float(label_sum/count)
            cx = (gx + 0.5) * self.resolution
            cy = (gy + 0.5) * self.resolution
            height_estimates.append([cx, cy, mean_z])
            label_estimates.append([cx, cy, mean_label])
        return np.array(height_estimates), np.array(label_estimates)
    def prune_far(self, cx, cy, max_radius_cells):
        to_del = []
        for (gx, gy) in self.grid.keys():
            if (gx - cx)**2 + (gy - cy)**2 > max_radius_cells**2:
                to_del.append((gx, gy))
        for k in to_del:
            del self.grid[k]

    
STEP_config ={
    # MAP
    'grid_margin': 6,
    'grid_resolution': 0.1,
    'radius_filter': 12,

    # RISK
    'max_height_diff': 0.032, 
    'max_slope_degrees': 20.0,
    'risk_radius': 0.5,

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
    args = parser.parse_args()
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')
    seg = RandlaGroundSegmentor(device=None, subsample_grid=0.1)
    
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
    # a circular structuring element if you want Euclidean radius.
    struct = generate_binary_structure(2,1)
    prev_t = time.time()
    stats_dict ={
        'count': 0,
        'collision_count':0,
        'total_length':[],
        'dist_to_goal': None,
        'reach_goal': False,
        'current_pos': None
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

            # Record stats
            distance_travelled = np.linalg.norm(last_pos - np.array([vehicle_x, vehicle_y]))
            stats_dict['total_length'].append(distance_travelled)
            last_pos = veh_xy.copy() 

            points_world = lidar_test.transform_to_world(points, pos.astype(points.dtype), R)
            points_world[:, 2] = -points_world[:, 2]
            labels = seg.segment(points_world)

            # Populate grid maps based on segmentation.
            for i, point in enumerate(points_world):
                x, y, z = point
                grid_map_ground.add_point(x, y, z, labels[i])
           # vehicle index in grid coordinates
            veh_gx = int(np.floor(vehicle_x / STEP_config['grid_resolution']))
            veh_gy = int(np.floor(vehicle_y / STEP_config['grid_resolution']))
            max_radius_cells = int(STEP_config['radius_filter'] / STEP_config['grid_resolution'])
            grid_map_ground.prune_far(veh_gx, veh_gy, max_radius_cells)

            ground_points, label_points = grid_map_ground.get_height_estimate()

            ground_points = filter_points_by_radius(ground_points, veh_xy, STEP_config['radius_filter'])
            label_points = filter_points_by_radius(label_points, veh_xy, STEP_config['radius_filter'])
            if ground_points.size == 0: continue
            Z_ground, _, _, _ = binned_statistic_2d(
                ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], statistic='mean', bins=[x_edges, y_edges]
            )
            randla_risk_grid, _, _, _ = binned_statistic_2d(
                label_points[:,0], label_points[:,1], label_points[:,2], statistic='mean', bins=[x_edges, y_edges]
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
            # Apply fading and transform risk values.
            
            max_risk = np.nanmax(total_risk_grid)
            threshold = 0.20 * max_risk
            mask = total_risk_grid > threshold
            # capo
            total_risk_grid[mask] = np.exp(total_risk_grid[mask])



            total_risk_grid = (total_risk_grid+ randla_risk_grid) / 2.0
            total_risk_grid = interpolate_in_radius(total_risk_grid, STEP_config['interpolate_radius'])
            masked_total_risk_grid = ma.masked_invalid(total_risk_grid)
            risk_grid = compute_cvar_cellwise(masked_total_risk_grid, alpha=STEP_config['cvar_a'], radius=STEP_config['cvar_radius'])
            risk_grid = risk_grid.filled(25)

            # Mask cells far from the vehicle.
            distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)
            risk_grid[distance_from_vehicle > STEP_config['distance_ignored']] = np.nan



            trigger_temp_dest = False
            valid = np.argwhere(~np.isnan(risk_grid))
            if temp_goal_idx is None or (temp_dest_xy is not None and
                                         np.hypot(vehicle_x - temp_dest_xy[0],
                                                  vehicle_y - temp_dest_xy[1]) < STEP_config['distance_to_temp'])or stats_dict['count'] %15 ==0:
                if valid.size > 0:
                    centers = np.column_stack((x_mid[valid[:,0]], y_mid[valid[:,1]]))
                    dists_to_goal = np.linalg.norm(centers - destination_point, axis=1)
                    best = valid[np.argmin(dists_to_goal)]
                    temp_goal_idx = (int(best[0]), int(best[1]))
                    temp_dest_xy  = (float(x_mid[temp_goal_idx[0]]),
                                     float(y_mid[temp_goal_idx[1]]))
                    trigger_temp_dest = True

   
            rows, cols = risk_grid.shape
            raw_si = np.digitize(vehicle_x, x_edges) - 1
            raw_sj = np.digitize(vehicle_y, y_edges) - 1
            start_idx = (int(np.clip(raw_si, 0, rows-1)),
                         int(np.clip(raw_sj, 0, cols-1)))

            if temp_goal_idx is None:
                goal_idx = start_idx
            else:
                gi, gj = temp_goal_idx
                goal_idx = (int(np.clip(gi, 0, rows-1)),
                            int(np.clip(gj, 0, cols-1)))

            # ------------ plan ------------
            planner = AStarPlanner(risk_grid)

            if prev_path is not None and not trigger_temp_dest:
                max_risk_value = np.nanmax(risk_grid)
                hr = (risk_grid >= STEP_config['HIGH_RISK']*max_risk_value)
                hr_dilated = binary_dilation(hr, structure=struct, iterations=3)

                needs_replan = False
                for (r, c) in prev_path:
                    if 0 <= r < hr_dilated.shape[0] and 0 <= c < hr_dilated.shape[1]:
                        if hr_dilated[r, c]:
                            needs_replan = True
                            break

                if not needs_replan:
                    path_idx = prev_path
                else:
                    path_idx = planner.plan(start_idx, goal_idx, STEP_config['MAX_RTSK_VALUE'])
            else:
                path_idx = planner.plan(start_idx, goal_idx, STEP_config['MAX_RTSK_VALUE'])

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

            prev_path = list(path_idx)
            raw_coords = np.array([[x_mid[r], y_mid[c]] for r, c in path_idx])
            smoothed_path = smooth_path(raw_coords, window_size=5)
     
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
                v_cmd, δ_cmd = nmpc.solve(x0, ref_pts, [dt_loop]*nmpc.N)
            except Exception:
                v_cmd, δ_cmd = (0.0, 0.0)

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
                ctr.steering = float(np.clip(δ_cmd / nmpc.delta_max, -1, 1))
                scale        = 1 - 0.8*abs(err_ψ)/np.deg2rad(20)
                ctr.throttle = float(np.clip(v_cmd*scale / nmpc.V_max, 0, nmpc.V_max))

            lidar_test.client.setCarControls(ctr)

            
            # Visualization
            if STEP_config['visualize']:
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
                # plt.draw(); plt.pause(0.1)
                if STEP_config['Capturing']:
                    path = os.path.join(base, "record/combine_2", args.name)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    plt.savefig(os.path.join(path, f'{stats_dict["count"]}.png'))

                    # save the car state
                    car_state = lidar_test.client.getCarState()
                    car_state_filename = os.path.join(path, f'{stats_dict["count"]}_car_state.json')
                    car_state_dict = serialize(car_state)
                    with open(car_state_filename, "w") as f:
                        json.dump(car_state_dict, f, indent=2)
            
            # Record REST INFO
            stats_dict['count'] += 1
            if lidar_test.client.simGetCollisionInfo().has_collided:
                stats_dict['collision_count'] += 1

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
    stats_file = os.path.join(base, "record/combine_stats_2.json")
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
        "current_pos": stats_dict["current_pos"]
    })

    # # write it back out
    with open(stats_file, "w") as f:
        json.dump(all_runs, f, indent=2)

    print(f"Saved stats for this run")