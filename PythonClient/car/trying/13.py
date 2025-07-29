import os, math, time, heapq
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
from function5 import calculate_combined_risks, compute_cvar_cellwise
import casadi as ca

class NMPCController:
    def __init__(self, horizon=10, dt=0.1, wheelbase=0.5,
                 V_max=0.5, delta_max=np.deg2rad(25)):
        self.N        = horizon
        self.dt       = dt
        self.L        = wheelbase
        self.V_max    = V_max
        self.delta_max= delta_max

        # Tuning weights
        self.Q_pose   = np.diag([5, 5, 2])     # x,y,ψ tracking
        self.R_u      = np.diag([0.01, 0.01])  # v,δ effort
        self.R_du     = 0.05                   # smoothness penalty

         # --- symbols ---
        x, y, ψ = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('ψ')
        states  = ca.vertcat(x, y, ψ)
        v, δ     = ca.SX.sym('v'), ca.SX.sym('δ')
        controls = ca.vertcat(v, δ)

        # dynamics
        rhs = ca.vertcat(v*ca.cos(ψ),
                         v*ca.sin(ψ),
                         v/self.L * ca.tan(δ))
        f   = ca.Function('f', [states, controls], [rhs])

        # decision variables
        X = ca.SX.sym('X', 3, self.N+1)
        U = ca.SX.sym('U', 2, self.N)

        # parameters: [ x0(3), ref_x/ref_y (2*N), dt_seq (N) ] → total 3+3N
        P = ca.SX.sym('P', 3 + 2*self.N + self.N)

        g   = []
        obj = 0

        # initial‐state constraint
        g.append(X[:,0] - P[0:3])

        for k in range(self.N):
            # extract references from P
            xr = P[3 + 2*k]
            yr = P[3 + 2*k + 1]
            # extract dt from P
            dt_k = P[3 + 2*self.N + k]

            st = X[:,k]
            uc = U[:,k]

            # tracking cost
            err = st - ca.vertcat(xr, yr,
                  ca.atan2(yr-st[1], xr-st[0]))
            obj += ca.mtimes([err.T, self.Q_pose, err])

            # control effort
            obj += ca.mtimes([uc.T, self.R_u, uc]) * dt_k

            # smoothness
            if k>0:
                du = U[:,k] - U[:,k-1]
                obj += self.R_du * ca.sumsqr(du)

            # dynamics
            st_next = X[:,k+1]
            fval    = f(st, uc)
            g.append(st_next - (st + dt_k * fval))

        # terminal cost  (unchanged)
        errT = X[:,self.N] - ca.vertcat(
            P[3+2*(self.N-1)],
            P[3+2*(self.N-1)+1],
            0
        )
        Qf   = np.diag([10,10,5])
        obj += ca.mtimes([errT.T, Qf, errT])

        # build the NLP
        G   = ca.vertcat(*g)
        OPT = ca.vertcat(ca.reshape(X, -1,1),
                         ca.reshape(U, -1,1))
        nlp = {'f': obj, 'x':OPT, 'g':G, 'p':P}
        opts = {
            # IPOPT itself
            'ipopt.print_level':           0,      # no iteration‐by‐iteration printouts
            'ipopt.sb':                    'yes',  # suppress solver banner
            'ipopt.print_timing_statistics':'no',  # no timing stats
            # CasADi wrapper
            'print_time':                  False,  # don’t print overall timing
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # bounds (X free, U in [0,V_max]×[-δ_max,δ_max])
        nX = 3*(self.N+1)
        self.lbx = [-ca.inf]*nX + [0, -self.delta_max]*self.N
        self.ubx = [ ca.inf]*nX + [self.V_max, self.delta_max]*self.N
        self.lbg = [0]*G.size1()
        self.ubg = [0]*G.size1()

    def solve(self, x0, ref_traj, dt_seq):
        N = self.N
        assert len(dt_seq)==N

        # Pack P = [x0, ref_x/ref_y, dt_seq]
        p = np.concatenate([
            x0,
            ref_traj[:N].reshape(-1),
            np.array(dt_seq)
        ])

        # Initial guess
        x_init = np.tile(x0, (N+1,1))
        u_init = np.zeros((N,2))
        init   = np.concatenate([x_init.flatten(), u_init.flatten()])

        sol = self.solver(x0=init,
                          lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg,
                          p=p)
        U_opt = sol['x'][-2*N:].full().reshape(N,2)
        return U_opt[0]
    
class lidarTest:
    def __init__(self, lidar_name, vehicle_name):
        self.client = airsim.CarClient(ip="100.123.124.47")
        self.client.confirmConnection()
        self.vehicleName = vehicle_name
        self.lidarName = lidar_name
        self.lastlidarTimeStamp = 0

    def get_data(self, gpulidar):
        if gpulidar:
            lidarData = self.client.getGPULidarData(self.lidarName, self.vehicleName)
        else:
            lidarData = self.client.getLidarData(self.lidarName, self.vehicleName)
        if lidarData.time_stamp != self.lastlidarTimeStamp:
            self.lastlidarTimeStamp = lidarData.time_stamp
            if len(lidarData.point_cloud) < 2:
                return None, None
            points = np.array(lidarData.point_cloud, dtype=np.float32)
            num_dims = 5 if gpulidar else 3
            points = points.reshape((-1, num_dims))
            if not gpulidar:
                points = points * np.array([1, -1, 1])
            return points, lidarData.time_stamp
        return None, None

    def get_vehicle_pose(self):
        vehicle_pose = self.client.simGetVehiclePose()
        pos = vehicle_pose.position
        orient = vehicle_pose.orientation
        position_array = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)])
        rotation_matrix = self.quaternion_to_rotation_matrix(orient)
        return position_array, rotation_matrix

    def quaternion_to_rotation_matrix(self, q):
        qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
        return np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])

    def transform_to_world(self, points, position, rotation_matrix):
        points_rotated = np.dot(points, rotation_matrix.T)
        return points_rotated + position

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
            mean_z = z_sum / count
            estimates.append([gx * self.resolution, gy * self.resolution, mean_z])
        return np.array(estimates)

class AStarPlanner:

    def __init__(self,
                 grid: np.ndarray,
                 risk_factor: float = 0.8,
                 surround_weight: float = 1.0,
                 surround_sigma: float = 3.0):
        self.grid = grid.copy()
        max_risk = np.nanmax(self.grid) if not np.isnan(np.nanmax(self.grid)) else 1.0
        self.threshold = risk_factor * max_risk
        self.rows, self.cols = grid.shape

        # 1) build high‑risk mask
        high_mask = (self.grid >= self.threshold) | np.isnan(self.grid)

        # 2) distance from “safe” regions
        dist = distance_transform_edt(~high_mask)

        # 3) fade cost: big near high-mask, decays with sigma
        self.proximity_cost = surround_weight * np.exp(-dist / surround_sigma)

        # 4) final cost map: sum of raw risk + proximity penalty
        #    (nan→very large to keep blocked cells blocked)
        self.cost_map = np.where(np.isnan(self.grid),
                                 np.inf,
                                 self.grid + self.proximity_cost)

    def _heuristic(self, a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def _reconstruct_path(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        return path[::-1]

    def plan(self, start, goal):
        # check validity
        for pt in (start, goal):
            r,c = pt
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return None
            if self.cost_map[r,c] == np.inf:
                return None

        open_set = []
        g_score = {start: 0.0}
        heapq.heappush(open_set, (self._heuristic(start, goal), start))
        came_from = {}

        # 8‑connected
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                     (-1,-1),(-1,1),(1,-1),(1,1)]

        while open_set:
            f, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            cg = g_score[current]
            for dr, dc in neighbors:
                nr, nc = current[0]+dr, current[1]+dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                step_cost = self.cost_map[nr, nc] * np.hypot(dr, dc)
                if step_cost == np.inf:
                    continue
                tentative = cg + step_cost
                neighbor = (nr, nc)
                if tentative < g_score.get(neighbor, np.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (tentative + self._heuristic(neighbor, goal),
                                              neighbor))

        return None

def interpolate_in_radius(grid, radius):

    valid_mask = ~np.isnan(grid)
    if np.sum(valid_mask) == 0:
        return grid  # Nothing to interpolate from

    # Grid coordinates
    X, Y = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
    coords = np.stack([X[valid_mask], Y[valid_mask]], axis=1)
    values = grid[valid_mask]

    nan_mask = np.isnan(grid)
    nan_coords = np.stack([X[nan_mask], Y[nan_mask]], axis=1)

    # KDTree on valid points
    tree = cKDTree(coords)
    neighbors_list = tree.query_ball_point(nan_coords, radius)

    for idx, neighbors in enumerate(neighbors_list):
        if neighbors:
            weights = 1.0 / (np.linalg.norm(coords[neighbors] - nan_coords[idx], axis=1) + 1e-6)
            grid[nan_coords[idx][0], nan_coords[idx][1]] = np.sum(weights * values[neighbors]) / np.sum(weights)

    return grid
def filter_points_by_radius(points, center, radius):
    distances = np.linalg.norm(points[:, :2] - center, axis=1)
    return points[distances <= radius]
def fade_with_distance_transform(risk_grid, high_threshold=0.4, fade_scale=4.0, sigma=5.0):
    grid_max = np.nanmax(risk_grid)
    threshold_val = high_threshold * grid_max
    high_mask = risk_grid > threshold_val
    dist_map = distance_transform_edt(~high_mask)
    fade_risk = fade_scale * np.exp(-dist_map / sigma)
    return np.maximum(risk_grid, fade_risk)
def smooth_path(path, window_size=5):
    path = np.array(path)
    n_points = len(path)
    if n_points < window_size:
        return path
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    sm = [np.mean(path[max(0, i-half):min(n_points, i+half+1)], axis=0)
          for i in range(n_points)]
    return np.array(sm)
def get_map_setting(sp, dp, margin, grid_resolution):
    # sp = start_point (x,y), dp = destination_point (x,y)
    min_x = min(sp[0], dp[0]) - margin
    max_x = max(sp[0], dp[0]) + margin
    min_y = min(sp[1], dp[1]) - margin
    max_y = max(sp[1], dp[1]) + margin

    # compute the grid edges
    x_edges = np.arange(min_x, max_x + grid_resolution, grid_resolution)
    y_edges = np.arange(min_y, max_y + grid_resolution, grid_resolution)
    x_mid = (x_edges[:-1] + x_edges[1:]) / 2
    y_mid = (y_edges[:-1] + y_edges[1:]) / 2

    return x_edges, y_edges, x_mid, y_mid

STEP_config ={
    # MAP
    'grid_margin': 4,
    'grid_resolution': 0.1,
    'radius_filter': 12,

    # RISK
    'max_height_diff': 0.032, 
    'max_slope_degrees': 20.0,
    'risk_radius': 0.5,

    'step_weight': 2.0,
    'slope_weight': 2.0,

    'interpolate_radius': 1.5,
    'cvar_a': 0.7,
    'cvar_radius': 4.0,
    'distance_ignored': 9.0,

    # a star
    'distance_to_temp': 5.0,
    'distance_to_goal': 0.75,
    # NMPC
    'N-npmc': 20,
    'Vmax-nmpc': 0.05,
    'delta-nmpc': 25,

    # others for replan:
    'HIGH_RISK': 0.6,

}
if __name__ == "__main__":
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')

    # Initialize ground segmentation.
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    config_path = os.path.join(BASE_DIR, "../assets/config.toml")
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default parameters")
        groundseg = ground_seg()
    else:
        groundseg = ground_seg(config_path)

    # Map setup:
    pos, _ = lidar_test.get_vehicle_pose()
    start_point = pos[:2]
    destination_point = np.array([17, -7])
    x_edges, y_edges, x_mid, y_mid = get_map_setting(start_point, destination_point, margin=STEP_config['grid_margin'], grid_resolution=STEP_config['grid_resolution'])
    X, Y = np.meshgrid(x_mid, y_mid)
    grid_map_ground = GridMap(resolution=STEP_config['grid_resolution'])
    grid_map_obstacle = GridMap(resolution=STEP_config['grid_resolution'])

    # Setup NMPC and plot
    nmpc = NMPCController(horizon=STEP_config['N-npmc'],
                          wheelbase=0.25,
                          V_max=STEP_config['Vmax-nmpc'],
                          delta_max=np.deg2rad(STEP_config['delta-nmpc']))
    ctr = airsim.CarControls()
    cmap = LinearSegmentedColormap.from_list("gray_yellow_red",
               [(0.5,0.5,0.5),(1,1,0),(1,0,0)], N=10)
    fig, ax = plt.subplots(); plt.ion(); 
    
    # Define variables
    colorbar = None
    prev_grid = None
    prev_path = None
    temp_dest = None

    # a circular structuring element if you want Euclidean radius.
    struct = generate_binary_structure(2,1)
    mask_elem = binary_dilation(np.zeros((5,5), bool), 
                                structure=struct, 
                                iterations=2)
    prev_t = time.time()
    stats_dict ={
        'count': 0,
        'collision_count':0,
        'total_length':[],
        'dist_to_goal':[]

    }
    last_pos = start_point.copy()
    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is None:
                continue
            
            # Process point cloud.
            points = np.array(point_cloud_data[:, :3], dtype=np.float64)
            points = points[np.linalg.norm(points, axis=1) > 0.6]
            pos, R = lidar_test.get_vehicle_pose()
            vehicle_x, vehicle_y = pos[0], pos[1]
            veh_xy = np.array([vehicle_x, vehicle_y]) 

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
            ground_points = filter_points_by_radius(ground_points, veh_xy, STEP_config['radius_filter'])
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
                obstacle_points = filter_points_by_radius(obstacle_points, veh_xy, STEP_config['radius_filter'])
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
            risk_grid = risk_grid.filled(0.50)

            # Mask cells far from the vehicle.
            distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)
            risk_grid[distance_from_vehicle.T > STEP_config['distance_ignored']] = np.nan

            
            # Check for replan condition - if we close to the temp dest
            trigger_temp_dest = False
            if (temp_dest is None or 
                np.hypot(vehicle_x - temp_dest[0], vehicle_y - temp_dest[1]) < STEP_config['distance_to_temp']):

                valid = np.argwhere(~np.isnan(risk_grid))
                if valid.size > 0:
                    # find the valid cell nearest the true goal
                    centers = np.column_stack((x_mid[valid[:,0]], y_mid[valid[:,1]]))
                    dists_to_goal = np.linalg.norm(centers - destination_point, axis=1)
                    best = valid[np.argmin(dists_to_goal)]
                    temp_dest = (x_edges[best[0]], y_edges[best[1]])
                    trigger_temp_dest = True

            # Plan A* Star
            planner = AStarPlanner(risk_grid)
            start_idx = (np.digitize(vehicle_x, x_edges) - 1,
                         np.digitize(vehicle_y, y_edges) - 1)
            goal_idx = (np.digitize(temp_dest[0], x_edges) - 1,
                        np.digitize(temp_dest[1], y_edges) - 1)
            if prev_path is not None and not trigger_temp_dest:
                # Compute a high‐risk mask (True where risk ≥ 0.6) & Dilate it by 2 cells
                max_risk_value = np.nanmax(risk_grid)
                hr = (risk_grid >= STEP_config['HIGH_RISK']* max_risk_value)
                hr_dilated = binary_dilation(hr, structure=struct, iterations=3)

                # Check if any of our old path indices hit the dilated high‐risk area
                needs_replan = False
                for (r,c) in prev_path:
                    # make sure (r,c) in bounds
                    if 0 <= r < hr_dilated.shape[0] and 0 <= c < hr_dilated.shape[1]:
                        if hr_dilated[r, c]:
                            needs_replan = True
                            break
                if not needs_replan:
                    path_idx = prev_path
                else:
                    path_idx = planner.plan(start_idx, goal_idx)
            else:
                path_idx = planner.plan(start_idx, goal_idx)

            # stash for next iteration
            if path_idx is not None:
                prev_path = path_idx.copy()
                prev_grid = risk_grid.copy()

            # Visualization
            ax.clear()
            c = ax.pcolormesh(Y, X, risk_grid.T, shading='auto', cmap=cmap, alpha=0.7)
            if colorbar is None:
                colorbar = fig.colorbar(c, ax=ax, label='Risk')
            else:
                colorbar.update_normal(c)
            ax.scatter(vehicle_y, vehicle_x, c='green', s=50, label='Vehicle')
            ax.scatter(destination_point[1], destination_point[0], c='red', s=50, label='Goal')
            ax.scatter(temp_dest[1], temp_dest[0], c='red', s=50, label='Goal')
            
            raw_coords = np.array([[x_mid[r], y_mid[c]] for r, c in path_idx])
            smoothed_path = smooth_path(raw_coords, window_size=5)
            ax.plot(smoothed_path[:,1],
                    smoothed_path[:,0],
                    color='blue',
                    linewidth=2,
                    label='Smoothed A* Path')
            
            # --- after you have `smoothed_path` and time dt  ---
            dt_loop = max(min(time.time() - prev_t, 0.2), 0.05)
            prev_t  = time.time()

            # 1) extract NMPC reference: next N waypoints
            d2sp    = np.linalg.norm(smoothed_path - pos[:2], axis=1)
            i0      = np.argmin(d2sp)
            ref_pts = smoothed_path[i0+1 : i0+1+nmpc.N]
            if len(ref_pts) < nmpc.N:
                ref_pts = np.vstack((
                    ref_pts,
                    np.tile(ref_pts[-1], (nmpc.N - len(ref_pts), 1))
                ))

            # 2) solve NMPC
            psi0 = math.atan2(R[1,0], R[0,0])
            x0   = np.array([vehicle_x, vehicle_y, psi0])
            v_cmd, δ_cmd = nmpc.solve(x0, ref_pts, [dt_loop]*nmpc.N)

            # 3) “big‐turn” fallback
            dx, dy       = ref_pts[0] - pos[:2]
            des_ψ        = math.atan2(dy, dx)
            err_ψ        = math.atan2(math.sin(des_ψ-psi0),
                                    math.cos(des_ψ-psi0))
            if abs(err_ψ) > np.deg2rad(20):
                ctr.steering = np.clip(err_ψ/np.deg2rad(20), -1, 1)
                ctr.throttle = 0.0
            else:
                ctr.steering = float(np.clip(δ_cmd / nmpc.delta_max, -1, 1))
                scale        = 1 - 0.8*abs(err_ψ)/np.deg2rad(25)
                ctr.throttle = float(np.clip(v_cmd*scale / nmpc.V_max, 0, nmpc.V_max))
            lidar_test.client.setCarControls(ctr)

            ax.plot(ref_pts[:,1], ref_pts[:,0], 'r--', linewidth=1, label='Reference Trajectory')
            ax.legend()
            plt.draw(); plt.pause(0.1)
        
            # Record REST INFO
            stats_dict['count'] += 1
            if lidar_test.client.simGetCollisionInfo().has_collided:
                stats_dict['collision_count'] += 1

            distance_last = np.linalg.norm(destination_point - np.array([vehicle_x, vehicle_y]))
            stats_dict['dist_to_goal'].append(distance_last)
            if distance_last < STEP_config['distance_to_goal']:
                # lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0), lidar_test.vehicleName)
                print("-----------------------------------------------")
                print("Reached Destination")
                print("Count: ", stats_dict['count'])
                print("collision_count: ", stats_dict['collision_count'])
                print("dist_to_goal: ", np.mean(stats_dict['dist_to_goal']))
                print("total_length: ", np.sum(stats_dict['total_length']))
                lidar_test.client.enableApiControl(False, lidar_test.vehicleName)
                break


    finally:
        plt.ioff()
        # plt.show()
        # plt.close()
