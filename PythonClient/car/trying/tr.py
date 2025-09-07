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
# from linefit import ground_seg
from function5 import calculate_combined_risks, compute_cvar_cellwise
base = os.path.dirname(__file__)        
project_root = base
print(project_root)
rand_dir = os.path.join(project_root, "rand")
os.chdir(rand_dir)
from predict1 import RandlaGroundSegmentor
import casadi as ca
from skimage.graph import route_through_array


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

        # symbols
        x, y, psi = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('psi')
        states  = ca.vertcat(x, y, psi)
        v, dlt  = ca.SX.sym('v'), ca.SX.sym('dlt')
        controls = ca.vertcat(v, dlt)

        # dynamics
        rhs = ca.vertcat(v*ca.cos(psi),
                         v*ca.sin(psi),
                         v/self.L * ca.tan(dlt))
        f   = ca.Function('f', [states, controls], [rhs])

        # decision variables
        X = ca.SX.sym('X', 3, self.N+1)
        U = ca.SX.sym('U', 2, self.N)

        # parameters: [ x0(3), ref_x/ref_y (2*N), dt_seq (N) ]
        P = ca.SX.sym('P', 3 + 2*self.N + self.N)

        g   = []
        obj = 0

        # initial-state
        g.append(X[:,0] - P[0:3])

        for k in range(self.N):
            xr = P[3 + 2*k]
            yr = P[3 + 2*k + 1]
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

        # terminal cost
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

        p = np.concatenate([x0, ref_traj[:N].reshape(-1), np.array(dt_seq)])
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

    def add_point(self, x, y, z, label):
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = [z, 1]
        else:
            self.grid[cell][0] += label
            self.grid[cell][1] += 1

    def get_label_estimate(self):
        estimates = []
        for (gx, gy), (z_sum, count) in self.grid.items():
            # get rhe mean label
            mean_label = float(z_sum/count)
            estimates.append([gx * self.resolution, gy * self.resolution, mean_label])
        return np.array(estimates)\

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

    def plan(self, start, goal, MAX_RTSK_VALUE=50, max_expansions=20000):
        # check validity
        for pt in (start, goal):
            r, c = pt
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return None
            if self.cost_map[r, c] == np.inf:
                return None

        # Compute risk threshold
        # max_risk = np.max(self.cost_map[np.isfinite(self.cost_map
        risk_threshold = 0.8 * MAX_RTSK_VALUE

        open_set = []
        g_score = {start: 0.0}
        heapq.heappush(open_set, (self._heuristic(start, goal), start))
        came_from = {}

        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        expansions = 0
        best_so_far = None
        best_f = float('inf')

        while open_set:
            f, current = heapq.heappop(open_set)

            # track best seen to allow graceful timeout return
            if f < best_f:
                best_f = f
                best_so_far = current

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if expansions >= max_expansions:
                # give a partial path toward the best node so far
                if best_so_far is not None:
                    return self._reconstruct_path(came_from, best_so_far)
                return None

            expansions += 1
            cg = g_score[current]
            for dr, dc in neighbors:
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                cell_cost = self.cost_map[nr, nc]
                if cell_cost == np.inf or cell_cost >= risk_threshold:
                    continue

                # small heuristic: skip tiny improvements to curb thrash
                step_cost = cell_cost * np.hypot(dr, dc)
                tentative = cg + step_cost
                neighbor = (nr, nc)
                if tentative + self._heuristic(neighbor, goal) >= best_f:
                    continue

                if tentative < g_score.get(neighbor, np.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (tentative + self._heuristic(neighbor, goal), neighbor))
        return None

def smooth_path(path, window_size=5):
    """
    Smooths a sequence of (x,y) points using a simple moving average filter.
    """
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
def interpolate_in_radius(grid, radius):
    """
    Vectorized interpolation using cKDTree: fills NaNs in a grid based on nearby valid cells.
    """
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
    'interpolate_radius': 1.5,
    'cvar_a': 0.7,
    'cvar_radius': 4.0,
    'distance_ignored': 9.0,

    # a star
    'distance_to_temp': 5.0,
    'distance_to_goal': 0.75,
    # NMPC
    'N-npmc': 20,
    'Vmax-nmpc': 0.8,
    'delta-nmpc': 25,

    # others for replan:
    'HIGH_RISK': 0.6,
    'MAX_RTSK_VALUE': 50,
    'visualize': False
}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xgoal', type=float, default=-21.5, help='X coordinate of the goal point')
    parser.add_argument('--ygoal', type=float, default=-6, help='Y coordinate of the goal point')
    args = parser.parse_args()
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    pos, _ = lidar_test.get_vehicle_pose()
    # lidar_test.client.enableApiControl(True, 'CPHusky')
    seg = RandlaGroundSegmentor(device=None, subsample_grid=0.1)
    SPAWN_POSE = airsim.Pose(
        airsim.Vector3r(-30, -20, pos[2]-0.5),
        airsim.Quaternionr(0, 0, 0, 1)
    )
    lidar_test.client.simSetVehiclePose(SPAWN_POSE, ignore_collision=True)
    time.sleep(0.1)
    pos, _ = lidar_test.get_vehicle_pose()
   # Map setup:
    
    start_point = pos[:2]
    destination_point = np.array([args.xgoal, args.ygoal])
    x_edges, y_edges, x_mid, y_mid = get_map_setting(start_point, destination_point, margin=STEP_config['grid_margin'], grid_resolution=STEP_config['grid_resolution'])
    X, Y = np.meshgrid(x_mid, y_mid)
    grid_map_ground = GridMap(resolution=STEP_config['grid_resolution'])

    # Setup NMPC and plot
    nmpc = NMPCController(horizon=STEP_config['N-npmc'],
                        wheelbase=0.25,
                        V_max=STEP_config['Vmax-nmpc'],
                        delta_max=np.deg2rad(STEP_config['delta-nmpc']))
    ctr = airsim.CarControls()

    if STEP_config['visualize']:
        cmap = LinearSegmentedColormap.from_list("gray_yellow_red",
                [(0.5,0.5,0.5),(1,1,0),(1,0,0)], N=50)
        fig, ax = plt.subplots(); plt.ion(); 
        colorbar = None
    prev_grid = None
    prev_path = None
    temp_goal_idx = None         
    temp_dest_xy  = None  
    i0_prev = 0
    # build a 5×5 connectivity for a radius≈2 square;
    struct = generate_binary_structure(2,1)
    prev_t = time.time()
    stats_dict ={
        'count': 0,
        'collision_count':0,
        'total_length':[],
        'dist_to_goal': None,
        'reach_goal': False
    }
    MAX_ITER = 350
    distance_last = np.linalg.norm(destination_point - np.array([pos[0], pos[1]]))
    stats_dict['dist_to_goal'] = distance_last
    last_pos = start_point.copy()
    try:
        while True:
            pc, ts = lidar_test.get_data(gpulidar=True)
            if pc is None:
                continue

            # Process point cloud.
            points = np.array(pc[:,:3])
            points = points[np.linalg.norm(points, axis=1) > 0.6]
            pos, R = lidar_test.get_vehicle_pose()
            vehicle_x, vehicle_y = pos[0], pos[1]  
            veh_xy = np.array([vehicle_x, vehicle_y])

            # Record stats
            distance_travelled = np.linalg.norm(last_pos - np.array([vehicle_x, vehicle_y]))
            stats_dict['total_length'].append(distance_travelled)
            last_pos = veh_xy.copy() 

            # Get labels for the points cloud
            world = lidar_test.transform_to_world(points, pos, R)
            world[:,2] = -world[:,2]

            # Populate grid maps based on segmentation.
            labels = seg.segment(world)
        
            for p, lab in zip(world, labels):
                grid_map_ground.add_point(p[0], p[1], p[2], lab)
            ground_pts = grid_map_ground.get_label_estimate()
            ground_pts = filter_points_by_radius(ground_pts, veh_xy, STEP_config['radius_filter'])
            if ground_pts.size == 0: continue
            risk_grid, _, _, _ = binned_statistic_2d(
                ground_pts[:,0], ground_pts[:,1], ground_pts[:,2], statistic='mean', bins=[x_edges, y_edges]
            )

            # Calculate risk grid            
            risk_grid = interpolate_in_radius(risk_grid, STEP_config['interpolate_radius'])
            risk_grid = compute_cvar_cellwise(risk_grid, alpha=STEP_config['cvar_a'], radius=STEP_config['cvar_radius'])
            risk_grid = np.nan_to_num(risk_grid, nan=1.0)
            dist = np.hypot(X - vehicle_x, Y - vehicle_y)
            risk_grid[dist.T > STEP_config['distance_ignored']] = np.nan

            trigger_temp_dest = False
            valid = np.argwhere(~np.isnan(risk_grid))
            if temp_goal_idx is None or (temp_dest_xy is not None and
                                         np.hypot(vehicle_x - temp_dest_xy[0],
                                                  vehicle_y - temp_dest_xy[1]) < STEP_config['distance_to_temp']) or stats_dict['count'] %15 ==0:
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
            # --- stable nearest index with hysteresis (fix #2) ---
            dists = np.linalg.norm(smoothed_path - pos[:2], axis=1)
            SEARCH_BACK, SEARCH_AHEAD = 2, 25
            s0 = max(i0_prev - SEARCH_BACK, 0)
            s1 = min(i0_prev + SEARCH_AHEAD, len(smoothed_path) - 1)
            i0 = s0 + int(np.argmin(dists[s0:s1+1]))
            i0 = max(i0, i0_prev - SEARCH_BACK)   # prevent big backward jumps
            i0_prev = i0

            # extract NMPC reference: next N waypoints
            ref_pts = smoothed_path[i0+1 : i0+1+nmpc.N]
            if len(ref_pts) < nmpc.N and len(ref_pts) > 0:
                ref_pts = np.vstack((ref_pts, np.tile(ref_pts[-1], (nmpc.N - len(ref_pts), 1))))
            elif len(ref_pts) == 0:
                ref_pts = np.tile(smoothed_path[-1], (nmpc.N, 1))

            # # 2) solve NMPC
            # psi0 = math.atan2(R[1,0], R[0,0])
            # x0   = np.array([vehicle_x, vehicle_y, psi0])
            # v_cmd, δ_cmd = nmpc.solve(x0, ref_pts, [dt_loop]*nmpc.N)

            # # 3) desired heading from path tangent (fix #4)
            # LOOKAHEAD_STEPS = 4
            # j = min(i0 + LOOKAHEAD_STEPS, len(smoothed_path) - 1)
            # dx = smoothed_path[j,0] - smoothed_path[i0,0]
            # dy = smoothed_path[j,1] - smoothed_path[i0,1]
            # des_ψ = math.atan2(dy, dx)
            # err_ψ = math.atan2(math.sin(des_ψ - psi0), math.cos(des_ψ - psi0))

            # # your existing big-turn branch (unchanged)
            # if abs(err_ψ) > np.deg2rad(20):
            #     ctr.steering = np.clip(err_ψ/np.deg2rad(20), -1, 1)
            #     ctr.throttle = 0.0
            # else:
            #     ctr.steering = float(np.clip(δ_cmd / nmpc.delta_max, -1, 1))
            #     scale        = 1 - 0.8*abs(err_ψ)/np.deg2rad(20)
            #     ctr.throttle = float(np.clip(v_cmd*scale / nmpc.V_max, 0, nmpc.V_max))

            # lidar_test.client.setCarControls(ctr)

            # Visualization
            if STEP_config['visualize']:
                ax.clear()
                c = ax.pcolormesh(Y, X, risk_grid.T, shading='auto', cmap=cmap, alpha=0.7)
                if colorbar is None:
                    colorbar = fig.colorbar(c, ax=ax, label='Risk')
                else:
                    colorbar.update_normal(c)
                ax.scatter(vehicle_y, vehicle_x, c='green', s=50, label='Vehicle')
                ax.scatter(destination_point[1], destination_point[0], c='red', s=50, label='Goal')
                if temp_dest_xy is not None:
                    ax.scatter(temp_dest_xy[1], temp_dest_xy[0], c='black', s=30, marker='s',
                               linewidth=0.15, label='Temp Goal')
                ax.plot(smoothed_path[:,1], smoothed_path[:,0], color='blue', linewidth=2, label='Smoothed A* Path')
                ax.plot(ref_pts[:,1], ref_pts[:,0], 'r--', linewidth=1, label='Reference Trajectory')
                # ax.legend()
                plt.draw(); plt.pause(0.1)
            
            # Record REST INFO
            stats_dict['count'] += 1
            if lidar_test.client.simGetCollisionInfo().has_collided:
                stats_dict['collision_count'] += 1

            distance_last = np.linalg.norm(destination_point - np.array([vehicle_x, vehicle_y]))
            # stats_dict['dist_to_goal'].append(distance_last)
            if distance_last < STEP_config['distance_to_goal']:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0), lidar_test.vehicleName)
                stats_dict['reach_goal'] = True
                break
            elif stats_dict['count'] >= MAX_ITER:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0), lidar_test.vehicleName)
                stats_dict['reach_goal'] = False
                break
    finally:
        print("-----------------------------------------------")
        print("Reached Destination")
        print("Count: ", stats_dict['count'])
        print("collision_count: ", stats_dict['collision_count'])
        print("dist_to_goal: ", stats_dict['dist_to_goal'])
        print("total_length: ", np.sum(stats_dict['total_length']))
        # lidar_test.client.enableApiControl(False, lidar_test.vehicleName)
        print("--------------Done--------------")

    
    # path to your “master” stats file
    os.chdir(base)
    stats_file = os.path.join(project_root, "record/randla_stats.json")
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
        "dist_to_goal": stats_dict["dist_to_goal"]
    })

    # write it back out
    # with open(stats_file, "w") as f:
    #     json.dump(all_runs, f, indent=2)

    print(f"Saved stats for this run")