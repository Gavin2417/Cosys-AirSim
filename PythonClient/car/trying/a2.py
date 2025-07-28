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
# from linefit import ground_seg
from function5 import calculate_combined_risks, compute_cvar_cellwise
from scipy.ndimage import generic_filter
base = os.path.dirname(__file__)        
project_root = base
print(project_root)
rand_dir = os.path.join(project_root, "rand")
os.chdir(rand_dir)
from predict1 import RandlaGroundSegmentor
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
            mean_label = np.ceil(z_sum/count)
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

if __name__ == '__main__':
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')
    seg = RandlaGroundSegmentor(device=None, subsample_grid=0.1)

   # Map setup:
    grid_resolution = 0.1
    margin = 4
    pos, _ = lidar_test.get_vehicle_pose()
    start_point = pos[:2]
    destination_point = np.array([17, -7])
    min_x = min(start_point[0], destination_point[0]) - margin
    max_x = max(start_point[0], destination_point[0]) + margin
    min_y = min(start_point[1], destination_point[1]) - margin
    max_y = max(start_point[1], destination_point[1]) + margin
    x_edges = np.arange(min_x, max_x + grid_resolution, grid_resolution)
    y_edges = np.arange(min_y, max_y + grid_resolution, grid_resolution)
    x_mid = (x_edges[:-1] + x_edges[1:]) / 2
    y_mid = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_mid, y_mid)
    
    # Setup NMPC and plot
    N = 20
    nmpc = NMPCController(horizon=N,
                          wheelbase=0.25,
                          V_max=0.05,
                          delta_max=np.deg2rad(25))
    ctr = airsim.CarControls()
    grid_map_ground = GridMap(resolution=0.1)
    cmap = LinearSegmentedColormap.from_list("gray_yellow_red",
               [(0.5,0.5,0.5),(1,1,0),(1,0,0)], N=10)
    colorbar = None
    fig, ax = plt.subplots(); plt.ion(); 
    
    # Define variables
    prev_grid = None
    prev_path = None
    temp_dest = None

    replan_thresh = 0.5    # only replan if risk on old path changed by >5%
    HIGH_RISK = 0.65
    DILATION_RADIUS = 2

    # build a 5×5 connectivity for a radius≈2 square; you can also use 
    # a circular structuring element if you want Euclidean radius.
    struct = generate_binary_structure(2,1)
    mask_elem = binary_dilation(np.zeros((5,5), bool), 
                                structure=struct, 
                                iterations=2)
    prev_t = time.time()
    try:
        while True:
            # Get points and vehicle pose
            pc, ts = lidar_test.get_data(gpulidar=True)
            if pc is None:
                continue
            points = np.array(pc[:,:3])
            points = points[np.linalg.norm(points,axis=1) > 0.6]
            pos, R = lidar_test.get_vehicle_pose()
            vehicle_x, vehicle_y = pos[0], pos[1]  

            # Get labels for the points cloud
            world = lidar_test.transform_to_world(points, pos, R)
            world[:,2] = -world[:,2]
            labels = seg.segment(world)
            for p,lab in zip(world, labels):
                grid_map_ground.add_point(p[0], p[1], p[2], lab)
            ground_pts = grid_map_ground.get_label_estimate()

            # Build the risk grid
            gx, gy, gz = ground_pts[:,0], ground_pts[:,1], ground_pts[:,2]
            risk_grid, _, _, _ = binned_statistic_2d(
                gx, gy, gz, statistic='mean', bins=[x_edges, y_edges]
            )
            risk_grid = interpolate_in_radius(risk_grid, 1.5)
            risk_grid = compute_cvar_cellwise(risk_grid, alpha=0.7, radius=4.0)
            risk_grid = np.nan_to_num(risk_grid, nan=1.0)
            dist = np.hypot(X - vehicle_x, Y - vehicle_y)
            risk_grid[dist.T > 9.0] = np.nan


            # Check for replan condition - if we close to the temp dest
            trigger_temp_dest = False
            if (temp_dest is None
                or np.hypot(vehicle_x - temp_dest[0],
                        vehicle_y - temp_dest[1]) < 5.0):

                valid = np.argwhere(~np.isnan(risk_grid))
                if valid.size > 0:
                    # find the valid cell nearest the true goal
                    centers = np.column_stack((x_mid[valid[:,0]],
                                            y_mid[valid[:,1]]))
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
                hr = (risk_grid >= HIGH_RISK* max_risk_value)
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
                ctr.steering = np.clip(err_ψ/np.deg2rad(25), -1, 1)
                ctr.throttle = 0.0
            else:
                ctr.steering = float(np.clip(δ_cmd / nmpc.delta_max, -1, 1))
                scale        = 1 - 0.8*abs(err_ψ)/np.deg2rad(20)
                ctr.throttle = float(np.clip(v_cmd*scale / nmpc.V_max, 0, 0.04))
            lidar_test.client.setCarControls(ctr)

            ax.plot(ref_pts[:,1], ref_pts[:,0], 'r--', linewidth=1, label='Reference Trajectory')
            ax.legend()
            plt.draw(); plt.pause(0.1)
            if np.hypot(vehicle_x - destination_point[0], vehicle_y - destination_point[1]) < 0.75:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0))
                break
    finally:
        plt.ioff(); plt.show()
