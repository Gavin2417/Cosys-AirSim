
import os
import time
import heapq
import cosysairsim as airsim
import numpy as np
import numpy.ma as ma
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LinearSegmentedColormap

from linefit import ground_seg
from function2 import calculate_combined_risks, compute_cvar_cellwise
import casadi as ca
class NMPCController:
    def __init__(self, dt=0.1, horizon=10, L=0.8):
        self.dt = dt
        self.N = horizon
        self.L = L
        self.steer_limit = 1.0
        self.throttle_limit = 1.0
        self.w_pos = 10
        self.w_yaw = 1
        self.w_vel = 1
        self.w_u = 0.1
        self.w_du = 1.0
        self._build_solver()

    def _build_solver(self):
        N, dt, L = self.N, self.dt, self.L
        x = ca.SX.sym('x'); y = ca.SX.sym('y'); theta = ca.SX.sym('theta'); v = ca.SX.sym('v')
        delta = ca.SX.sym('delta'); a = ca.SX.sym('a')
        state = ca.vertcat(x, y, theta, v)
        control = ca.vertcat(delta, a)
        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v / L * ca.tan(delta), a)
        f = ca.Function("f", [state, control], [rhs])

        X = ca.SX.sym("X", 4, N+1)
        U = ca.SX.sym("U", 2, N)
        P = ca.SX.sym("P", 4 + 3*N)  # state + [x,y,yaw]*N
        obj = 0; g = []
        for k in range(N):
            st, con = X[:,k], U[:,k]
            x_ref, y_ref, yaw_ref = P[4+3*k:4+3*k+3]
            obj += self.w_pos * ((st[0] - x_ref)**2 + (st[1] - y_ref)**2)
            obj += self.w_yaw * (st[2] - yaw_ref)**2
            obj += self.w_u * ca.sumsqr(con)
            if k > 0:
                obj += self.w_du * ca.sumsqr(U[:,k] - U[:,k-1])
            st_next = X[:,k] + dt * f(X[:,k], U[:,k])
            g.append(X[:,k+1] - st_next)

        opt_vars = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)))
        nlp = {'f': obj, 'x': opt_vars, 'p': P, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.nx = 4; self.nu = 2; self.X = X; self.U = U; self.P = P

    def solve(self, state, ref_traj):
        x0 = np.array(state).flatten()
        if len(ref_traj) < self.N:
            ref_traj = np.vstack([ref_traj, np.tile(ref_traj[-1], (self.N - len(ref_traj), 1))])
        p = np.concatenate([x0] + [r for r in ref_traj[:self.N]])
        x_init = np.tile(x0.reshape(-1,1), (1,self.N+1))
        u_init = np.zeros((2,self.N))

        lbx, ubx = [], []
        for _ in range(self.N+1):
            lbx += [-ca.inf]*4
            ubx += [ ca.inf]*4
        for _ in range(self.N):
            lbx += [-1.0, -1.0]
            ubx += [ 1.0,  1.0]

        sol = self.solver(
            x0=ca.vertcat(x_init.flatten(), u_init.flatten()),
            p=p, lbg=0, ubg=0, lbx=lbx, ubx=ubx
        )
        u = sol['x'][4*(self.N+1):4*(self.N+1)+2]
        return float(u[0]), float(u[1])

def interpolate_in_radius(grid, radius):
    valid = ~np.isnan(grid)
    coords = np.column_stack(np.where(valid))
    values = grid[valid]
    tree = cKDTree(coords)
    nan_coords = np.column_stack(np.where(np.isnan(grid)))
    for coord in nan_coords:
        nbrs = tree.query_ball_point(coord, radius)
        if not nbrs:
            continue
        w_vals = []
        wts    = []
        for i in nbrs:
            d = np.linalg.norm(coord - coords[i]) + 1e-6
            w = 1.0/d
            wts.append(w)
            w_vals.append(w * values[i])
        grid[coord[0], coord[1]] = sum(w_vals)/sum(wts)
    return grid


def filter_points_by_radius(points, center, radius):
    d = np.linalg.norm(points[:, :2] - center, axis=1)
    return points[d <= radius]

# A* helpers
def is_valid(r,c,g):  return 0<=r<g.shape[0] and 0<=c<g.shape[1]
def is_unblocked(g,r,c): return not np.isnan(g[r,c]) and g[r,c]<1.0
def h(r,c,d):      return np.hypot(r-d[0], c-d[1])

def trace_path(parents, dest):
    path = []
    r,c = dest
    while True:
        path.append((r,c))
        pr,pc = parents[r,c]
        if (r,c)==(pr,pc): break
        r,c = pr,pc
    return path[::-1]

def a_star_search(risk, start, dest):
    R,C = risk.shape
    open_list = [(0.0, start)]
    g = np.full((R,C), np.inf)
    g[start] = 0
    f = np.full((R,C), np.inf)
    f[start] = h(*start, dest)
    parents = np.zeros((R,C,2), dtype=int)
    for i in range(R):
        for j in range(C): parents[i,j] = (i,j)

    while open_list:
        _, (r,c) = heapq.heappop(open_list)
        if (r,c)==dest:
            return trace_path(parents, dest)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            if is_valid(nr,nc,risk) and is_unblocked(risk,nr,nc):
                tg = g[r,c] + risk[nr,nc]
                if tg < g[nr,nc]:
                    g[nr,nc] = tg
                    f[nr,nc] = tg + h(nr,nc,dest)
                    parents[nr,nc] = (r,c)
                    heapq.heappush(open_list, (f[nr,nc], (nr,nc)))
    return None


def smooth_path(path, window_size=5):
    p = np.array(path)
    n = len(p)
    if n<window_size: return p
    if window_size%2==0: window_size+=1
    hw = window_size//2
    sm = []
    for i in range(n):
        start = max(0,i-hw)
        end   = min(n, i+hw+1)
        sm.append(p[start:end].mean(axis=0))
    return np.array(sm)

class lidarTest:
    def __init__(self, lidar_name, vehicle_name):
        self.client = airsim.CarClient(ip="100.123.124.47")
        self.client.confirmConnection()
        self.vehicleName = vehicle_name
        self.lidarName = lidar_name
        self.lastlidarTimeStamp = 0

    def get_data(self, gpulidar=True):
        if gpulidar:
            d = self.client.getGPULidarData(self.lidarName, self.vehicleName)
        else:
            d = self.client.getLidarData(self.lidarName, self.vehicleName)
        if d.time_stamp==self.lastlidarTimeStamp:
            return None, None
        self.lastlidarTimeStamp = d.time_stamp
        if len(d.point_cloud)<2:
            return None, None
        pts = np.array(d.point_cloud, dtype=np.float32)
        dims = 5 if gpulidar else 3
        pts = pts.reshape(-1, dims)
        if not gpulidar:
            pts *= np.array([1, -1, 1])
        return pts, d.time_stamp

    def get_vehicle_pose(self):
        p = self.client.simGetVehiclePose()
        pos = np.array([p.position.x_val,
                        p.position.y_val,
                        p.position.z_val])
        q  = p.orientation
        R  = self.quaternion_to_rotation_matrix(q)
        return pos, R

    def quaternion_to_rotation_matrix(self, q):
        qw,qx,qy,qz = q.w_val, q.x_val, q.y_val, q.z_val
        return np.array([
            [1-2*qy*qy-2*qz*qz,   2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
            [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz,     2*qy*qz-2*qx*qw],
            [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx*qx-2*qy*qy],
        ])
    def transform_to_world(self, points, position, rotation_matrix):
        points_rotated = np.dot(points, rotation_matrix.T)
        return points_rotated + position

class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        self.grid = {}
    def get_cell(self, x,y):
        return round(int(x/self.resolution),1), round(int(y/self.resolution),1)
    def add_point(self, x,y,z,_):
        c = self.get_cell(x,y)
        self.grid.setdefault(c, []).append(z)
    def get_estimate(self):
        out = []
        for (gx,gy), zs in self.grid.items():
            out.append([gx*self.resolution, gy*self.resolution, np.mean(zs)])
        return np.array(out)

def main():
    # mpc = NMPCController()
    print("here")
    # AirSim & Lidar
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')

    # Grid maps & segmentation
    ground_map   = GridMap(0.1)
    obstacle_map = GridMap(0.1)
    base = os.path.dirname(__file__)
    cfg  = os.path.join(base, '../assets/config.toml')
    if os.path.exists(cfg): groundseg = ground_seg(cfg)
    else:
        print(f"No config at {cfg}, using defaults.")
        groundseg = ground_seg()

    # Plot setup
    fig, ax = plt.subplots()
    plt.ion()
    cbar = None

    # Manual world path
    manual_world = [(-1,0), (0,0), (1,0), (2,0), (3,0), (4,0), (5,0),
                    (6,0), (7,0), (8,0), (8,-1),(8,-2),(8,-3),(8,-4),(8,-5),(9,-5),(10,-5),(10,-6)]

    try:
        while True:
            pts, ts = lidar_test.get_data(gpulidar=True)
            if pts is None: continue

            # World coords
            pos, R = lidar_test.get_vehicle_pose()
            pts_w = lidar_test.transform_to_world(pts[:,:3], pos, R)
            pts_w[:,2] *= -1
            labels = np.array(groundseg.run(pts_w))

            # Populate maps
            for p,lab in zip(pts_w, labels):
                if lab==1:      ground_map.add_point(*p, ts)
                elif p[2] > -pos[2]: obstacle_map.add_point(*p, ts)
                else:           ground_map.add_point(*p, ts)

            gpts = filter_points_by_radius(ground_map.get_estimate(), pos[:2], 15)
            x, y, z = gpts.T

            # Grid for risks
            start = np.array([-1,0]); dest = np.array([10,-5]); m=0.1; M=1
            mins = np.minimum(start,dest)-1; maxs = np.maximum(start,dest)+1
            xe = np.arange(mins[0], maxs[0]+m, m)
            ye = np.arange(mins[1], maxs[1]+m, m)
            xm = (xe[:-1]+xe[1:])/2; ym = (ye[:-1]+ye[1:])/2
            X,Y = np.meshgrid(xm, ym)
            Zg = np.full((len(xm),len(ym)), np.nan)
            for xi, yi, zi in zip(x,y,z):
                i = np.digitize(xi, xe)-1; j = np.digitize(yi, ye)-1
                if 0<=i<len(xm) and 0<=j<len(ym): Zg[i,j]=zi

            sr, sop = calculate_combined_risks(Zg, np.argwhere(~np.isnan(Zg)),
                                              max_height_diff=0.05,
                                              max_slope_degrees=30.0,
                                              radius=0.5)
            mr = sr; ms = sop  # misaligned names
            tot = np.ma.mean([ma.masked_invalid(sr), ma.masked_invalid(sop)], axis=0).filled(np.nan)

            # obstacles
            obst = filter_points_by_radius(obstacle_map.get_estimate(), pos[:2], 15)
            for ox, oy, _ in obst:
                i = np.clip(np.digitize(ox, xe)-1, 0, len(xm)-1)
                j = np.clip(np.digitize(oy, ye)-1, 0, len(ym)-1)
                tot[i,j] = 1.0
            print("jhere")
            # interpolate
            tot = interpolate_in_radius(tot, 1.5)
            cvar = compute_cvar_cellwise(ma.masked_invalid(tot), alpha=0.8)
            dist = np.hypot((X-pos[0]), (Y-pos[1]))
            cvar[dist.T>13] = np.nan

            # plot risk
            ax.clear()
            cmap = LinearSegmentedColormap.from_list('risk', [(0.5,0.5,0.5),(1,1,0),(1,0,0)])
            pcm = ax.pcolormesh(xm, ym, cvar.T, shading='auto', cmap=cmap, alpha=0.7)
            if cbar is None: cbar = fig.colorbar(pcm, ax=ax, label='Risk')
            else: cbar.update_normal(pcm)
            ax.set_aspect('equal')

            # smooth manual path
            mw = np.array(manual_world)
            smp = smooth_path(mw, window_size=5)
            ax.plot(smp[:,0], smp[:,1], '--m', lw=2, label='Path')
            ax.scatter(pos[0], pos[1], c='g', label='Start')
            ax.scatter(dest[0], dest[1], c='r', label='Goal')
            ax.legend()

            # Get vehicle yaw from rotation matrix
            yaw = np.arctan2(R[1,0], R[0,0])
            speed = 1.0  # desired speed

            # Closest path segment
            dists = np.linalg.norm(smp[:,:2] - pos[:2], axis=1)
            i0 = np.argmin(dists)
            seg = smp[i0:i0+10]
            if len(seg) < 2:
                continue

            # Heading for each path segment
            dxy = np.diff(seg[:,:2], axis=0)
            yaws = np.arctan2(dxy[:,1], dxy[:,0])
            yaws = np.append(yaws, yaws[-1])
            ref_traj = np.hstack([seg[:,:2], yaws[:,None]])

            # Solve MPC
            steer, throttle = mpc.solve([pos[0], pos[1], yaw, speed], ref_traj)

            # Send to AirSim
            ctr = airsim.CarControls()
            ctr.steering = np.clip(steer, -1, 1)
            ctr.throttle = np.clip(throttle, -1, 1)
            lidar_test.client.setCarControls(ctr)


            plt.draw(); plt.pause(0.1)

    finally:
        plt.ioff(); plt.show()