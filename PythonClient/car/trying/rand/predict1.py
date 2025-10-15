import torch
import numpy as np
from sklearn.neighbors import KDTree
from network.RandLANet import Network
from utils.config import Config10labels as cfg
from utils.data_process import DataProcessing as DP
import open3d as o3d
class RandlaGroundSegmentor:
    def __init__(self, ckpt_path='log/checkpoint_slope_only.tar', device=None, subsample_grid=0.1):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Network(cfg).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.num_points = cfg.num_points
        self.subsample_grid = subsample_grid
        self.k_n = cfg.k_n
        self.num_layers = cfg.num_layers
        self.sub_ratio = cfg.sub_sampling_ratio

    def segment(self, points_world):
        """
        points_world: (M,3) np.array of live LiDAR points
        returns:
          - if return_probs is False: labels_world (M,) ints
          - if return_probs is True: (labels_world (M,), probs_world (M,C)) with per‑class probabilities
        """
        # --- 1) statistical outlier removal ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        clean_pts = np.asarray(pcd.points, dtype= np.float32)

        # --- 2) voxel‐grid subsample to reduce density ---
        pts = DP.grid_sub_sampling(clean_pts, grid_size=self.subsample_grid)

        # --- 3) enforce exactly num_points via random trunc/pad ---
        N = pts.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            dup = np.random.choice(N, self.num_points - N, replace=True)
            idx = np.concatenate([np.arange(N), dup])
        sampled = pts[idx]

        # --- 4) build per-layer xyz lists and features ---
        xyz_list = [
            torch.from_numpy(sampled).float().to(self.device).unsqueeze(0)
        ]  # [1, N, 3]
        feat = xyz_list[0].permute(0, 2, 1).contiguous()  # [1, 3, N]

        neigh_idx, sub_idx, interp_idx = [], [], []
        pts_cur = sampled
        for i in range(self.num_layers):
            # k-NN at this scale
            knn = DP.knn_search(pts_cur[None, ...], pts_cur[None, ...], self.k_n)[0]
            neigh_idx.append(
                torch.from_numpy(knn).long().to(self.device).unsqueeze(0)
            )

            # subsample
            n_sub = pts_cur.shape[0] // self.sub_ratio[i]
            sel = np.random.permutation(pts_cur.shape[0])[:n_sub]
            sub_idx.append(
                torch.from_numpy(sel[None, ...]).long().to(self.device).unsqueeze(-1)
            )

            # build next‐level points
            pts_next = pts_cur[sel]
            xyz_list.append(
                torch.from_numpy(pts_next).float().to(self.device).unsqueeze(0)
            )

            # interpolation indices back
            interp = DP.knn_search(pts_next[None, ...], pts_cur[None, ...], 1)[0]
            interp_idx.append(
                torch.from_numpy(interp).long().to(self.device).unsqueeze(0)
            )

            pts_cur = pts_next

        # --- 5) inference ---
        with torch.no_grad():
            inputs = {
                "xyz":        xyz_list,
                "features":   feat,
                "neigh_idx":  neigh_idx,
                "sub_idx":    sub_idx,
                "interp_idx": interp_idx
            }
            end_pts = self.model(inputs)
            # logits: [1, C, N]
            logits = end_pts["logits"]
            preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [N]
            # probs_sampled: [N, C]
            probs_sampled = torch.softmax(logits, dim=1).squeeze(0).permute(1, 0).contiguous().cpu().numpy()

        # --- 6) project back to original M points ---
        tree = KDTree(sampled)
        nn = tree.query(points_world, return_distance=False).squeeze(-1)  # [M]
        labels_world = preds[nn]

        probs_world = probs_sampled[nn]  # [M, C]
        return labels_world, probs_world

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d
# from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
# # 1) instantiate model ONCE
# seg = RandlaGroundSegmentor()

# # 2) set up interactive plotting
# plt.ion()
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# for i in range(1,150):
#     point_path = f"../data/4/{i}.ply"
#     pcd = o3d.io.read_point_cloud(point_path)
#     points_np = np.asarray(pcd.points, dtype= np.float32)
    

#     # 3) segment once per frame
#     labels, probs = seg.segment(points_np)
#     # print(np.unique(probs))
#     # print("Unique labels:", np.unique(labels))
#     # assert len(labels) == len(points_np)
#     # print(f"Frame {i}, points: {len(points_np):,}, labels: {len(labels):,}")
#     # print("len of probs: ", len(probs))
#     # print("probs: ", probs[0])
#     # print("probs shape: ", probs[0].shape)
#     # print("max of probs: ", np.max(probs[0]))
#     # print("maxz pos of probs: ", np.argmax(probs[0]))
#     # print("labels: ", labels[0])
#     for i in range(len(probs)):
#         # print("probs: ", probs[i])
#         print("max of probs: ", np.max(probs[i]))
#         # print("maxz pos of probs: ", np.argmax(probs[i]))
#         # print("labels: ", labels[i])
      

#     num_classes = 51
#     labels_i = np.clip(labels.astype(int), 0, num_classes - 1)

#     colors_list = [
#             (0.5, 0.5, 0.5),  # gray
#             (1.0, 1.0, 0.0),  # yellow
#             (1.0, 0.5, 0.0),  # orange
#             (1.0, 0.0, 0.0),  # red
#             (0.0, 0.0, 0.0),  # black
#     ]
#     cmap = LinearSegmentedColormap.from_list(
#         "gray_yellow_orange_red_black", colors_list, N=num_classes
#     )
#     norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, cmap.N)

#     # 5) clear the old scatter and draw new one
#     ax.clear()
#     ax.scatter(
#         points_np[:,0], points_np[:,1], points_np[:,2],
#         c=labels_i, cmap=cmap, norm=norm,
#         s=0.5, depthshade=False
#     )
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(f'Ground vs. Non-Ground (frame {i})')

#     # 6) redraw & pause briefly
#     plt.draw()
#     plt.pause(0.1)
#     break
# # finish
# plt.ioff()
# plt.show()
