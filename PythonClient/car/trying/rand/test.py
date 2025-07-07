import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load point cloud and labels
points = np.load("output/0/velodyne/5.npy")  # (N, 4)
labels_gt = np.load("output/0/labels/5.npy")  # (N,)
# labels_gt = np.load("result/0/predictions/5.npy")  # (N,)

# Extract XYZ (ignore reflectance)
xyz = points[:, :3]

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Unique labels and assign color to each
labels = labels_gt.astype(np.int32)
unique_labels = np.unique(labels)
colormap = plt.get_cmap("tab20")

# Map each unique label to a color
label_to_color = {label: colormap(i / len(unique_labels))[:3]
                  for i, label in enumerate(unique_labels)}

# Apply colors to each point (cast label to int for dict lookup)
colors = np.array([label_to_color[int(label)] for label in labels])
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])

# import numpy as np
# import open3d as o3d
# import os
# # Load
# # Get len of the dataset]\
# test_id = 0

# label_path = f'result/{test_id}/predictions'
# get_len = len(os.listdir(label_path))
# for i in range(1, get_len + 1):
#     points = np.load(f"output/{test_id}/velodyne/{i}.npy")       # (N, 4)
#     # labels = np.load(f"output/{test_id}/labels/{i}.npy").astype(int).ravel()  # (N,)
#     labels = np.load(label_path+ f"/{i}.npy").astype(int).ravel()  # (N,)
#     print(points.shape)
#     print(labels.shape)
#     # Point cloud
#     xyz = points[:, :3]
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)

#     # 3) Build a strict 0→black, 1→green, 2→red mapping
#     label_to_color = {
#         0: (0.0, 0.0, 0.0),  # black
#         1: (0.0, 1.0, 0.0),  # green
#         2: (1.0, 0.0, 0.0),  # red
#     }

#     # 4) Convert each integer label to its RGB triple
#     colors = np.array([label_to_color[int(lbl)] for lbl in labels], dtype=np.float64)
#     print(colors)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # 5) (Optional) sanity‐check that colors were applied
#     print("  → pcd.has_colors() =", pcd.has_colors())  # should be True

#     # 6) Visualize (you’ll see only black, green, or red)
#     o3d.visualization.draw_geometries([pcd])

#     # Now this draw call will show exactly black/green/red
#     # o3d.visualization.draw_geometries([pcd])
# import numpy as np
# import open3d as o3d
# import os

# test_id    = 0
# label_path = f"result/{test_id}/predictions"
# n_frames   = len(os.listdir(label_path))

# # for i in range(1, n_frames + 1):
# i =5
# # ── 1) Load your points (N×4) and labels (N,) ─────────────────────────────
# points = np.load(f"output/{test_id}/velodyne/{i}.npy")       # (N,4)
# labels = np.load(os.path.join(label_path, f"{i}.npy")).astype(int).ravel()  # (N,)

# # Quick sanity check: make sure you have exactly one label per point
# if points.shape[0] != labels.shape[0]:
#     raise ValueError(
#         f"Frame {i}: #points = {points.shape[0]} but #labels = {labels.shape[0]}"
#     )

# # ── 2) Build an Open3D PointCloud for the XYZ coords ───────────────────────
# xyz = points[:, :3]  # (N,3)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)

# # ── 3) Create a zero‐array of shape (N,3) and fill masks ─────────────────────
# N = labels.shape[0]
# colors = np.zeros((N, 3), dtype=np.float64)

# # Mask where label == 1 → paint green:
# mask1 = (labels == 1)
# colors[mask1, :] = np.array([0.0, 1.0, 0.0])

# # Mask where label == 2 → paint red:
# mask2 = (labels == 2)
# colors[mask2, :] = np.array([1.0, 0.0, 0.0])

# # Mask where label == 0 stays [0,0,0] (black)

# # ── 4) Assign those RGBs to the point cloud ───────────────────────────────
# pcd.colors = o3d.utility.Vector3dVector(colors)

# # (Optional) Check that Open3D actually sees colors:
# print(f"Frame {i}: pcd.has_colors() = {pcd.has_colors()}")  # should be True

# # ── 5) Visualize: you will only see black/green/red ───────────────────────
# o3d.visualization.draw_geometries([pcd])
