import os
import yaml
import pickle
import argparse
import numpy as np
from os.path import join, exists
from sklearn.neighbors import KDTree
from utils.data_process import DataProcessing as DP
import re

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', default='data', help='source dataset path [default: None]')
parser.add_argument('--dst_path', default='output', help='destination dataset path [default: None]')
parser.add_argument('--grid_size', type=float, default=0.05, help='Subsample Grid Size [default: 0.06]')
parser.add_argument('--yaml_config', default='utils/semantic-kitti.yaml', help='semantic-kitti.yaml path')
FLAGS = parser.parse_args()


data_config = FLAGS.yaml_config
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

grid_size = FLAGS.grid_size
dataset_path = FLAGS.src_path
output_path = FLAGS.dst_path
seq_list = np.sort(os.listdir(dataset_path))
print('seq_list', seq_list)
for seq_id in seq_list:
    print('sequence ' + seq_id + ' start')
    seq_path = join(dataset_path, seq_id)
    # print('seq_path', seq_path)
    seq_path_out = join(output_path, seq_id)
    pc_path_out = join(seq_path_out, 'velodyne')
    KDTree_path_out = join(seq_path_out, 'KDTree')
    label_path_out = join(seq_path_out, 'labels')
    os.makedirs(seq_path_out) if not exists(seq_path_out) else None
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None
    os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None
    os.makedirs(label_path_out) if not exists(label_path_out) else None
    
    # Get the list of frames in the sequence
    names = os.listdir(seq_path)
    pattern_ply = re.compile(r'^[^\.]+\.ply$')
    names_json = [n for n in names if pattern_ply.match(n)]
    
    for scan_id in range(1, len(names_json)):
   
        pc_path = join(seq_path, f"{scan_id}.ply")
        label_path = join(seq_path, f"{scan_id}_labels.npy")
        points = DP.load_pc_kitti(pc_path)
        labels = DP.load_label_kitti(label_path)
        # print('points', points.shape)
        sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)
        # print('sub_points', sub_points.shape)
        # print("a-------")
        search_tree = KDTree(sub_points)
        KDTree_save = join(KDTree_path_out, str(scan_id) + '.pkl')
        # print('KDTree_save', KDTree_save)
     
        np.save(join(pc_path_out, str(scan_id) + '.npy'), sub_points)
        np.save(join(label_path_out, str(scan_id)+ '.npy'), sub_labels)
        with open(KDTree_save, 'wb') as f:
            pickle.dump(search_tree, f)
    #     if seq_id == '08':
    #         proj_path = join(seq_path_out, 'proj')
    #         os.makedirs(proj_path) if not exists(proj_path) else None
    #         proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
    #         proj_inds = proj_inds.astype(np.int32)
    #         proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
    #         with open(proj_save, 'wb') as f:
    #             pickle.dump([proj_inds], f)
#     else:
#         proj_path = join(seq_path_out, 'proj')
#         os.makedirs(proj_path) if not exists(proj_path) else None
#         scan_list = np.sort(os.listdir(pc_path))
#         for scan_id in scan_list:
#             print(scan_id)
#             points = DP.load_pc_kitti(join(pc_path, scan_id))
#             sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
#             search_tree = KDTree(sub_points)
#             proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
#             proj_inds = proj_inds.astype(np.int32)
#             KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
#             proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
#             np.save(join(pc_path_out, scan_id)[:-4], sub_points)
#             with open(KDTree_save, 'wb') as f:
#                 pickle.dump(search_tree, f)
#             with open(proj_save, 'wb') as f:
#                 pickle.dump([proj_inds], f)
