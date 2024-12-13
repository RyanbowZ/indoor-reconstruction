import open3d as o3d
import numpy as np
import os
import glob
import re

# Define the directory containing .npy files
folder_path = './seg_object_on_slam'

# Use glob to search for all .npy files in the folder
npy_files = glob.glob(os.path.join(folder_path, '*.npy'))

pcd = o3d.io.read_point_cloud("./evans_aligned.ply")
vertices = np.asarray(pcd.points)
# Iterate through all .npy files
for npy_file in npy_files:
    # Load the .npy file
    index = np.load(npy_file)
    i = npy_file.split('/')[-1].split('.')[0].split('_')[0]
    # Process the data
    print(f"Loaded data from {npy_file} with index {i}:")
    print(index.shape)

    selected_vertices = vertices[index]

    seg_ply = o3d.geometry.PointCloud()
    seg_ply.points = o3d.utility.Vector3dVector(selected_vertices)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        seg_ply.colors = o3d.utility.Vector3dVector(colors[index])

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        seg_ply.normals = o3d.utility.Vector3dVector(normals[index])

    o3d.visualization.draw_geometries([seg_ply])