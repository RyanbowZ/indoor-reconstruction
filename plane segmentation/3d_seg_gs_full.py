import open3d as o3d
import numpy as np
import json
import glob

rootpath = '716'
# Step 1: Load the .ply file into Open3D
ply_file = f'{rootpath}/splat_aligned.ply'
mesh = o3d.io.read_triangle_mesh(ply_file)

# Step 2: Load the vertex indices from the .npy file
npy_file = f'{rootpath}/label.npy'
json_file = f'{rootpath}/label.json'
data = np.load(npy_file)
json_content = json.loads(open(json_file).read())
threshold = 50

import matplotlib.pyplot as plt
colors = [
        [1, 0, 0],   # Red
        [0, 1, 0],   # Green
        [0, 0, 1],   # Blue
        [1, 1, 0],   # Yellow
        [1, 0, 1],   # Magenta
        [0, 1, 1],   # Cyan
        [0.5, 0, 0], # Dark Red
        [0, 0.5, 0], # Dark Green
        [0, 0, 0.5], # Dark Blue
        [0.5, 0.5, 0], # Olive
        [0.5, 0, 0.5], # Purple
        [0, 0.5, 0.5], # Teal
        [1, 0.5, 0],   # Orange
        [0, 1, 0.5],   # Spring Green
        [0.5, 0, 1],   # Violet
        [1, 0.5, 0.5], # Light Coral
        [0.5, 1, 0.5], # Pale Green
        [0.5, 0.5, 1], # Light Blue
        [0.75, 0.75, 0], # Gold
        [0.75, 0, 0.75]  # Orchid
    ]
pcds = []
for i in range(0, 20):
    vertex_indices = np.where(data == str(i))[0]
    color = colors[i % len(colors)]
    if len(vertex_indices) > threshold:
        vertices = np.asarray(mesh.vertices)
        selected_vertices = vertices[vertex_indices]
        label_name = json_content[str(i)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(selected_vertices)

        centroid = pcd.get_center()
        print(f"{i}th {label_name}_Centroid: {centroid}")
        pcd.paint_uniform_color(color)
        pcds.append(pcd)
        # o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud(f"{rootpath}/seg_{label_name}_{i}.ply", pcd)
o3d.visualization.draw_geometries(pcds)
