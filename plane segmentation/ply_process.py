import numpy as np
import open3d as o3d

# Read point cloud:
pcd = o3d.io.read_point_cloud("build copy/results1_030_039.ply")
# Create a 3D coordinate system:
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
# geometries to draw:
geometries = [pcd, origin]
# Visualize:
o3d.visualization.draw_geometries(geometries)