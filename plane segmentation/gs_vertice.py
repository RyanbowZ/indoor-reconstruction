import open3d as o3d
import numpy as np

# Step 1: Load the .ply file into Open3D
ply_file = 'scan707_splat_aligned.ply'
mesh = o3d.io.read_triangle_mesh(ply_file)

# Step 2: Load the vertex indices from the .npy file
npy_file = './scan707/label.npy'
data = np.load(npy_file)
# print(data)
vertex_indices = np.where(data == '3')[0]
# print(vertex_indices)
# Step 3: Select the vertices using the indices
# Get the vertices from the mesh
vertices = np.asarray(mesh.vertices)

# Select the vertices corresponding to the indices
selected_vertices = vertices[vertex_indices]

# Create a point cloud from the selected vertices
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(selected_vertices)

centroid = pcd.get_center()
print(f"Centroid: {centroid}")

# Step 4: Render the selected vertices
o3d.visualization.draw_geometries([pcd])
