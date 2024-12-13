import open3d as o3d
import numpy as np
import trimesh

pcd = o3d.io.read_point_cloud("./scan716_aligned.ply")

# Height Filter Here

bbox = pcd.get_axis_aligned_bounding_box()
# Find the minimum and maximum bounds of the bounding box along the Z-axis (height)
min_bound, max_bound = bbox.min_bound[2], bbox.max_bound[2]
# Calculate the threshold at 50% of the bounding box height
threshold_z = min_bound + 0.5 * (max_bound - min_bound)
# Get all points in the point cloud
points = np.asarray(pcd.points)
# Filter points that are above the threshold height
filtered_points = points[points[:, 2] > threshold_z]
# Create a new point cloud with the filtered points
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

o3d.visualization.draw_geometries([filtered_pcd])

pcd = filtered_pcd

plane_normal = np.array([0, 0, 1])
plane_point = np.array([0, 0, 0])
centroid = plane_point
# 3. Project point cloud onto the plane
projected_points = []
distances = []
for point in np.asarray(pcd.points):
    distance = np.abs(point[2])  # Distance is just the z-coordinate
    distances.append(distance)
    projected_point = np.array([point[0], point[1], 0])  # Project to z=0
    projected_points.append(projected_point)

projected_pcd = o3d.geometry.PointCloud()
projected_pcd.points = o3d.utility.Vector3dVector(projected_points)

# 5. Create a grid on the plane
grid_size = 30
x_min, y_min, _ = np.min(projected_points, axis=0)
x_max, y_max, _ = np.max(projected_points, axis=0)
x_step = (x_max - x_min) / grid_size
y_step = (y_max - y_min) / grid_size

# 6. Count points in each grid cell
grid_counts = np.zeros((grid_size, grid_size))
for point in projected_points:
    x_idx = min(int((point[0] - x_min) / x_step), grid_size - 1)
    y_idx = min(int((point[1] - y_min) / y_step), grid_size - 1)
    grid_counts[x_idx, y_idx] += 1
# 6. Sum distances in each grid cell
grid_distance_sums = np.zeros((grid_size, grid_size))
for point, distance in zip(projected_points, distances):
    x_idx = min(int((point[0] - x_min) / x_step), grid_size - 1)
    y_idx = min(int((point[1] - y_min) / y_step), grid_size - 1)
    grid_distance_sums[x_idx, y_idx] += distance


# 7. Render grid cells exceeding the threshold
count_threshold = np.mean(grid_counts) * 0.5 # evans 3.1 / banana: 1.5
# print("np.mean(grid_counts)", np.mean(grid_counts))
# print("np.mean(grid_distance_sums)", np.mean(grid_distance_sums))
# print("count_threshold: ", count_threshold)
sum_threshold = np.mean(grid_distance_sums) * 0.3# evans 1.8 / banana: 1

grid_meshes = []
grid_mask = np.zeros((grid_size, grid_size), dtype=bool)

from scipy.ndimage import binary_closing, label

binary_grid = (grid_counts > count_threshold) & (grid_distance_sums > sum_threshold)

# 6. Denoise the binary grid
def denoise_grid(grid, min_neighbors):
    # Apply binary closing to connect nearby cells
    structure = np.ones((3, 3))
    closed_grid = binary_closing(grid, structure=structure)

    # Count neighbors for each cell
    neighbor_count = np.zeros_like(grid, dtype=int)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            neighbor_count += np.roll(np.roll(closed_grid, i, axis=0), j, axis=1)

    # Keep only cells with at least min_neighbors
    denoised_grid = (neighbor_count >= min_neighbors) & grid
    return denoised_grid


def filter_small_areas(grid, min_area):
    # Label connected components
    labeled_array, num_features = label(grid)

    # Count the size of each labeled area
    area_sizes = np.bincount(labeled_array.ravel())[1:]

    # Create a mask of areas larger than the minimum size
    large_areas_mask = np.in1d(labeled_array, np.where(area_sizes >= min_area)[0] + 1).reshape(grid.shape)

    return grid & large_areas_mask

min_area_threshold = 5
min_neighbors=0
denoised_grid = denoise_grid(binary_grid, min_neighbors)

filtered_grid = filter_small_areas(denoised_grid, min_area_threshold)
cuboid_meshes = []
for i in range(grid_size):
    for j in range(grid_size):
        if filtered_grid[i, j]:
            grid_mask[i, j] = True
            corner = np.array([x_min + i * x_step, y_min + j * y_step, 0])
            cell = o3d.geometry.TriangleMesh.create_box(width=x_step, height=y_step, depth=0.01)
            cell.translate(corner)
            cuboid = o3d.geometry.TriangleMesh.create_box(width=x_step, height=y_step, depth=2.01)
            cuboid.translate(corner)
            normalized_sum = (grid_distance_sums[i, j] - sum_threshold) / (np.max(grid_distance_sums) - sum_threshold)
            color = [normalized_sum, 0, 1 - normalized_sum]  # Blue to Red gradient
            cell.paint_uniform_color(color)
            grid_meshes.append(cell)
            cuboid_meshes.append(cuboid)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
o3d.visualization.draw_geometries(grid_meshes)

def combine_meshes(meshes):
    # Initialize vertices and triangles lists
    vertices = []
    triangles = []
    vertex_offset = 0

    for mesh in meshes:
        # Add vertices from this mesh
        vertices.extend(mesh.vertices)

        # Add triangles from this mesh, accounting for the offset
        triangles.extend([t + vertex_offset for t in mesh.triangles])

        # Update the offset for the next mesh
        vertex_offset += len(mesh.vertices)

    # Create a new mesh with combined vertices and triangles
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    combined_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return combined_mesh

combined_cuboid_mesh = combine_meshes(cuboid_meshes)
combined_grid_mesh = combine_meshes(grid_meshes)

hull, _ = combined_grid_mesh.compute_convex_hull()
# Visualize the result
o3d.visualization.draw_geometries([combined_cuboid_mesh, hull])
# plane_normal = np.array([0, 0, 1])  # Z-axis normal (slices horizontally)
# plane_origin = np.array([0, 0, 0])  # Plane passes through the origin (Z=0)
# trimesh_cuboid = trimesh.Trimesh(vertices=np.asarray(combined_cuboid_mesh.vertices), faces=np.asarray(combined_cuboid_mesh.triangles))
#
# cross_section = trimesh_cuboid.section(plane_origin=plane_origin, plane_normal=plane_normal)
#
# if cross_section:
#     # Convert the slice to a planar mesh (2D polygon)
#     slice_mesh, transform_matrix = cross_section.to_planar()
#
#     vertices = np.array(slice_mesh.vertices)
#     edges = np.array(slice_mesh.entities[0].points)
#
#     # Step 5: Create an Open3D LineSet for visualization
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(vertices)
#     line_set.lines = o3d.utility.Vector2iVector(edges)
#
#     # Step 5: Visualize the cross-section
#     o3d.visualization.draw_geometries([combined_cuboid_mesh, line_set])
# else:
#     print("No valid intersection was found.")
#trimesh_cuboid = trimesh.Trimesh(vertices=np.asarray(combined_cuboid_mesh.vertices), faces=np.asarray(combined_cuboid_mesh.triangles))
#trimesh_plane = trimesh.Trimesh(vertices=np.asarray(hull.vertices), faces=np.asarray(hull.triangles))
#intersection_mesh = trimesh_cuboid.intersection(trimesh_plane)

#intersection_o3d_mesh = o3d.geometry.TriangleMesh(
 #   vertices=o3d.utility.Vector3dVector(intersection_mesh.vertices),
  #  triangles=o3d.utility.Vector3iVector(intersection_mesh.faces)
#)
#o3d.visualization.draw_geometries(grid_meshes + [intersection_o3d_mesh])

o3d.io.write_triangle_mesh("./716_hull.obj", hull)
o3d.io.write_triangle_mesh("./716_wall.obj", combined_cuboid_mesh)

