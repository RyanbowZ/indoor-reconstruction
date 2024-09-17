import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("./rooms_full.ply")
line_count = 0
floor_vertex_list = []
with open("./build copy/results1_030_039.txt", "r") as file:
    for lines in file:
        line_count += 1
        if int(lines) == 0:
            floor_vertex_list.append(line_count)  # index of vertices
print(floor_vertex_list)
line_count = 0

# get normal directions
xavg = 0
yavg = 0
zavg = 0
avgcount = 0

with open("./build copy/mycloud_features.txt", "r") as file:
    for lines in file:
        line_content = lines.split()
        point_count = 38398
        scale_count = 50
        if len(line_content) == 2:
            point_count = line_content[0]
            scale_count = line_content[1]
        else:
            line_count += 1
            x_range = 0
            y_range = 0
            z_range = 0
            for scales in range(30, 40):
                x_range += float(line_content[scales * 5 + 0])
                y_range += float(line_content[scales * 5 + 1])
                z_range += float(line_content[scales * 5 + 2])
            x = x_range / 10
            y = y_range / 10
            z = z_range / 10

            if line_count in floor_vertex_list and x != 0 and y != 0 and z != 0 and not np.isnan(x):
                avgcount += 1
                xavg += x
                yavg += y
                zavg += z
                print("=======line count: ", line_count)
                print(x, y, z)

pcd_remove_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list], invert=True)
pcd_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list])

plane_model, inliers = pcd_ground.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=1000)
[a, b, c, d] = plane_model
centroid = np.mean(np.asarray(pcd_ground.points), axis=0)
plane_point = centroid
plane_normal = np.asarray([a, b, c])


# 3. Project point cloud onto the plane
projected_points = []
distances = []
for point in np.asarray(pcd.points):
    distance = np.abs(np.dot(plane_normal, point - centroid))
    distances.append(distance)
    projected_point = point - distance * plane_normal
    projected_points.append(projected_point)

projected_pcd = o3d.geometry.PointCloud()
projected_pcd.points = o3d.utility.Vector3dVector(projected_points)

# 4. Create a coordinate system on the plane
up = np.array([0, 0, 1])
axis1 = np.cross(up, plane_normal)
axis1 = axis1 / np.linalg.norm(axis1)
axis2 = np.cross(plane_normal, axis1)

# 5. Create a grid on the plane
grid_size = 50
projected_points_2d = []
for point in projected_points:
    point_centered = point - centroid
    x = np.dot(point_centered, axis1)
    y = np.dot(point_centered, axis2)
    projected_points_2d.append([x, y])

projected_points_2d = np.array(projected_points_2d)
x_min, y_min = np.min(projected_points_2d, axis=0)
x_max, y_max = np.max(projected_points_2d, axis=0)
x_step = (x_max - x_min) / grid_size
y_step = (y_max - y_min) / grid_size

# 6. Count points in each grid cell
grid_counts = np.zeros((grid_size, grid_size))
for point in projected_points_2d:
    x_idx = min(int((point[0] - x_min) / x_step), grid_size - 1)
    y_idx = min(int((point[1] - y_min) / y_step), grid_size - 1)
    grid_counts[x_idx, y_idx] += 1
# 6. Sum distances in each grid cell
grid_distance_sums = np.zeros((grid_size, grid_size))
for point_2d, distance in zip(projected_points_2d, distances):
    x_idx = min(int((point_2d[0] - x_min) / x_step), grid_size - 1)
    y_idx = min(int((point_2d[1] - y_min) / y_step), grid_size - 1)
    grid_distance_sums[x_idx, y_idx] += distance


# 7. Render grid cells exceeding the threshold
threshold = np.mean(grid_distance_sums) * 1.7   # Adjust this value as needed
print("np.mean(grid_counts)", np.mean(grid_counts))
print("np.mean(grid_distance_sums)", np.mean(grid_distance_sums))
print("threshold: ", threshold)

grid_meshes = []
grid_mask = np.zeros((grid_size, grid_size), dtype=bool)

for i in range(grid_size):
    for j in range(grid_size):
        if grid_distance_sums[i, j] > threshold:
            grid_mask[i, j] = True
            corner_2d = np.array([x_min + i * x_step, y_min + j * y_step])
            corner_3d = centroid + corner_2d[0] * axis1 + corner_2d[1] * axis2

            grid_cell = o3d.geometry.TriangleMesh.create_box(width=x_step, height=y_step, depth=0.01)
            R = np.column_stack((axis1, axis2, plane_normal))
            grid_cell.rotate(R, center=(0, 0, 0))
            grid_cell.translate(corner_3d)
            # grid_cell.paint_uniform_color([1, 0, 0])  # Red color for cells exceeding threshold
            # Color the grid cell based on the distance sum
            normalized_sum = (grid_distance_sums[i, j] - threshold) / (np.max(grid_distance_sums) - threshold)
            color = [normalized_sum, 0, 1 - normalized_sum]  # Blue to Red gradient
            grid_cell.paint_uniform_color(color)
            grid_meshes.append(grid_cell)

# Visualize the result
o3d.visualization.draw_geometries([projected_pcd] + grid_meshes)