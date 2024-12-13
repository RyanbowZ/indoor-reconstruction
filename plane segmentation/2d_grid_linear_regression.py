import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing, label


def detect_major_lines(grid_mask, threshold_percentage=0.5):
    """
    Detect non-adjacent major horizontal and vertical lines in the grid.

    Args:
        grid_mask: Binary 2D numpy array representing occupied cells
        threshold_percentage: Minimum percentage of occupied cells to consider as a major line

    Returns:
        major_line_grid: Binary 2D numpy array with only major lines
        horizontal_lines: Indices of horizontal major lines
        vertical_lines: Indices of vertical major lines
    """
    rows, cols = grid_mask.shape
    major_line_grid = np.zeros_like(grid_mask)

    # Analyze rows (horizontal lines)
    row_occupancy = np.sum(grid_mask, axis=1) / cols
    potential_rows = np.where(row_occupancy >= threshold_percentage)[0]

    # Filter adjacent horizontal lines
    horizontal_lines = []
    prev_row = -2  # Initialize with impossible index
    for row in potential_rows:
        if row > prev_row + 1:  # Ensure non-adjacent
            horizontal_lines.append(row)
            prev_row = row

    # Analyze columns (vertical lines)
    col_occupancy = np.sum(grid_mask, axis=0) / rows
    potential_cols = np.where(col_occupancy >= threshold_percentage)[0]

    # Filter adjacent vertical lines
    vertical_lines = []
    prev_col = -2  # Initialize with impossible index
    for col in potential_cols:
        if col > prev_col + 1:  # Ensure non-adjacent
            vertical_lines.append(col)
            prev_col = col

    # Fill in major lines
    for row in horizontal_lines:
        major_line_grid[row, :] = grid_mask[row, :]

    for col in vertical_lines:
        # Avoid overlapping with horizontal lines
        non_horizontal = ~np.any(major_line_grid[:, col].reshape(-1, 1), axis=1)
        major_line_grid[:, col] = grid_mask[:, col] & non_horizontal

    return major_line_grid, horizontal_lines, vertical_lines


def snap_noise_to_major_lines(grid_mask, major_line_grid, horizontal_lines, vertical_lines, max_distance=2):
    """
    Snap noise grids to their closest major line if within max_distance.

    Args:
        grid_mask: Original binary grid with noise
        major_line_grid: Binary grid containing only major lines
        horizontal_lines: Indices of horizontal major lines
        vertical_lines: Indices of vertical major lines
        max_distance: Maximum distance to consider for snapping

    Returns:
        snapped_grid: Binary grid with noise snapped to major lines
    """
    rows, cols = grid_mask.shape
    snapped_grid = major_line_grid.copy()

    # For each occupied cell in the original grid that's not part of a major line
    for i in range(rows):
        for j in range(cols):
            if grid_mask[i, j] and not major_line_grid[i, j]:
                # Find distances to nearest horizontal and vertical lines
                h_distances = [abs(i - h) for h in horizontal_lines]
                v_distances = [abs(j - v) for v in vertical_lines]

                min_h_dist = min(h_distances) if h_distances else float('inf')
                min_v_dist = min(v_distances) if v_distances else float('inf')

                # Only snap if within max_distance
                if min(min_h_dist, min_v_dist) <= max_distance:
                    if min_h_dist < min_v_dist:
                        # Snap to horizontal line
                        nearest_h_line = horizontal_lines[np.argmin(h_distances)]
                        snapped_grid[nearest_h_line, j] = True
                    else:
                        # Snap to vertical line
                        nearest_v_line = vertical_lines[np.argmin(v_distances)]
                        snapped_grid[i, nearest_v_line] = True

    return snapped_grid


# Modified main code
def process_floor_plan(binary_grid, threshold_percentage=0.3, max_snap_distance=2):
    """
    Process the floor plan grid to detect major lines and snap noise.
    """
    # Detect major lines
    major_line_grid, horizontal_lines, vertical_lines = detect_major_lines(
        binary_grid,
        threshold_percentage=threshold_percentage
    )

    # Snap noise grids to major lines
    final_grid = snap_noise_to_major_lines(
        binary_grid,
        major_line_grid,
        horizontal_lines,
        vertical_lines,
        max_distance=max_snap_distance
    )

    return final_grid, horizontal_lines, vertical_lines


# Modified main code
pcd = o3d.io.read_point_cloud("./scan716_aligned.ply")

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
cleaned_grid, horizontal_lines, vertical_lines = process_floor_plan(
    filtered_grid,
    threshold_percentage=0.25,  # Adjust as needed
    max_snap_distance=2        # Adjust as needed
)

# Create visualization geometries
grid_meshes = []
for i in range(grid_size):
    for j in range(grid_size):
        if cleaned_grid[i, j]:
            corner = np.array([x_min + i * x_step, y_min + j * y_step, 0])
            cell = o3d.geometry.TriangleMesh.create_box(width=x_step, height=y_step, depth=0.01)
            cell.translate(corner)
            # Color cells based on whether they're part of horizontal or vertical lines
            is_horizontal = np.sum(cleaned_grid[i, :]) > np.sum(cleaned_grid[:, j])
            color = [1, 0, 0] if is_horizontal else [0, 0, 1]  # Red for horizontal, Blue for vertical
            cell.paint_uniform_color(color)
            grid_meshes.append(cell)

# Visualize the results
o3d.visualization.draw_geometries(grid_meshes)
# combined_point_cloud = o3d.geometry.PointCloud()
#
# # Combine vertices and colors from all meshes into the PointCloud
# for mesh in grid_meshes:
#     # Extract vertices and colors
#     vertices = np.asarray(mesh.vertices)
#     colors = np.asarray(mesh.vertex_colors)
#
#     # Append vertices and colors to the combined PointCloud
#     combined_point_cloud.points.extend(o3d.utility.Vector3dVector(vertices))
#     combined_point_cloud.colors.extend(o3d.utility.Vector3dVector(colors))
#
# # Save the combined PointCloud to a .ply file
# o3d.io.write_point_cloud("floor_plan_grid.ply", combined_point_cloud)

combined_mesh = o3d.geometry.TriangleMesh()
# Combine vertices, triangles and colors from all meshes
for mesh in grid_meshes:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    # Offset triangle indices by current number of vertices
    triangles += len(np.asarray(combined_mesh.vertices))
    # Extend combined mesh
    combined_mesh.vertices.extend(vertices)
    combined_mesh.triangles.extend(triangles)
    combined_mesh.vertex_colors.extend(mesh.vertex_colors)
# Save the combined mesh
o3d.io.write_triangle_mesh("floor_plan_grid.obj", combined_mesh)