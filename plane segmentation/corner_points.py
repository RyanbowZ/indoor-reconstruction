import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
def create_sphere(center, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere
def angle_between(v1, v2):
    """Compute the angle between two vectors"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def find_corner_points_with_angle_threshold(plane_model, points, angle_threshold):
    # Extract plane parameters
    a, b, c, d = plane_model

    # Project points onto the plane
    norm_vec = np.array([a, b, c])
    proj_matrix = np.eye(3) - np.outer(norm_vec, norm_vec) / np.dot(norm_vec, norm_vec)
    projected_points = np.dot(points, proj_matrix.T)

    # Find two orthogonal vectors in the plane
    u = np.cross(norm_vec, [1, 0, 0])
    if np.allclose(u, 0):
        u = np.cross(norm_vec, [0, 1, 0])
    v = np.cross(norm_vec, u)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    # Project points onto 2D coordinate system in the plane
    points_2d = np.column_stack((np.dot(projected_points, u), np.dot(projected_points, v)))

    # Compute convex hull
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]

    # Filter hull points based on angle threshold
    filtered_hull_points = []
    # for i in range(1, len(hull_points) + 1):
    #     p1 = filtered_hull_points[-1]
    #     p2 = hull_points[i % len(hull_points)]
    #     if len(filtered_hull_points) > 1:
    #         p0 = filtered_hull_points[-2]
    #         v1 = p1 - p0
    #         v2 = p2 - p1
    #         angle = angle_between(v1, v2)
    #         if angle > angle_threshold:
    #             filtered_hull_points.append(p2)
    #     else:
    #         filtered_hull_points.append(p2)
    for i in range(1, len(hull_points) + 1):
        p1 = hull_points[i % len(hull_points)]
        if len(hull_points) > 1:
            p0 = hull_points[(i - 1) % len(hull_points)]
            p2 = hull_points[(i + 1) % len(hull_points)]
            v1 = p1 - p0
            v2 = p2 - p1
            angle = angle_between(v1, v2)
            if angle > angle_threshold:
                filtered_hull_points.append(p1)
        else:
            filtered_hull_points.append(p1)
    filtered_hull_points = np.array(filtered_hull_points)
    # Convert 2D corner points back to 3D
    corner_points_3d = np.dot(filtered_hull_points[:, 0][:, np.newaxis] * u +
                              filtered_hull_points[:, 1][:, np.newaxis] * v,
                              proj_matrix)

    return corner_points_3d

line_count = 0
floor_vertex_list = []

begin_scale = 30

with open("./build copy/results1_030_039.txt", "r") as file:
    for lines in file:
        line_count += 1
        if int(lines) == 0:
            floor_vertex_list.append(line_count) # index of vertices
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

xavg = xavg / avgcount
yavg = yavg / avgcount
zavg = zavg / avgcount
print("normal avg:", xavg, yavg, zavg)

normals = np.asarray([xavg, yavg, zavg])

import numpy as np
import open3d as o3d


pcd = o3d.io.read_point_cloud("./rooms_full.ply")
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
geometries = [pcd, origin]

pcd_remove_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list], invert=True)
pcd_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list])

plane_model, inliers = pcd_ground.segment_plane(distance_threshold=0.1,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print("plane model parameter:", a, b, c, d)

plane_normal = np.asarray([a, b, c])
centroid = np.mean(np.asarray(pcd_ground.points), axis=0)

corner_points_3d = find_corner_points_with_angle_threshold(plane_model, pcd_ground.points, np.radians(20))
# print(corner_points_3d)
corner_pcd = o3d.geometry.PointCloud()
corner_pcd.points = o3d.utility.Vector3dVector(corner_points_3d)
corner_pcd.paint_uniform_color([1, 0, 0])
projected_spheres = []
for p in corner_pcd.points:
    projected_sphere = create_sphere(p, 0.08, [1, 0, 0])
    projected_spheres.append(projected_sphere)

apple_npy = np.load('apple_corners_mean.npy')
banana_npy = np.load('banana_corners_mean.npy')
cup_npy = np.load('cup_corners_mean.npy')
d = -np.dot(plane_normal, centroid)
apple_distance = np.dot(apple_npy[:8], plane_normal) + d
apple_points = apple_npy[:8] - np.outer(apple_distance, plane_normal)
banana_distance = np.dot(banana_npy[:8], plane_normal) + d
banana_points = banana_npy[:8] - np.outer(banana_distance, plane_normal)
cup_distance = np.dot(cup_npy[:8], plane_normal) + d
cup_points = cup_npy[:8] - np.outer(cup_distance, plane_normal)


for p in apple_points:
    apple_sphere = create_sphere(p, 0.05, [0, 1, 1])
    projected_spheres.append(apple_sphere)

for p in banana_points:
    banana_sphere = create_sphere(p, 0.05, [1, 1, 0])
    projected_spheres.append(banana_sphere)

for p in cup_points:
    cup_sphere = create_sphere(p, 0.05, [1, 0, 1])
    projected_spheres.append(cup_sphere)


def create_line_set(vertices):
    # Create a LineSet
    line_set = o3d.geometry.LineSet()

    # Set the points
    line_set.points = o3d.utility.Vector3dVector(vertices)

    # Create lines connecting the points in order
    lines = [[i, (i + 1) % len(vertices)] for i in range(len(vertices))]
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set


def compute_polygon_area(vertices):
    # Use the Shoelace formula to compute the area
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    return area

line_set = create_line_set(corner_pcd.points)
colors = [[1, 0, 0] for _ in range(len(lines))]
line_set.colors = o3d.utility.Vector3dVector(colors)
area = compute_polygon_area(corner_pcd.points)

print("area constructed by the vertices: ", area)

o3d.visualization.draw_geometries([corner_pcd, line_set] + projected_spheres)
