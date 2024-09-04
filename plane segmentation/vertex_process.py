import numpy as np
line_count = 0
floor_vertex_list = []

begin_scale = 20

with open("./build copy/results1_0"+str(begin_scale)+"_0"+str(begin_scale+9)+".txt", "r") as file: #wall: 020_029 ground:030_039
    for lines in file:
        line_count += 1
        if int(lines) == 2: #wall 2, 4
            floor_vertex_list.append(line_count) # index of vertices
# line_count = 0
# with open("./build copy/results1_0"+str(begin_scale)+"_0"+str(begin_scale+9)+".txt", "r") as file: #wall: 020_029 ground:030_039
#     for lines in file:
#         line_count += 1
#         if int(lines) == 4: #wall 2, 4
#             floor_vertex_list.append(line_count) # index of vertices
# line_count = 0
# with open("./build copy/results1_030_039.txt", "r") as file: #wall: 020_029 ground:030_039
#     for lines in file:
#         line_count += 1
#         if int(lines) == 0: #wall 2, 4
#             floor_vertex_list.append(line_count) # index of vertices
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
            # x = float(line_content[target_scale * 5 + 0])
            # y = float(line_content[target_scale * 5 + 1])
            # z = float(line_content[target_scale * 5 + 2])
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
print("avg:", xavg, yavg, zavg)

normals = np.asarray([xavg, yavg, zavg])

import numpy as np
import open3d as o3d

# Read point cloud:
# pcd = o3d.io.read_point_cloud("build copy/results1_030_039.ply")
pcd = o3d.io.read_point_cloud("build copy/rooms_full.ply")
# Create a 3D coordinate system:
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
# geometries to draw:
# geometries = [pcd, origin]
# Visualize:
# o3d.visualization.draw_geometries(geometries)


pcd_remove_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list], invert=True)
pcd_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list])
# Find the minimum and maximum coordinates along x and y
x_min, y_min, z_min = np.min(pcd_ground.points, axis=0)[:3]
x_max, y_max, z_max = np.max(pcd_ground.points, axis=0)[:3]
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=np.array([x_min, y_min, z_min]), max_bound=np.array([x_max, y_max, z_max]),
)
bbox.color = (1, 0, 0)
plane_model, inliers = pcd_ground.segment_plane(distance_threshold=0.1,
                                         ransac_n=3,
                                         num_iterations=1000)
obb = pcd_ground.get_oriented_bounding_box()
obb.color = (0, 0, 1)
corner_vertices = np.asarray(obb.get_box_points())
print("corner_vertices for obb: ", corner_vertices)
spheres = []
for i in range(len(corner_vertices)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    num_vertices = np.asarray(sphere.vertices).shape[0]
    colors = np.array([[1.0, 0.0, 0.0]] * num_vertices)
    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
    #target_position = np.asarray(pcd.points)[floor_vertex_list[0]]
    translation = np.eye(4)
    translation[:3, 3] = corner_vertices[i]
    sphere.transform(translation)
    spheres.append(sphere)

[a, b, c, d] = plane_model
print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")
# Create a plane from the estimated equation
# pcd_ground_plane = pcd_ground.select_by_index(inliers)
# pcd_ground_plane.paint_uniform_color([1.0, 0, 0])
#plane = o3d.geometry.TriangleMesh.create_from_plane_coefficients(a, b, c, d)

centroid = np.mean(np.asarray(pcd_ground.points), axis=0)
print(f"Centroid: {centroid}")

#plane = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=centroid)
# plane_center = np.array([0, 0, 0])  # Center of the plane
# plane_normal = np.array([0, 0, 1])  # Normal vector of the plane
# plane_width = 2.0  # Width of the plane
# plane_height = 2.0  # Height of the plane
# plane = o3d.geometry.TriangleMesh.create_box(width=plane_width, height=plane_height, depth=0.001)
# # Rotate the plane to the desired orientation
# R = plane.get_rotation_matrix_from_xyz(normals)
# plane.rotate(R, center=plane_center)
# # Translate the plane to the desired position
# plane.translate(centroid)

# Create a grid of points
x = np.linspace(-2, 2, 2)
y = np.linspace(-2, 2, 2)
X, Y = np.meshgrid(x, y)
Z = -(a*X + b*Y + d) / c

# Create a plane from the points
plane = o3d.geometry.PointCloud()
plane.points = o3d.utility.Vector3dVector(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))

triangles = np.array([
    [0, 1, 2],  # First triangle
    [3, 2, 1]   # Second triangle
])
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([pcd_remove_ground, mesh] + spheres)
# o3d.io.write_point_cloud("processed_room_ground.ply", pcd_remove_ground)
#o3d.io.write_point_cloud("wall_plane_02.ply", plane)
o3d.io.write_triangle_mesh("wall_plane_01.obj", mesh)
# o3d.io.write_point_cloud("./remove_plane.ply", pcd_remove_ground)