import numpy as np
line_count = 0
floor_vertex_list = []

begin_scale = 30

def align_object_to_plane(obj, plane_normal):
    # Normalize the plane normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    # Calculate the rotation axis (cross product of y-axis and plane normal)
    rotation_axis = np.cross([0, 1, 0], plane_normal)
    # Calculate the rotation angle
    rotation_angle = np.arccos(np.dot([0, 1, 0], plane_normal))
    # Create the rotation matrix
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    # Apply the rotation to the object
    obj.rotate(rotation_matrix, center=(0, 0, 0))

    return obj

with open("./build copy/results1_0"+str(begin_scale)+"_0"+str(begin_scale+9)+".txt", "r") as file: #wall: 020_029 ground:030_039
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


pcd = o3d.io.read_point_cloud("rooms_full.ply")
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
geometries = [pcd, origin]

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
R_obb = obb.R
print("obb.R", R_obb)

box_points = np.asarray(obb.get_box_points())

# Define the edges of the bounding box
edges = [
    (0, 1), (0, 2), (0, 4),  # edges from vertex 0
    (1, 3), (1, 5),          # edges from vertex 1
    (2, 3), (2, 6),          # edges from vertex 2
    (3, 7),                  # edge from vertex 3
    (4, 5), (4, 6),          # edges from vertex 4
    (5, 7),                  # edge from vertex 5
    (6, 7)                   # edge from vertex 6
]

# Calculate and print the length of each edge
for i, (start, end) in enumerate(edges):
    length = np.linalg.norm(box_points[end] - box_points[start])
    print(f"Edge {i+1} length: {length:.4f}")

spheres = []
for i in range(len(corner_vertices)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    num_vertices = np.asarray(sphere.vertices).shape[0]
    colors = np.array([[1.0, 0.0, 0.0]] * num_vertices)
    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
    translation = np.eye(4)
    translation[:3, 3] = corner_vertices[i]
    sphere.transform(translation)
    spheres.append(sphere)

[a, b, c, d] = plane_model
print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

centroid = np.mean(np.asarray(pcd_ground.points), axis=0)
print(f"Centroid: {centroid}")

# Create a grid of points
x = np.linspace(-2, 2, 2)
y = np.linspace(-2, 2, 2)
X, Y = np.meshgrid(x, y)
Z = -(a*X + b*Y + d) / c

# Create a plane from the points
plane = o3d.geometry.PointCloud()
plane.points = o3d.utility.Vector3dVector(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))

triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))
mesh.triangles = o3d.utility.Vector3iVector(triangles)

cup = o3d.io.read_point_cloud("39-cup/Cup.ply")

bbox_cup = cup.get_axis_aligned_bounding_box()
half_extents = bbox_cup.get_half_extent()
longest_axis = max(half_extents)
scale = 0.1 / (2.0 * longest_axis)
# Scale and center the mesh
cup.scale(scale, center=cup.get_center())
cup.translate(-cup.get_center())  # Move the mesh to the origin
# align_object_to_plane(cup, [xavg, yavg, zavg])
cup.rotate(R_obb, center=(0, 0, 0))
x_axis = R_obb[:, 0]  # The third column of R_obb is the new x-axis
rotation_vector = x_axis * -np.pi/2  # 90 degrees in radians
R_x = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
cup.rotate(R_x, center=(0, 0, 0))


o3d.visualization.draw_geometries([pcd_remove_ground, obb, cup, mesh] + spheres)
