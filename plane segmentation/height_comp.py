import numpy as np
import open3d as o3d

line_count = 0
floor_vertex_list = []
begin_scale = 30

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

pcd = o3d.io.read_point_cloud("rooms_full.ply")
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
geometries = [pcd, origin]

pcd_remove_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list], invert=True)
pcd_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list])

obb = pcd_ground.get_oriented_bounding_box()
obb.color = (0, 0, 1)
corner_vertices = np.asarray(obb.get_box_points())
R_obb = obb.R
print("obb.R", R_obb)

box_points = np.asarray(obb.get_box_points())

# Define the edges of the bounding box
edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 6), (1, 7),
    (2, 5), (2, 7),
    (3, 5), (3, 6),
    (4, 5), (4, 6), (4, 7)
]

# Calculate and print the length of each edge
edge_angles = []
for i, (start, end) in enumerate(edges):
    # length = np.linalg.norm(box_points[end] - box_points[start])
    # print(f"Edge {i+1} length: {length:.4f}")
    edge_vector = box_points[end] - box_points[start]
    edge_length = np.linalg.norm(edge_vector)

    # Normalize the edge vector
    edge_vector_normalized = edge_vector / edge_length

    # Calculate the angle between the edge and the up vector
    angle = np.arccos(np.dot(edge_vector_normalized, normals))
    angle_degrees = np.degrees(angle)

    # Store the edge index, angle, and length
    edge_angles.append((i, angle_degrees, edge_length, start, end))
edge_angles.sort(key=lambda x: min(x[1], 180 - x[1]))
print("height edges:", edge_angles[:4])

for edge in edge_angles:
    print("The "+str(edge[0])+"th edge, angle with normal direction: "+ str(edge[1])+", length: "+str(edge[2]))

layout2D_points=[]

for edge in edge_angles[:4]:
    if edge[1] > 120:
        layout2D_points.append(edge[3])
    else:
        layout2D_points.append(edge[4])


spheres = []
# for i in range(len(corner_vertices)):
for i in layout2D_points:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    num_vertices = np.asarray(sphere.vertices).shape[0]
    colors = np.array([[1, 0, 0]] * num_vertices)
    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
    translation = np.eye(4)
    translation[:3, 3] = corner_vertices[i]
    sphere.transform(translation)
    spheres.append(sphere)

o3d.visualization.draw_geometries([pcd, origin, obb] + spheres)
