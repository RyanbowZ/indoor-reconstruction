import numpy as np
import open3d as o3d


def create_sphere(center, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def getLayoutFromBeginScale(begin_scale, label):
    line_count = 0
    floor_vertex_list = []

    with open("./build copy/results1_0" + str(begin_scale) + "_0" + str(begin_scale + 9) + ".txt",
              "r") as file:  #wall: 020_029 ground:030_039
        for lines in file:
            line_count += 1
            if int(lines) == label:
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

    xavg = xavg / avgcount
    yavg = yavg / avgcount
    zavg = zavg / avgcount
    print("normal avg:", xavg, yavg, zavg)

    normals = np.asarray([xavg, yavg, zavg])

    pcd = o3d.io.read_point_cloud("rooms_full.ply")
    pcd_remove_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list], invert=True)
    pcd_ground = pcd.select_by_index([num - 1 for num in floor_vertex_list])

    obb = pcd_ground.get_oriented_bounding_box()
    obb.color = (0, 0, 1)

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
        edge_vector = box_points[end] - box_points[start]
        edge_length = np.linalg.norm(edge_vector)
        edge_vector_normalized = edge_vector / edge_length

        # Calculate the angle between the edge and the up vector
        angle = np.arccos(np.dot(edge_vector_normalized, normals))
        angle_degrees = np.degrees(angle)

        # Store the edge index, angle, and length
        edge_angles.append((i, angle_degrees, edge_length, start, end))
    edge_angles.sort(key=lambda x: min(x[1], 180 - x[1]))
    print("height edges:", edge_angles[:4])

    layout2D_points = []

    for edge in edge_angles[:4]:
        layout2D_points.append(0.5 * (box_points[edge[3]] + box_points[edge[4]]))
    return layout2D_points


floorlayout = getLayoutFromBeginScale(30, 0)
wall1layout = getLayoutFromBeginScale(20, 2)
wall2layout = getLayoutFromBeginScale(20, 4)
spheres = []
# for i in range(len(corner_vertices)):
layout2D_points = floorlayout + wall1layout + wall2layout
for i in range(len(layout2D_points)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    num_vertices = np.asarray(sphere.vertices).shape[0]
    colors = np.array([[1, 0, 0]] * num_vertices)
    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
    translation = np.eye(4)
    translation[:3, 3] = layout2D_points[i]
    sphere.transform(translation)
    spheres.append(sphere)

sphere_points = []


for s in spheres:
    sphere_points.append(s.get_center())

#project the spheres on the plane
plane_normal = np.array([0.47536998974646244, 0.1407673607465304, 0.8684514511456034])
plane_point = np.array([0.40798303, 0.00871838, -0.66888969])
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(sphere_points)
points_array = np.asarray(pcd2.points)
d = -np.dot(plane_normal, plane_point)
distances = np.dot(points_array, plane_normal) + d - 1
projected_points = points_array - np.outer(distances, plane_normal)

apple_npy = np.load('apple_corners_mean.npy')
banana_npy = np.load('banana_corners_mean.npy')
cup_npy = np.load('cup_corners_mean.npy')
apple_distance = np.dot(apple_npy[:8], plane_normal) + d - 1
apple_points = apple_npy[:8] - np.outer(apple_distance, plane_normal)
banana_distance = np.dot(banana_npy[:8], plane_normal) + d - 1
banana_points = banana_npy[:8] - np.outer(banana_distance, plane_normal)
cup_distance = np.dot(cup_npy[:8], plane_normal) + d - 1
cup_points = cup_npy[:8] - np.outer(cup_distance, plane_normal)

projected_spheres = []
for p in projected_points:
    projected_sphere = create_sphere(p, 0.1, [0, 0, 1])
    projected_spheres.append(projected_sphere)

for p in apple_points:
    apple_sphere = create_sphere(p, 0.05, [0, 1, 1])
    projected_spheres.append(apple_sphere)

for p in banana_points:
    banana_sphere = create_sphere(p, 0.05, [1, 1, 0])
    projected_spheres.append(banana_sphere)

for p in cup_points:
    cup_sphere = create_sphere(p, 0.05, [1, 0, 1])
    projected_spheres.append(cup_sphere)

pcd = o3d.io.read_point_cloud("rooms_full.ply")
o3d.visualization.draw_geometries([pcd] + projected_spheres)

