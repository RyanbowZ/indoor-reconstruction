import os
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
path = "/home/hazel/Sources/SplaTAM/experiments/iPhone_Captures/240926124017/SplaTAM_iPhone/eval/pcd"
labels = ["box_1", "box_2", "chair_1","box_3","sofa_1","bottom","window","curtain_down","curtain"]
on_floor_pts = []
on_floor_pts.append([0.20664,4.00292,2.56811])
on_floor_pts.append([1.17476 ,1.4997, 0.781315])
on_floor_pts.append([0.0455656, 2.29834, 2.84025])


if __name__ == "__main__":

    p1 = np.array(on_floor_pts[0])
    p2 = np.array(on_floor_pts[1])
    p3 = np.array(on_floor_pts[2])

    
    # calculate normal
    vec1 = p2 - p1
    vec2 = p3 - p1
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(normal)


    # construct a rotaion matrix to align the normal with the z axis
    # determine the new xyz axis
    # origin is p1
    translation_matrix = np.array([[1, 0, 0, -p1[0]],
                                    [0, 1, 0, -p1[1]],
                                    [0, 0, 1, -p1[2]],
                                    [0, 0, 0, 1]])
    z = normal / np.linalg.norm(normal)
    x = vec1 / np.linalg.norm(vec1)
    y = np.cross(z, x) / np.linalg.norm(np.cross(z, x))
    rotation_matrix = np.stack([x, y, z], axis=0)
    rotation_matrix = np.vstack([rotation_matrix, [0, 0, 0]])
    rotation_matrix = np.hstack([rotation_matrix, [[0], [0], [0], [1]]])
    transformation_matrix = np.matmul(rotation_matrix, translation_matrix)


    dir_names= os.listdir(path)
    centroids = []

    for dir_name in sorted(dir_names):
        if not os.path.isdir(os.path.join(path, dir_name)):
            continue
        sum = 0
        count = 0
        pcd_files = os.listdir(os.path.join(path, dir_name))
        for pcd_file in pcd_files:
            if pcd_file.endswith(".ply"):
                pcd = o3d.io.read_point_cloud(os.path.join(path, dir_name, pcd_file))
                count += pcd.get_center()
                sum += 1
        centroids.append(count/sum)
        print(f"Centroid for {dir_name}: {count/sum}")


    projected =[]
    for i in range(len(centroids)):
        vec3 = centroids[i] - p1
        vertical = vec3.dot(z) * z
        horizontal = vec3 - vertical
        projected.append(horizontal+ p1)


    p_transformed = []
    for point in projected:
        p_transformed.append(transformation_matrix.dot(np.append(point, 1)))
    p_transformed = np.array(p_transformed)
    p_transformed = p_transformed[:,:2]

    # plot 2d scatter of p_transformed with labels
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # x and y axis the same scale
    ax.set_aspect('equal', adjustable='box')
    for i in range(len(p_transformed)):
        ax.scatter(p_transformed[i][0], p_transformed[i][1], color='blue')
        ax.text(p_transformed[i][0], p_transformed[i][1], labels[i])

    plt.show(block=True)


    # build a point cloud from the mean value
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centroids+projected)
    # o3d.visualization.draw_geometries([pcd])
    cen_path = os.path.join(path, "centroids.ply")
    o3d.io.write_point_cloud(cen_path, pcd)

                