import trimesh
import numpy as np
import open3d as o3d
import vedo
import vedo.file_io
from trimesh import transformations, primitives, util
import os
# import scipy
from sklearn import datasets, decomposition

# floor_normal = np.array([0.39913959343393945, 0.2912914540947337, 0.730937833055142]) # banana
floor_normal = np.array([0.35143699525195327, 0.10679395759570112, 0.9301005800392375]) # evans
on_plane_pt = np.array([1.4529548, 0.68345192, -1.90888154]) # evans



object_idx = 2

objectname_list = ["banana","apple","cup"]
iteration_times_list = [1,1,2]

objectname = objectname_list[object_idx]
iteration_times = iteration_times_list[object_idx]


def build_spheres(pt: np.array, radius, color):
    if len(pt.shape) > 1:
        meshes = []
        for p in pt:
            meshes.append(build_spheres(p, radius, color))
        return util.concatenate(meshes)

    sphere = primitives.Sphere(radius=radius, center=pt)
    sphere.visual.vertex_colors = color

    return sphere


def build_cylinders(pt0: np.array, pt1: np.array, radius, color):
    
    if len(pt0.shape) > 1:
        meshes = []
        for p0, p1 in zip(pt0, pt1):
            meshes.append(build_cylinders(p0, p1, radius, color))
        return util.concatenate(meshes)

    h = np.linalg.norm(pt0 - pt1)
    stick = primitives.Cylinder(radius=radius, height=h, sections=6)
    stick.visual.vertex_colors = color

    normal = pt0 - pt1
    normal = normal / np.linalg.norm(normal)
    rot_axis = np.cross(stick.direction, normal)
    rot_angle = np.arccos(np.dot(stick.direction, normal))
    rot_mat = transformations.rotation_matrix(rot_angle, rot_axis, (0, 0, 0))
    trans_mat1 = transformations.translation_matrix((0, 0, h / 2))
    trans_mat2 = transformations.translation_matrix(pt1)
    transform_mat = trans_mat2 @ rot_mat @ trans_mat1
    stick.apply_transform(transform_mat)
    
    return stick

top_vertex_index = 11589
bottom_vertex_index = 24788
reference_mesh_name = "12.ply"




mesh_path = "/ghome/l6/yqliang/littleduck/datasets/fruits/world_corrdinate/" + objectname
preprocess_path = "/ghome/l6/yqliang/littleduck/datasets/fruits/world_corrdinate/" + objectname +"_preprocess"
projected_path = "/ghome/l6/yqliang/littleduck/datasets/fruits/world_corrdinate/" + objectname + "_projected"
filter_first_path ="/ghome/l6/yqliang/littleduck/datasets/fruits/world_corrdinate/" + objectname + "_filtered"
save_path = "/ghome/l6/yqliang/littleduck/datasets/fruits/world_corrdinate/" + objectname + "_bbox"



def preprocess_and_save():
    os.makedirs(preprocess_path,exist_ok=True)
    file_list = os.listdir(mesh_path)
    for name in file_list:
        if name[-4:] != ".ply":
            continue
        mesh = trimesh.load_mesh(
            os.path.join(mesh_path, name),
            maintain_order=True,
            preprocess=False
        )
        mesh = mesh_pre_process(mesh)
        if mesh.vertices.shape[0] < 100:
            continue
        mesh.export(os.path.join(preprocess_path,name))


def mesh_pre_process(mesh):
    mean = np.mean(mesh.vertices, axis=0)
    dis =  np.linalg.norm(mesh.vertices - mean, axis=-1)
    std = np.std(dis, axis=0)
    mu = np.mean(dis, axis=0)

    v_mask = np.ones(mesh.vertices.shape[0], dtype=bool)
    v_mask[dis > 3 * std + mu] = False
    v_mask[dis < -3 * std + mu] = False
    # remove the None value
    v = mesh.vertices[v_mask]
    new_pcd = trimesh.PointCloud(v)
    # mesh.update_vertices(mask=v_mask)
    # mesh.remove_unreferenced_vertices
    new_pcd.export("preprocess_object.ply")
    return new_pcd

def project_to_the_floor_and_save():
    os.makedirs(projected_path,exist_ok=True)
    file_list = os.listdir(preprocess_path)
    for name in file_list:
        if name[-4:] != ".ply":
            continue
        mesh = trimesh.load_mesh(
            os.path.join(preprocess_path, name),
            maintain_order=True,
            preprocess=False
        )
        floor_normal = trimesh.load_mesh("/ghome/l6/yqliang/littleduck/kinect/normal.obj",maintain_order=True,
        preprocess=False)
        on_plane_pts = project_to_the_floor(mesh, floor_normal)
        trimesh.PointCloud(on_plane_pts).export(os.path.join(projected_path,name))


def calibra_floor(object_mesh):
    # Compute the normal of the floor
    object_vector =  object_mesh.vertices[top_vertex_index] - object_mesh.vertices[bottom_vertex_index]
    object_normal = object_vector / np.linalg.norm(object_vector)
    normal_pts = np.array([object_mesh.vertices[bottom_vertex_index], object_mesh.vertices[bottom_vertex_index] + object_normal])

    # render the normal
    line_mesh = vedo.Line(normal_pts, c='r')
    line_trimesh = vedo.vedo2trimesh(line_mesh)
    line_trimesh.export("normal.obj")
    return object_normal


def project_to_the_floor(object_mesh, object_normal,reference = False):
    # Project the object mesh to the floor
    bottom_pt = object_normal.vertices[0]
    top_pt = object_normal.vertices[1]
    normal = top_pt - bottom_pt

    collected = []
    for v in object_mesh.vertices:
        vec_1 = v - bottom_pt
        distance = np.dot(vec_1, normal)
        projected = vec_1 - distance * normal
        translated_projected = projected + bottom_pt
        collected.append(translated_projected)
    if reference:
        trimesh.PointCloud(collected).export("reference_projected_object.ply")

    return collected
    

def pca_axes(on_plane_pts):
    on_plane_pts = np.array(on_plane_pts)
    pca = decomposition.PCA(n_components=2)
    pca.fit(on_plane_pts)
    coordinate = pca.transform(on_plane_pts)
    x_min, x_max = coordinate[:, 0].min(), coordinate[:, 0].max()
    x_m = x_min * pca.components_[0] + pca.mean_
    x_M = x_max * pca.components_[0] + pca.mean_   
    y_min, y_max = coordinate[:, 1].min(), coordinate[:, 1].max()
    y_m = y_min * pca.components_[1] + pca.mean_
    y_M = y_max * pca.components_[1] + pca.mean_
    # ones = np.ones(on_axis.shape[0])
    # on_axis = np.column_stack((on_axis, ones))

    margin_pts = [x_min, x_max, y_min, y_max]
    return pca,margin_pts
    trimesh.PointCloud(pts).export("on_axes.ply")
    # collected = pca.transform(collected)
    # print(axes)

def pca_filter(pca_model,margin_pts,on_plane_pts,name):
    coorinadats = pca_model.transform(on_plane_pts)
    on_plane_pts = np.array(on_plane_pts)
    mask = np.ones(coorinadats.shape[0], dtype=bool)
    mask[coorinadats[:, 0] < margin_pts[0]] = False
    mask[coorinadats[:, 0] > margin_pts[1]] = False
    mask[coorinadats[:, 1] < margin_pts[2]] = False
    mask[coorinadats[:, 1] > margin_pts[3]] = False
    selected_pts = on_plane_pts[mask]
    return selected_pts

def on_plane_bounding_box(file_path):
    file_name = os.listdir(file_path)
    center_sum = np.zeros(3)
    first_component_sum = np.zeros(3)
    second_component_sum = np.zeros(3)
    pts_no = 0
    file = file_name[0]
    on_plane_pts = trimesh.load_mesh(os.path.join(file_path,file),maintain_order=True,preprocess=False)
    pca, margin_pts = pca_axes(on_plane_pts.vertices)
    x_max_sum = 0
    y_max_sum = 0
    x_min_sum = 0
    y_min_sum = 0

    for file in file_name:
        pts_no += 1 
        on_plane_pts = trimesh.load_mesh(os.path.join(file_path,file),maintain_order=True,preprocess=False)
        pca, margin_pts = pca_axes(on_plane_pts.vertices)
        center_sum += pca.mean_
        first_component_sum += pca.components_[0]

        coorinadats = pca.transform(on_plane_pts.vertices)
        x_max_sum += coorinadats[:, 0].max() 
        y_max_sum += coorinadats[:, 1].max() 
        x_min_sum += coorinadats[:, 0].min() 
        y_min_sum += coorinadats[:, 1].min() 
        second_component_sum += pca.components_[1]
    pca.mean_ = center_sum / pts_no
    pca.components_[0] = first_component_sum / pts_no
    pca.components_[1] = second_component_sum / pts_no
    x_max_v = x_max_sum / pts_no * 2
    y_max_v = y_max_sum / pts_no * 2
    x_min_v = x_min_sum / pts_no * 2
    y_min_v = y_min_sum / pts_no * 2
    print("center mean is " + str(pca.mean_))
    print("first component is " + str(pca.components_[0]))
    print("second component is " + str(pca.components_[1]))
    # x_max_v, y_max_v, x_min_v, y_min_v = x_max_v, x_max_v, -x_max_v, -x_max_v
    x_max = pca.mean_ + pca.components_[0] * x_max_v
    y_max = pca.mean_ + pca.components_[1] * y_max_v
    x_min = pca.mean_ + pca.components_[0] * x_min_v
    y_min = pca.mean_ + pca.components_[1] * y_min_v
    print(x_max_v, y_max_v, x_min_v, y_min_v)
    # corners = np.array([x_max, y_max, x_min, y_min,pca.mean_])
    corners = np.array([x_max, y_max, x_min, y_min])

    # print(corners)
    build_spheres(corners, 0.01, (0, 200, 0)).export("corners1.ply")
    return pca, margin_pts

    x_max_sum = np.zeros(3)
    y_max_sum = np.zeros(3)

    for file in file_name:
        on_plane_pts = trimesh.load_mesh(os.path.join(file_path,file),maintain_order=True,preprocess=False)
        coorinadats = pca.transform(on_plane_pts)
        x_max_sum += coorinadats[:, 0].max() * pca.components_[0]
        y_max_sum += coorinadats[:, 1].max() * pca.components_[1]
        on_plane_pts = np.array(on_plane_pts)
        selected_pts = pca_filter(pca,margin_pts, on_plane_pts.vertices, file)
        trimesh.PointCloud(selected_pts).export(os.path.join(filter_first_path,file))



def on_plane_bounding_box2(file_path):
    file_name = os.listdir(file_path)
    center_sum = np.zeros(3)
    first_component_sum = np.zeros(3)
    second_component_sum = np.zeros(3)
    pts_no = 0
    file = file_name[0]
    on_plane_pts = trimesh.load_mesh(os.path.join(file_path,file),maintain_order=True,preprocess=False)
    pca, margin_pts = pca_axes(on_plane_pts.vertices)
    x_max_sum = 0
    y_max_sum = 0
    x_min_sum = 0
    y_min_sum = 0
    
    vs = []
    for file in file_name:
        pts_no += 1 
        on_plane_pts = trimesh.load_mesh(os.path.join(file_path,file),maintain_order=True,preprocess=False)
        vs.append(on_plane_pts.vertices)
    
    vs = np.concatenate(vs)
    pca, margin_pts = pca_axes(vs)
    # center_sum = pca.mean_
    # first_component_sum = pca.components_[0]

    coorinadats = pca.transform(vs)
    x_max_sum += coorinadats[:, 0].max() 
    y_max_sum += coorinadats[:, 1].max() 
    x_min_sum += coorinadats[:, 0].min() 
    y_min_sum += coorinadats[:, 1].min() 
    # second_component_sum += pca.components_[1]
    # pca.mean_ = center_sum / pts_no
    # pca.components_[0] = first_component_sum / pts_no
    # pca.components_[1] = second_component_sum / pts_no
    # x_max_v = x_max_sum / pts_no * 2
    # y_max_v = y_max_sum / pts_no * 2
    # x_min_v = x_min_sum / pts_no * 2
    # y_min_v = y_min_sum / pts_no * 2
    print("center mean is " + str(pca.mean_))
    print("first component is " + str(pca.components_[0]))
    print("second component is " + str(pca.components_[1]))
    x_max_v, y_max_v, x_min_v, y_min_v = x_max_sum, y_max_sum, x_min_sum, y_min_sum
    # x_max = pca.mean_ + pca.components_[0] * x_max_v
    # y_max = pca.mean_ + pca.components_[1] * y_max_v
    # x_min = pca.mean_ + pca.components_[0] * x_min_v
    # y_min = pca.mean_ + pca.components_[1] * y_min_v
    x_max = pca.mean_ + pca.components_[0] * x_max_v + pca.components_[1] * y_max_v
    y_max = pca.mean_ + pca.components_[0] * x_max_v + pca.components_[1] * y_min_v 
    y_min = pca.mean_ + pca.components_[0] * x_min_v + pca.components_[1] * y_max_v
    x_min = pca.mean_ + pca.components_[0] * x_min_v + pca.components_[1] * y_min_v
    print(x_max_v, y_max_v, x_min_v, y_min_v)
    # corners = np.array([x_max, y_max, x_min, y_min,pca.mean_])
    corners = np.array([x_max, y_max, x_min, y_min])
    print("corners:",corners)

    # print(corners)
    down_spheres = build_spheres(corners, 0.01, (100, 200, 0))
    return corners, pca.mean_, down_spheres





if __name__ == "__main__":
    already_preprocess = True
    if not already_preprocess:
        preprocess_and_save()
    
    already_project = True
    if not already_project:
        project_to_the_floor_and_save()
    
    if iteration_times == 1:
        first_is_enough = True
    else:
        first_is_enough = False

    pre_filter_path = projected_path
    post_filter_path = filter_first_path

    if not first_is_enough:
        pre_filter_path = filter_first_path
        post_filter_path = filter_first_path + "_2"
        reference_mesh_name = "20.ply"

    # empth the folder
    os.system("rm -rf " + post_filter_path)
    os.makedirs(post_filter_path,exist_ok=True)

    file_name = os.listdir(pre_filter_path)
    # for file in file_name[::10]:
    for i in [1]:
        file = reference_mesh_name
        print(file + " is processing")
        on_plane_pts = trimesh.load_mesh(os.path.join(pre_filter_path,file),maintain_order=True,preprocess=False)
        pca, margin_pts = pca_axes(on_plane_pts.vertices)
        center_mean = on_plane_pts.vertices.mean(axis=0)
        print("center mean is " + str(center_mean))
        reliable_region = np.linalg.norm(on_plane_pts.vertices - center_mean, axis=-1)
        reliable_region_mean = np.mean(reliable_region)
        print("reliable region is " + str(reliable_region_mean)  )
        retain_ratio = 0.6
        for cropped_file in file_name:
            if cropped_file == file:
                continue

            print(cropped_file + " is cropping")
            cropped_plane_pts = trimesh.load_mesh(os.path.join(pre_filter_path,cropped_file),maintain_order=True,preprocess=False)
            no_pre_cropped_pts = cropped_plane_pts.vertices.shape[0]
            cropped_center_mean = cropped_plane_pts.vertices.mean(axis=0)
            distance_to_reference = np.linalg.norm(cropped_center_mean - center_mean , axis=-1)
            print("distance to reference is " + str(distance_to_reference))

            if distance_to_reference > retain_ratio * reliable_region_mean:
                continue
            selected_pts = pca_filter(pca,margin_pts, cropped_plane_pts.vertices, cropped_file)
            if len(selected_pts) < 100:
                continue
            no_post_cropped_pts = selected_pts.shape[0]
            if no_post_cropped_pts < 0.3 * no_pre_cropped_pts:
                continue
            trimesh.PointCloud(selected_pts).export(os.path.join(post_filter_path,cropped_file))

    corners, mean, down_spheres = on_plane_bounding_box2(filter_first_path)
    on_plane_pt = corners[3]

    # determine the height
    # global floor_normal
    floor_normal_unit = floor_normal / np.linalg.norm(floor_normal)
    files_name = os.listdir(preprocess_path)
    # files_name = files_name[0:1]
    print(files_name)
    max_list = np.array([])
    for file in files_name:
        on_plane_pts = trimesh.load_mesh(os.path.join(preprocess_path,file),maintain_order=True,preprocess=False)
        vec = on_plane_pts.vertices - on_plane_pt
        height = np.dot(vec, floor_normal_unit)
        max_val =np.quantile(height, 0.75, method='closest_observation')
        max_list = np.append(max_list, max_val)
        print(file, "height is " + str(max_val))

    max = np.median(max_list)
    print("max is " + str(max)) 
    up_corners = corners + max * floor_normal_unit
    up_four_spheres = build_spheres(up_corners, 0.01, (0, 0, 200))
    up_2_down_connected_lines = build_cylinders(corners, up_corners, 0.001, (0, 0, 200))
    for i in range(4):
        up_2_down_connected_lines += build_cylinders(corners[i], corners[(i+1)%4], 0.001, (0, 0, 200))
        up_2_down_connected_lines += build_cylinders(up_corners[i], up_corners[(i+1)%4], 0.001, (0, 0, 200))
    # x_max, y_max, x_min, y_min
    all_to_draw = up_2_down_connected_lines  + down_spheres + up_four_spheres
    # if os.path.exists( save_path):
    #     os.system("rm -rf " + save_path)
    os.makedirs(save_path,exist_ok=True)
    bbox_file_path = os.path.join(save_path, f"{objectname}_bbox.ply")
    for i in range(4):
        build_spheres(corners[i], 0.01, (0, 0, 0)).export(os.path.join(save_path, f"corners{i}.ply"))
        build_spheres(up_corners[i], 0.01, (0, 0, 0)).export(os.path.join(save_path, f"up_corners{i}.ply"))

    all_to_draw.export(bbox_file_path)
    eight_pts = np.array([corners[0], corners[1], corners[2], corners[3], up_corners[0], up_corners[1], up_corners[2], up_corners[3]])
    color_list = np.array([(0, 0, 0), (0, 50, 0), (0, 100, 0), (0, 150, 0), (0, 200, 0), (0, 250, 0), (0, 300, 0), (0, 350, 0)])
    # debug only: export the corners
    # for color_r in range(8):
    #     build_spheres(eight_pts[color_r], 0.01, (0,0,0)).export("corners" + str(color_r) + ".ply")
    #     print(eight_pts[color_r])
    pts_and_mean = np.append(eight_pts, [mean], axis=0)
    coor_path = os.path.join(save_path, f"{objectname}_corners_mean.npy")
    np.save(coor_path, pts_and_mean)
    print("corner and mean are saved at: " + bbox_file_path + " and " + coor_path)

    # remove corners to the origin by subtracting the mean
    corners = corners - mean


    
    
    







