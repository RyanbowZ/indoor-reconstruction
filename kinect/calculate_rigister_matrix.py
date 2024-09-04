import numpy as np
import os
import trimesh
import open3d as o3d


objectname_list = ["banana","apple","cup","chair"]
object_idx = 3
object_name = objectname_list[object_idx]
scene_name = "annex"

suffix_list = ["obj","ply"]
suffix_idx = 0
suffix = suffix_list[suffix_idx]

model_path = f"/ghome/l6/yqliang/littleduck/datasets/{scene_name}/models/{object_name}_points.npy"
scan_path = f"/ghome/l6/yqliang/littleduck/datasets/{scene_name}/world_corrdinate/{object_name}_bbox/{object_name}_corners_mean.npy"
model_path_mesh = f"/ghome/l6/yqliang/littleduck/datasets/{scene_name}/models/{object_name}_model.{suffix}"
new_object_path = f"/ghome/l6/yqliang/littleduck/datasets/{scene_name}/models/{object_name}_rotated.obj"

def rotation_matrix(v1, v2):
    
    # 计算夹角
    cos_theta = np.dot(v1, v2)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # 计算旋转轴
    rotation_axis = np.cross(v1, v2)
    print(rotation_axis)
    rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
    
    # 构造旋转矩阵
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    return R


if __name__ == "__main__":
    ins = np.load(model_path)
    ins_mean = np.mean(ins, axis=0) # size 3
    print("ins_mean", ins_mean)
    reference_direction = (ins[0] - ins_mean) / np.linalg.norm(ins[0]-ins_mean)
    y_len = np.linalg.norm(ins[0] - ins[3])
    # print(ins[0] - ins[3])
    print("x_len",y_len)
    x_len = np.linalg.norm(ins[0] - ins[1])
    z_len = np.linalg.norm(ins[0] - ins[4])
    max_len = max(x_len, y_len, z_len)
    min_len = min(x_len, y_len, z_len)
    median_len = x_len + y_len + z_len - max_len - min_len
    outs = np.load(scan_path)[:8]
    outs_mean = np.mean(outs, axis=0) # size 3
    print("gt_mean", outs_mean)
    reference_direction_scan = (outs[0] - outs_mean) / np.linalg.norm(outs[0] - outs_mean)
    y_len_scan = np.linalg.norm(outs[0] - outs[3])
    x_len_scan = np.linalg.norm(outs[0] - outs[1])
    z_len_scan = np.linalg.norm(outs[0] - outs[4])
    max_len_scan = max(x_len_scan, y_len_scan, z_len_scan)
    min_len_scan = min(x_len_scan, y_len_scan, z_len_scan)
    median_len_scan = x_len_scan + y_len_scan + z_len_scan - max_len_scan - min_len_scan
    max_ratio = max_len_scan / max_len
    min_ratio = min_len_scan / min_len
    median_ratio = median_len_scan / median_len

    average_ratio = (max_ratio + min_ratio + median_ratio) / 3



    translation = outs_mean - ins_mean

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = -ins_mean
    fix_ratio = True
    if fix_ratio: 
        scale = np.array([average_ratio, average_ratio, average_ratio]) 
    else:
        scale = np.array([x_len_scan / x_len, y_len_scan / y_len, z_len_scan / z_len])
    reference_direction = reference_direction * scale
    reference_direction = reference_direction / np.linalg.norm(reference_direction)
    print('scale', scale)
    rotation = rotation_matrix( reference_direction, reference_direction_scan)
    four_by_four_matrix = np.eye(4)
    four_by_four_matrix[:3, :3] = np.diag(scale)
    # four_by_four_matrix[:3, :3] = rotation
    four_by_four_matrix[:3, 3] = outs_mean 
    # four_by_four_matrix[:3, 3] = 1 / scale * -ins_mean + translation 
    four_by_four_matrix[:3, :3] = np.matmul(rotation, four_by_four_matrix[:3, :3])

    transformation_matrix = np.matmul(four_by_four_matrix, transformation_matrix)

    print('four_by_four_matrix', transformation_matrix)
    print('ins_mean', ins_mean)



    ins_homo = np.ones((8,4))
    ins_homo [:,:3] = ins
    print(ins)
    rotated = np.matmul( transformation_matrix,ins_homo.T).T
    # print(rotated)
    rotated_coordinates = rotated[:,:3] / rotated[:,3:4]
    rotated_dir = rotated_coordinates[0]-np.mean(rotated_coordinates,axis=0)
    rotated_dir /= np.linalg.norm(rotated_dir)
    print('reference_direction', reference_direction)
    print('rotated_dir', rotated_dir)
    print('reference_direction_scan', reference_direction_scan)

    # verify
    rotated_mean = np.mean(rotated_coordinates,axis=0)
    print("matrix",transformation_matrix)
    print("rotated_mean", rotated_mean)
    print('outs_mean', outs_mean)

    outs_homo = np.ones((8,4))
    outs_homo[:,:3] = outs
    print(rotated_coordinates)
    print(outs)



    # print(model_bbox)
    # print(scan_bbox)
    # Calculate the rigister matrix


    if suffix == "obj":
        object_f = trimesh.load(model_path_mesh,process = False,maintain_order = True,force="mesh")
        vertices_no = object_f.vertices.shape[0]
    else:
        object_f = o3d.io.read_triangle_mesh(model_path_mesh)
        vertices_no = np.asarray(object_f.vertices).shape[0]

    vertices_homo = np.ones((vertices_no,4))
    vertices_homo[:,:3] = np.asarray(object_f.vertices)
    applied_transform = (np.matmul(transformation_matrix , vertices_homo.T)).T


    if suffix =="obj":
        object_f.vertices =  applied_transform[:,:3] / applied_transform[:,3:4]
        object_f.export(new_object_path)
        print("obj file is saved at",new_object_path)
    else:
        object_f.vertices = o3d.utility.Vector3dVector( applied_transform[:,:3] / applied_transform[:,3:4])
        o3d.io.write_triangle_mesh(new_object_path, object_f)
    