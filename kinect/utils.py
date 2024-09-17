import os
from pathlib import Path
from tqdm import trange,tqdm
import shutil
import argparse
import numpy as np
from view_camera_in_the_scene import read_pose, from_quaernion_and_loc_to_4_by_4_matrix
import cv2
import pathlib
import open3d as o3d
from tools_visualization import *



def img_keep_every_k(in_dir, out_dir, k = 3):
    # keep every k frame; by default, keep every 3 frames
    # images should end with .png or .jpg
    # in_dir: the directory containing the images
    # out_dir: the directory to save the images
    # the first frame will be saved
    # for example, pk+q in total, the last q frames will be discarded, where p is the quotient and q is the remainder
    if os.path.exists(out_dir):
        print(f"{out_dir} already exists, skip img_keep_every_k.")
        return
    os.makedirs(out_dir, exist_ok=True)
    pngimgs = list(Path(in_dir).glob("*.png"))
    jpgimgs = list(Path(in_dir).glob("*.jpg"))
    imgs = pngimgs.extend(jpgimgs)
    imgs = sorted(imgs,key  = lambda x: float(x.stem))
    for i in range(0, len(imgs), k):
        shutil.copy(imgs[i], out_dir+ f"/{i//k}."+imgs[i].suffix)

def pose_txt_to_npy(in_dir):
    # convert pose txt to npy, save it, and return the npy
    # if the npy file already exists, read it, and return it
    npy_path = os.path.join(in_dir, f"estimate_c2w.npy")
    if os.path.exists(npy_path):
        # print("npy file already exists at", npy_path)
        pose = np.load(npy_path)
        N = len(pose)
        return pose, N
    # last frame is the txt file name with the largest number
    last_frame = max([int(file.stem.split("_")[-1]) for file in Path(in_dir).glob("*.txt")])
    txt_path = os.path.join(in_dir, f"traj_{last_frame}.txt")
    pose_l, pose_r = read_pose(txt_path,filter_ratio = 1)
    N  = last_frame + 1
    estimate_c2w = [from_quaernion_and_loc_to_4_by_4_matrix(pose_r[i],pose_l[i]) for i in range(N)]
    np.save(npy_path, estimate_c2w)
    # print("cvt pose_txt to npy. # of frames:",len(estimate_c2w))
    # print("saved at:", npy_path)
    return estimate_c2w, N

def img_cvt_filename_timestep_to_sequential(in_dir, out_dir, start = 0):
    # convert the filename from timestep to sequential
    # for example, 0.png, 1.png, 2.png, 3.png, 4.png, 5.png
    # by default, start from 0
    os.makedirs(out_dir, exist_ok=True)
    filename_original = []
    suffix = [".png", ".jpg"]
    for file in os.listdir(in_dir):
        if file.endswith(tuple(suffix)):
            filename_original.append(file)
    sorted_file = sorted(filename_original, key=lambda x: float(x[:-4]))
    for i, file in enumerate(sorted_file):
        shutil.copy(os.path.join(in_dir, file), os.path.join(out_dir, f"{i+start}."+file.split(".")[-1]))
    # print(f"cvt filename to sequential int number, start at {start}, saved in {out_dir}. ")

def depth_erosion(in_dir, out_dir, threshold = 500,iteration = 1,kernel_size = 3):
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(in_dir):
        depth_path = os.path.join(in_dir, filename)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        indicate_image_shape = (depth_img.shape[0], depth_img.shape[1])
        indicate_image = np.zeros(indicate_image_shape).astype(np.uint8)
        indicate_image[ depth_img > threshold ] = 1

        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
        # after erosion, the area with value 1 will be smaller
        img_reliable = cv2.erode(indicate_image, kernel, iterations=iteration) 
        img_irreliable = 1 - img_reliable
        depth_img[img_irreliable.astype(bool)] = 0
        cv2.imwrite(os.path.join(out_dir, filename), depth_img)

def depth_bilateral_filter(in_dir, out_dir, d=9, sigmaColor=75, sigmaSpace=75):
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(in_dir):
        depth_path = os.path.join(in_dir, filename)
        color_path = os.path.join(in_dir, filename).replace("depth", "rgb")
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        color_img = cv2.imread(color_path)
        # cvt to one channel
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        color_edge = cv2.Canny(color_img.astype(np.uint8), 100, 200)
        kernal = np.ones((3,3), np.uint8)
        color_edge = cv2.dilate(color_edge, kernel= kernal, iterations=1)
        depth_img[color_edge == 255] = 0
        zero_mask = depth_img == 0
        depth_float = depth_img.astype(np.float32) 
        # reference is color_img as the joint image
        depth_processed = cv2.ximgproc.jointBilateralFilter(color_img, depth_float, d, sigmaColor, sigmaSpace)
         # depth_processed = cv2.bilateralFilter(depth_float, d, sigmaColor, sigmaSpace)
        depth_img = depth_processed.astype(np.uint16)
        depth_img[zero_mask] = 0
        cv2.imwrite(os.path.join(out_dir, filename), depth_img)

def read_npy_in_folder(in_dir):
    in_dir = Path(in_dir)
    debug_dir = Path("debug")   
    os.makedirs(debug_dir, exist_ok=True)
    for file in os.listdir(in_dir):
        if file.endswith(".npy"):

            img_seg = np.load(in_dir / file)
            ones = np.ones_like(img_seg)
            mask = img_seg == ones *5
            cv2.imwrite(str(debug_dir / f"{file[:-4]}.png"), mask.astype(np.uint8) * 255)

def K_npy_to_txt(in_dir,out_dir):
    color_k, depth_k = np.load(in_dir)
    np.savetxt(str(out_dir), color_k, fmt="%f")

def make_subfolders(in_dir, n ,start_from = 1):
    in_dir = Path(in_dir)
    for i in range(start_from, n+1):
        os.makedirs(in_dir/str(i), exist_ok=True)
    return n

def remove_empty_folder(in_dir):    
    in_dir = Path(in_dir)
    for folder in in_dir.iterdir():
        if folder.is_dir():
            if not list(folder.glob("*")):
                # print(f"remove {folder}")
                shutil.rmtree(folder)

def split_2d_seg_into_folder( in_dir):
    in_dir = Path(in_dir)
    # for id,content in enumerate(id_list):
    #     os.makedirs(in_dir/str(id), exist_ok=True)


    n = 20
    make_subfolders(in_dir, n)

    # note: 0 is the background
    npy_files = in_dir.glob("*.npy")
    sorted_file = sorted(npy_files, key = lambda x: str(x.stem))
    has_writed = np.zeros(n)
    
    img_id = 0
    for file in sorted_file:
        img_seg = np.load( file)
        for i in range(1, n+1):
            mask = img_seg == np.ones_like(img_seg) * i
            if mask.any():
                cv2.imwrite(str(in_dir /str(i) /f"{img_id}.png"), mask.astype(np.uint8) * 255)
                has_writed[i] = 1
        img_id += 1
    
    remove_empty_folder(in_dir)
    print("# of valid folders", np.sum(has_writed))
    return np.sum(has_writed)

def split_3d_seg_into_folder(color_dir,depth_dir,mask_dir,k_dir, out_dir):
    mask_f = os.listdir(mask_dir)
    # mask_f remain folders remove others
    mask_f = [f for f in mask_f if os.path.isdir(os.path.join(mask_dir, f))]
    n = len(mask_f)
    color_f = os.listdir(color_dir)
    img_num = len(color_f)
    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)
    k_path = os.path.join(k_dir, "cam_K.txt")
    for i in trange(img_num):
        color_path = os.path.join(color_dir, f"{i}.png")
        # depth_path = os.path.join(depth_dir, f"{i}.png")
        for f in mask_f:
            f_path = Path(f)
            mask_path = os.path.join(mask_dir, f_path.stem, f"{i}.png")
            depth_path = os.path.join(depth_dir, f_path.stem, f"{i}.png")
            if not os.path.exists(mask_path):
                continue
            to_colorful_xyz(color_path, depth_path, mask_path, k_path, i,out_dir/f_path.stem)

def to_colorful_xyz(color_dir,depth_dir,mask_dir,k_dir,i,out_dir):
    color_dir = Path(color_dir)
    depth_dir = Path(depth_dir)
    mask_dir = Path(mask_dir)
    k_dir = Path(k_dir)
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    color = cv2.imread(str(color_dir))
    # print('depth_dir', depth_dir)
    depth = cv2.imread(str(depth_dir), cv2.IMREAD_UNCHANGED) / 1e3
    mask = cv2.imread(str(mask_dir), cv2.IMREAD_UNCHANGED).astype(bool)
    K = np.loadtxt(str(k_dir))
    xyz_map = depth2xyzmap(depth, K)
    if mask.any():
        pcd = toOpen3dCloud(xyz_map[mask], color[mask])
        o3d.io.write_point_cloud(str(out_dir/f'{i}.ply'), pcd)

def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud

def depth2xyzmap(depth, K, uvs=None):
    invalid_mask = (depth<0.1)
    H,W = depth.shape[:2]
    if uvs is None: # u -> x, v -> y
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map

def apply_cam2world(in_dir, c2w_path, out_dir):
    seg_folder = os.listdir(in_dir)
    seg_folder = [f for f in seg_folder if os.path.isdir(os.path.join(in_dir, f))]
    os.makedirs(out_dir, exist_ok=True)
    c2w = np.load(c2w_path)
    for f in seg_folder:
        f_path = Path(f)
        in_folder = os.path.join(in_dir, f_path.stem)
        out_folder = os.path.join(out_dir, f_path.stem)
        apply_cam2world_single_folder(in_folder, c2w, out_folder)
    
def apply_cam2world_single_folder(in_dir, c2w, out_dir):
    pts_path = os.listdir(in_dir)
    os.makedirs(out_dir, exist_ok=True)
    n = len(pts_path)
    for i in range(n):
        pts_io = o3d.io.read_point_cloud(os.path.join(in_dir, pts_path[i]))
        pts = np.asarray(pts_io.points)
        pts_color = np.asarray(pts_io.colors)
        idx = pts_path[i].split(".")[0]
        c2w_adjust = c2w[int(idx)].copy()
        c2w_adjust[:3,1] *=   -1
        c2w_adjust[:3,2] *=   -1
        # c2w[int(idx)][:3,1] *= -1
        # c2w[int(idx)][:3,2] *= -1
        pts_io.transform(c2w_adjust)
        # pts = np.dot(c2w[int(idx)][:3,:3], pts.T).T + c2w[int(idx)][:3,3]
        o3d.io.write_point_cloud(os.path.join(out_dir, f"{idx}.ply"), pts_io)

def remove_outlier(in_dir,out_dir):
    pcd_raw = o3d.io.read_point_cloud(str(in_dir))
    pcd = pcd_raw.voxel_down_sample(voxel_size=0.02)

    # print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.5)
    inline_cloud = pcd.select_by_index(ind)

    # cl, ind = inline_cloud.remove_radius_outlier(nb_points=200, radius=2)
    # outlier_cloud2 = inline_cloud.select_by_index(ind, invert=True)
    # inline_cloud = inline_cloud.select_by_index(ind)


    if inline_cloud.is_empty():
        inline_cloud = pcd
    # print("Number of inline points: ", len(inline_cloud.points))
    o3d.io.write_point_cloud(str(out_dir), inline_cloud)
    return inline_cloud

def remove_outlier_folder(in_dir,out_path,work_dir):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    folders = [f for f in in_dir.iterdir() if f.is_dir()]
    for f in folders:
        all_pts = np.empty((0, 3))
        os.makedirs(out_path/f.name, exist_ok=True)
        ply_path = [p  for p in f.glob("*.ply")]
        # print(ply_path)
        for file in ply_path:

            inline_cloud = remove_outlier(file, str(file).replace("3d_seg_world", "3d_seg_world_filtered"))
            if inline_cloud is not None:
                inline_array = np.asarray(inline_cloud.points)
                all_pts = np.concatenate((all_pts, inline_array), axis=0)

        all_pts_cloud = o3d.geometry.PointCloud()
        all_pts_cloud.points = o3d.utility.Vector3dVector(all_pts)
        o3d.io.write_point_cloud(str(work_dir/f"{f.stem}.ply"),all_pts_cloud)
            
def erode_mask_folder(in_dir, out_dir, kernel_size = 10):
    f = os.listdir(in_dir)
    folders = [f for f in f if os.path.isdir(os.path.join(in_dir, f))]
    for folder in folders:
        in_folder = os.path.join(in_dir, folder)
        out_folder = os.path.join(out_dir, folder)
        erode_mask_single_folder(in_folder, out_folder, kernel_size)

def erode_mask_single_folder(in_dir, out_dir, kernel_size = 10):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(in_dir):
        mask_path = os.path.join(in_dir, file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
        mask_erode = cv2.erode(mask, kernel, iterations=1) 
        cv2.imwrite(os.path.join(out_dir, file), mask_erode)

def masked_depth_folder(depth_dir, mask_dir, grad_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    folders = [f for f in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, f))]
    for folder in folders:
        mask_folder = os.path.join(mask_dir, folder)
        out_folder = os.path.join(out_dir, folder)
        grad_folder = os.path.join(grad_dir, folder)
        masked_depth_single_folder(depth_dir, mask_folder,grad_folder, out_folder)

def masked_depth_single_folder(depth_dir, mask_dir, grad_folder, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(grad_folder, exist_ok=True)

    for file in os.listdir(depth_dir):
        depth_path = os.path.join(depth_dir, file)
        mask_path = os.path.join(mask_dir, file)
        if os.path.exists(mask_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = mask.astype(np.float32) / 255
            masked_depth = depth * mask
            masked_depth = masked_depth.astype(np.uint16)
            cv2.imwrite(os.path.join(out_dir, file), masked_depth)
            gradx = cv2.Sobel(masked_depth, cv2.CV_16U, 1, 0)
            grady = cv2.Sobel(masked_depth, cv2.CV_16U, 0, 1)
            grad = np.sqrt(gradx ** 2 + grady ** 2)
            grad = grad.astype(np.uint8)
            grad_thres = np.mean(grad)
            masked_depth[grad > grad_thres] = 0
            cv2.imwrite(os.path.join(grad_folder, file), masked_depth)

def annotate_on_slam_result_single_folder(tmp,object_path,color,out_path,original_color):

    object = o3d.io.read_point_cloud(str(object_path))
    
    object_pts = np.asarray(object.points)
    tree_object = o3d.geometry.KDTreeFlann(object)
    slam_res_pts = tmp.points
    # a bool array of size slam_res_pts
    indicator = np.zeros(len(slam_res_pts)).astype(bool)
    # print("indicator", indicator.shape)
    empty_pts = np.array([]).reshape(0,3)

    for id,pts in enumerate(slam_res_pts):
        [_, idx, _] = tree_object.search_knn_vector_3d(pts, 1)
        distance = np.linalg.norm(object_pts[idx] - pts)
        if np.linalg.norm(distance) < 0.01:
            indicator[id] = True
            empty_pts = np.vstack((empty_pts, pts))
            color_array = np.array(color).astype(np.float64).transpose()
            # print('color array', color_array.shape)
            tmp.colors [id] = color_array
            # o3d.utility.Vector3dVector(color_array)
    # print("indicator", indicator)   
    bool_path = Path(out_path)/ "indicator.npy"
    np.save(bool_path, indicator)
    ply_path = Path(out_path)/ "pts.ply"
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(empty_pts)
    # print("original_color", original_color.shape)
    # print("indicator", original_color[indicator].shape)
    print("indicator", original_color[indicator])
    pointcloud.colors = o3d.utility.Vector3dVector(original_color[indicator])
    # print(tmp.colors[indicator].shape)
    # pointcloud.colors = o3d.utility.Vector3dVector(tmp.colors[indicator])
    o3d.io.write_point_cloud(str(ply_path), pointcloud)
    # print("bool_ok_number", np.sum(indicator))
    # print("empty_pts", empty_pts.shape)
    return tmp



def annotate_on_slam_result(object_path_dir,slam_res_path,out_ply_path,annotated_folder):
    object_paths = os.listdir(object_path_dir)
    color_list = [(1,0,0),    (0,1,0),  (0,0,1),
                  (1,1,0),    (1,0,1),   (0,1,1),
                  (0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),
                  (0.5,0.5,0.5),(0.5,0,0), (0,0.5,0),
                  (0,0,0.5),   (0,0.5,0), (0.5,0,0),
                  (0.5,0,0.5), (0,0.5,0.5),(0.5,0.5,0),
                  (0.5,0.5,0.5),(0.5,0,0), (0,0.5,0)]
    tmp = o3d.io.read_point_cloud(str(slam_res_path)) #slam ply
    original_color = np.asarray(tmp.colors).copy()

    for p in tqdm(object_paths):
        object_path = os.path.join(object_path_dir, p)
        out_file_path = os.path.join(annotated_folder, Path(p).stem)
        os.makedirs(out_file_path, exist_ok=True)
        tmp = annotate_on_slam_result_single_folder(tmp,object_path,color_list[int(p.split(".")[0])],out_file_path,original_color)
    o3d.io.write_point_cloud(str(out_ply_path), tmp)

def masked_rgb_folder(rgb_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    folders = [f for f in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, f))]

    for folder in folders:
        mask_folder = os.path.join(mask_dir, folder)    
        out_folder = os.path.join(out_dir, folder)
        masked_rgb_single_folder(rgb_dir, mask_folder, out_folder)

def masked_rgb_single_folder(rgb_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(rgb_dir):
        rgb_path = os.path.join(rgb_dir, file)
        mask_path = os.path.join(mask_dir, file)
        if os.path.exists(mask_path):
            rgb = cv2.imread(rgb_path) 
            if not os.path.exists(mask_path):
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # h, w,1
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            mask = mask.astype(np.float32) / 255
            grad_mask_1 = cv2.Sobel(mask, cv2.CV_32F, 1, 0)
            grad_mask_2 = cv2.Sobel(mask, cv2.CV_32F, 0, 1)
            grad_mask = np.sqrt(grad_mask_1 ** 2 + grad_mask_2 ** 2)
            rgb[grad_mask != 0] = [0, 255, 0]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # print('masked rgb', out_dir)
            # print(os.path.join(out_dir, file))
            cv2.imwrite(os.path.join(out_dir, file), rgb)

if __name__ == "__main__":
    # visualization
    render_depth_save_as_orginal_format("datasets/annex/after_remove_gradient/0", "/ghome/l6/yqliang/littleduck/to_render/after_remove_gradient")
    # render_depth_save_as_orginal_format("datasets/annex/depth/0", "/ghome/l6/yqliang/littleduck/to_render/after_remove_gradient")
    render_depth_save_as_orginal_format("datasets/annex/depth_masked/0", "/ghome/l6/yqliang/littleduck/to_render/masked_depth")
    # render_mask_contour("datasets/annex/mask_data/0", "/ghome/l6/yqliang/littleduck/to_render/contour")
    render_combine_three_imgs("/ghome/l6/yqliang/littleduck/to_render/contour", "/ghome/l6/yqliang/littleduck/to_render/masked_depth",
                              "/ghome/l6/yqliang/littleduck/datasets/annex/depth_masked_grad/0",
                              "/ghome/l6/yqliang/littleduck/to_render/combined")

    