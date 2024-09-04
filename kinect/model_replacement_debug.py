import os
from utils import *
import pathlib
import time



def data_prepare(dataset_path, start_time,start_from = 0): 
    # 0. intrinsics camera npy -> txt
    if start_from <= 0:
        K_npy_to_txt(dataset_path/"K.npy", dataset_path/"cam_K.txt")
    # 1. pose_txt_to_npy
    if start_from <= 1 :
        pose,N = pose_txt_to_npy(dataset_path/"0")
    print("   pose_txt_to_npy done", time.time()-start_time)
    # 2. img_cvt_filename_timestep_to_sequential
    if start_from <= 2:
        img_cvt_filename_timestep_to_sequential(dataset_path/"depth_timestamp", dataset_path/"depth_sequential", start = 0)
        img_cvt_filename_timestep_to_sequential(dataset_path/"rgb_timestamp", dataset_path/"rgb_sequential", start = 0)
    print("   img_cvt_filename_timestep_to_sequential done", time.time()-start_time)
    # 3. filter depth (erosion / bilateral)
    if start_from <= 3:
        depth_erosion(dataset_path/"depth_sequential", dataset_path/"depth_filtered", threshold = 500,iteration = 1,kernel_size = 10)
    print("   depth_filter done", time.time()-start_time)
    # 4. get 2d segment mask for video recording
    if start_from <= 4:
        segment(dataset_path / "rgb_sequential", dataset_path / "2d_seg")
    print("   2d segment mask done", time.time() - start_time)
    # 5. split_2d_seg_into_individual_objects
    if start_from <= 5 :
        id_list = np.load(dataset_path/"id_list.npy")
        n = split_2d_seg_into_folder(id_list, dataset_path/"mask_data")
    print("   split_2d_seg_into_individual_objects done", time.time()-start_time)
    # 6. erode mask
    if start_from <= 6:
        erode_mask_folder(dataset_path/"mask_data", dataset_path/"mask_data_erode", kernel_size = 10)
    print("   erode_mask_folder done", time.time()-start_time)
    # 7. masked_depth
    if start_from <= 7:
        masked_depth_folder(dataset_path/"depth_sequential", dataset_path/"mask_data", dataset_path/"depth_masked")
    print("   masked_depth_folder done", time.time()-start_time)
    # 8. masked_rgb
    if start_from <= 8:
        masked_rgb_folder(dataset_path/"rgb_sequential", dataset_path/"mask_data", dataset_path/"rgb_masked")
    # 9. Plane Segmentation and generate 2d layout for wall/ground reconstruction
    if start_from <= 9:
        plane_segmentation(dataset_path/"room_mesh", dataset_path/"plane_seg")


def three_D_segmentation_for_individual_frames(dataset_path,start_from =2):
    # 0. split_3d_seg_into_individual_objects
    if start_from <= 0:
        split_3d_seg_into_folder(dataset_path/"rgb_sequential", dataset_path/"after_remove_gradient", dataset_path/"mask_data",dataset_path, dataset_path/"3d_seg")
    print("   split_3d_seg_into_folder done")
    # 1. apply_cam2world_matrix
    if start_from <= 1:
        apply_cam2world(dataset_path/"3d_seg", dataset_path/"0"/"estimate_c2w.npy", dataset_path/"3d_seg_world")
    print("   apply_cam2world done")
    # 2. filter pointcloud
    if start_from <= 2:
        remove_outlier_folder(dataset_path/"3d_seg_world", dataset_path/"3d_seg_world_filtered",dataset_path/"seg_object")
    print("   remove_outlier_folder done")
    if start_from <= 3:
        annotate_on_slam_result(dataset_path/"seg_object",dataset_path/"slam_res.ply",dataset_path/"slam_annotation.ply")
    print("   annotate_on_slam_result done")


if __name__ ==  "__main__":
    start_time = time.time()
    phases = [1]
    print("Phase 1: Data Preparation", time.time()-start_time)
    sence_name = "annex"
    dataset_path  = Path("datasets")/sence_name
    if 1 in phases:
        data_prepare(dataset_path,start_time,start_from =7)
    print("Phase 2: 3D Segmentation for individual frames;", time.time()-start_time)
    if 2 in phases:
        three_D_segmentation_for_individual_frames(dataset_path,start_from = 3)
