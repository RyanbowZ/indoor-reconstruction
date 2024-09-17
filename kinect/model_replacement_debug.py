import os
from utils import *
import pathlib
import time



def data_prepare(dataset_path, start_time,start_from = 4): 
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
    # 4.
    # 
    # 5. split_2d_seg_into_individual_objects 
    #     write into mask_data subfolder
    if start_from <= 4 :
        split_2d_seg_into_folder( dataset_path/"mask_data")
    print("   split_2d_seg_into_individual_objects done", time.time()-start_time)
    # 6. erode mask // not used
    # if start_from <= 5:
    #     erode_mask_folder(dataset_path/"mask_data", dataset_path/"mask_data_erode", kernel_size = 10)
    # print("   erode_mask_folder done", time.time()-start_time)
    # 7. masked_depth
    #    write into depth_masked and after_remove_gradient subfolder
    if start_from <= 6:
        if os.path.exists(dataset_path/"depth_masked"):
            if input("Do you want to remove the folder depth_masked? (y/n)") == "y":
                shutil.rmtree(dataset_path/"depth_masked")
        if os.path.exists(dataset_path/"after_remove_gradient"):
            if input("Do you want to remove the folder after_remove_gradient? (y/n)") == "y":
                shutil.rmtree(dataset_path/"after_remove_gradient")
        masked_depth_folder(dataset_path/"depth_sequential", dataset_path/"mask_data", dataset_path / "after_remove_gradient",dataset_path/"depth_masked")
    print("   masked_depth_folder done", time.time()-start_time)
    
    # 8. masked_rgb
    # write into rgb_masked subfolder
    if start_from <= 7:
        if os.path.exists(dataset_path/"rgb_masked"):
            if input("Do you want to remove the folder rgb_masked? (y/n)") == "y":
                shutil.rmtree(dataset_path/"rgb_masked")
        masked_rgb_folder(dataset_path/"rgb_sequential", dataset_path/"mask_data", dataset_path/"rgb_masked")


def three_D_segmentation_for_individual_frames(dataset_path,start_time,start_from =2):
    # 0. split_3d_seg_into_individual_objects
    #    write into 3d_seg subfolder
    if start_from <= 0:
        split_3d_seg_into_folder(dataset_path/"rgb_sequential", dataset_path/"after_remove_gradient", dataset_path/"mask_data",dataset_path, dataset_path/"3d_seg")
    print("   split_3d_seg_into_folder done", time.time()-start_time)
    # 1. apply_cam2world_matrix
    #    write into 3d_seg_world subfolder
    if start_from <= 1:
        apply_cam2world(dataset_path/"3d_seg", dataset_path/"0"/"estimate_c2w.npy", dataset_path/"3d_seg_world")
    print("   apply_cam2world done", time.time()-start_time)
    # 2. filter pointcloud
    #   write into 3d_seg_world_filtered and seg_object subfolder
    if start_from <= 2:
        remove_outlier_folder(dataset_path/"3d_seg_world", dataset_path/"3d_seg_world_filtered",dataset_path/"seg_object")
    print("   remove_outlier_folder done", time.time()-start_time)
    # 3. annotate_on_slam_result and save 3d seg individually
    if start_from <= 3:
        annotate_on_slam_result(dataset_path/"seg_object",dataset_path/"slam_res.ply",dataset_path/"slam_annotation.ply",dataset_path/"seg_object_on_slam")
    print("   annotate_on_slam_result and save individually done")



if __name__ ==  "__main__":
    start_time = time.time()
    phases = [2]
    print("time unit: second")
    print("Phase 1: Data Preparation", time.time()-start_time)
    sence_name = "annex"
    dataset_path  = Path("datasets")/sence_name
    if 1 in phases:
        data_prepare(dataset_path,start_time,start_from =4)
    print("Phase 2: 3D Segmentation for individual frames;", time.time()-start_time)
    if 2 in phases:
        three_D_segmentation_for_individual_frames(dataset_path,start_time,start_from = 3)
