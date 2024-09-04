import os
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import shutil
import argparse

def depth_render_with_colorized_range(in_dir,save_imgs = False):
    # three color range: red, green, blue
    # red: 0
    # green: 0-500
    # blue: 500-1000
    # black: 1000+
    img_names = os.listdir(in_dir)
    if in_dir.endswith("/"):
        in_dir = in_dir[:-1]
    if save_imgs:
        if os.path.exists(os.path.join("colorized", in_dir.split("/")[-1])):
            shutil.rmtree(os.path.join("colorized", in_dir.split("/")[-1]))
        os.makedirs(os.path.join("colorized", in_dir.split("/")[-1]), exist_ok=False)
    for img_depth in img_names:
        depth = cv2.imread(os.path.join(in_dir, img_depth), cv2.IMREAD_UNCHANGED)
        colorized_depth = np.zeros( (depth.shape[0], depth.shape[1], 3)) # black  
        colorized_depth[depth == 0] = (1,0,0) # red
        colorized_depth[(depth < 500) & (depth != 0)] = (0,1,0) # green
        colorized_depth[(depth < 1000) & (depth >= 500)] = (0,0,1) # blue

        fig, (ax1, ax2) = plt.subplots(1, 2)
        # add text for color range into legend
        ax1.text( 0,1200, "red: 0", color='red', fontsize=10)
        ax1.text( 1000,1200, "green: 0-500", color='green', fontsize=10)
        ax1.text( 0,1400, "blue: 500-1000", color='blue',   fontsize=10)
        ax1.text(1000,1400, "black: 1000+", color='black', fontsize=10)
        ax1.imshow(colorized_depth)
        ax1.axis('off')
        ax2.imshow(depth)
        ax2.axis('off')
        if save_imgs:
            plt.savefig(os.path.join(os.path.join("colorized", in_dir.split("/")[-1]), img_depth))
        plt.close()

def render_depth_save_as_orginal_format(in_dir,out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_names = os.listdir(in_dir)
    for img_depth in img_names:
        # print("img path",os.path.join(in_dir, img_depth))
        depth_img = cv2.imread(os.path.join(in_dir, img_depth), cv2.IMREAD_UNCHANGED)
        # print("depth_img",depth_img.max(), depth_img.min())
        depth_img = depth_img.astype(np.float32)
        depth_img /= 5000
        depth_img[depth_img == 0] = 1
        # depth_img = depth_img /1e3 * 15000
        depth_img *= 255
        depth_img = depth_img.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, img_depth), depth_img)

def render_mask_contour(in_dir,out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_names = os.listdir(in_dir)
    for img in img_names:
        mask = cv2.imread(os.path.join(in_dir, img), cv2.IMREAD_UNCHANGED) # h,w,1
        edges = cv2.Canny(mask, 100, 200)
        cv2.imwrite(os.path.join(out_dir, img), edges)

def render_combine_three_imgs(in_dir1, in_dir2,in_dir3, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_names = os.listdir(in_dir1)
    for img in img_names:
        img1 = cv2.imread(os.path.join(in_dir1, img), cv2.IMREAD_UNCHANGED) 
        img2 = cv2.imread(os.path.join(in_dir2, img), cv2.IMREAD_UNCHANGED) #uint16
        img3 = cv2.imread(os.path.join(in_dir3, img), cv2.IMREAD_UNCHANGED)
        img_to_save = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
        img_to_save[:,:,0] = img1
        img_to_save[:,:,1] = 256 - img2 // 256
        img_to_save[:,:,2] = img3 
        cv2.imwrite(os.path.join(out_dir, img), img_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default="/ghome/l6/yqliang/littleduck/datasets/annex/depth_filtered")
    args = parser.parse_args()
    print("args",args)
    depth_render_with_colorized_range(args.in_dir, save_imgs=True)