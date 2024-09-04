from queue import Empty
import open3d as o3d
import numpy as np
import os
import time
from multiprocessing import Process, Queue
import pytorch3d.ops
from torch import Tensor
import torch
import tqdm

def from_quaernion_and_loc_to_4_by_4_matrix(quaternion,loc):
    quaternion = Tensor(quaternion)
    loc = Tensor(loc)
    pose = np.eye(4)
    qua_order = [3, 0, 1, 2]
    quaternion = quaternion[qua_order]
    pose[:3, :3] = pytorch3d.transforms.quaternion_to_matrix(quaternion)
    pose[0:3, 3] = loc.reshape(3)
    return pose

def create_camera_actor(i, is_gt=False, scale=0.005):
    cam_points = scale * np.array([
        [0,   0,   0],
        [-1,  -1, 1.5],
        [1,  -1, 1.5],
        [1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5]])
    cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4],
                          [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])
    points = []
    for cam_line in cam_lines:
        begin_points, end_points = cam_points[cam_line[0]
                                              ], cam_points[cam_line[1]]
        t_vals = np.linspace(0., 1., 100)
        begin_points, end_points
        point = begin_points[None, :] * \
            (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    camera_actor = o3d.geometry.PointCloud(
    points=o3d.utility.Vector3dVector(points))
    red_level = float(i)/90
    color = (0.0, 0.0, 0.0) if is_gt else (red_level, .0, .0)
    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor

def vis_cameras():
    vis = o3d.visualization.Visualizer()
    camera_actor = create_camera_actor(0)
    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.001, resolution=10)

    vis.create_window(window_name="camera", height=100, width=190)
    vis.add_geometry(camera_actor)
    vis.add_geometry(origin)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True

    # vis.update_geometry(camera_actor)
    # vis.reset_view_point(True)
    vis.run()
    # vis.destroy_window()

    return vis

def filter_pose_by_1_over_N(poses,N=5):
    # poses = np.array(poses)
    if N ==1:
        return poses
    poses = poses[:-(N-1):N] # get every Nth element and remove the last N-1 elements
    return poses

def read_pose(file_path,filter_ratio=1):
    pose_loc = []
    pose_rot = []
    with open(file_path, 'r') as f:
        # read until eof
        while True:
            line = f.readline()
            if not line:
                break
            pose = line.strip().split(' ')
            pose_loc.append( [float(p) for p in pose[1:4]])
            pose_rot.append( [float(p) for p in pose[4:]])
            # print(pose_loc, pose_rot)
    pose_loc = filter_pose_by_1_over_N(pose_loc, filter_ratio)
    # print( "@ read_pose: the length of pose_location is", len(pose_loc))
    pose_rot = filter_pose_by_1_over_N(pose_rot,filter_ratio)
    # print( "@ read_pose: the length of pose_rotation is", len(pose_rot))

    return pose_loc, pose_rot

def draw_trajectory(queue, output, init_pose, cam_scale,
                    save_rendering, near, estimate_c2w_list, gt_c2w_list):

    draw_trajectory.queue = queue
    draw_trajectory.cameras = {}
    draw_trajectory.points = {}
    draw_trajectory.ix = 0
    draw_trajectory.warmup = 0
    draw_trajectory.mesh = None
    draw_trajectory.frame_idx = 0
    draw_trajectory.traj_actor = None
    draw_trajectory.traj_actor_gt = None
    if save_rendering:
        os.system(f"rm -rf {output}/tmp_rendering")

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        while True:
            try:
                data = draw_trajectory.queue.get_nowait()
                if data[0] == 'pose':
                    i, pose, is_gt = data[1:]
                    if is_gt:
                        i += 100000

                    if i in draw_trajectory.cameras:
                        cam_actor, pose_prev = draw_trajectory.cameras[i]
                        pose_change = pose @ np.linalg.inv(pose_prev)
                        print(f"when {i} is in draw_trajectory.cameras: pose_change: ", draw_trajectory.cameras[i])

                        cam_actor.transform(pose_change)
                        vis.update_geometry(cam_actor)

                        if i in draw_trajectory.points:
                            pc = draw_trajectory.points[i]
                            pc.transform(pose_change)
                            vis.update_geometry(pc)

                    else:
                        cam_actor = create_camera_actor(i, is_gt, cam_scale)
                        cam_actor.transform(pose)
                        vis.add_geometry(cam_actor)

                    draw_trajectory.cameras[i] = (cam_actor, pose)

                elif data[0] == 'mesh':
                    meshfile = data[1]
                    if draw_trajectory.mesh is not None:
                        vis.remove_geometry(draw_trajectory.mesh)
                    draw_trajectory.mesh = o3d.io.read_triangle_mesh(meshfile)
                    # draw_trajectory.mesh.compute_vertex_normals()
                    # flip face orientation
                    # new_triangles = np.asarray(draw_trajectory.mesh.triangles)[:, ::-1]
                    new_triangles = np.asarray(draw_trajectory.mesh.triangles)
                    new_triangles = np.concatenate([new_triangles, new_triangles[:, ::-1]], axis=0)
                    draw_trajectory.mesh.triangles = o3d.utility.Vector3iVector(
                        new_triangles)
                    # draw_trajectory.mesh.triangle_normals = o3d.utility.Vector3dVector(
                        # -np.asarray(draw_trajectory.mesh.triangle_normals))
                    vis.add_geometry(draw_trajectory.mesh)

                elif data[0] == 'traj':
                    i, is_gt = data[1:]

                    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
                    traj_actor = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(gt_c2w_list[1:i, :3, 3] if is_gt else estimate_c2w_list[1:i, :3, 3]))
                    traj_actor.paint_uniform_color(color)

                    if is_gt:
                        if draw_trajectory.traj_actor_gt is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor_gt)
                            tmp = draw_trajectory.traj_actor_gt
                            del tmp
                        draw_trajectory.traj_actor_gt = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor_gt)
                    else:
                        if draw_trajectory.traj_actor is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor)
                            tmp = draw_trajectory.traj_actor
                            del tmp
                        draw_trajectory.traj_actor = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor)

                elif data[0] == 'reset':
                    draw_trajectory.warmup = -1

                    for i in draw_trajectory.points:
                        vis.remove_geometry(draw_trajectory.points[i])

                    for i in draw_trajectory.cameras:
                        vis.remove_geometry(draw_trajectory.cameras[i][0])

                    draw_trajectory.cameras = {}
                    draw_trajectory.points = {}

            except Empty:
                break

        # hack to allow interacting with vizualization during inference
        if len(draw_trajectory.cameras) >= draw_trajectory.warmup:
            cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        vis.poll_events()
        vis.update_renderer()
        if save_rendering:
            # save the renderings, useful when making a video
            draw_trajectory.frame_idx += 1
            os.makedirs(f'{output}/tmp_rendering', exist_ok=True)
            vis.capture_screen_image(
                f'{output}/tmp_rendering/{draw_trajectory.frame_idx:06d}.jpg')

    vis = o3d.visualization.Visualizer()

    vis.register_animation_callback(animation_callback)
    vis.create_window(window_name=output, height=1080, width=1920)
    vis.get_render_option().point_size = 4
    vis.get_render_option().mesh_show_back_face = False

    ctr = vis.get_view_control()
    ctr.set_constant_z_near(near)
    ctr.set_constant_z_far(1000)

    # set the viewer's pose in the back of the first frame's pose
    param = ctr.convert_to_pinhole_camera_parameters()
    init_pose[:3, 3] += 2*normalize(init_pose[:3, 2])
    init_pose[:3, 2] *= -1
    init_pose[:3, 1] *= -1
    init_pose = np.linalg.inv(init_pose)

    param.extrinsic = init_pose
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    vis.destroy_window()
    

    vis = o3d.visualization.Visualizer() # create a visualizer object

    vis.register_animation_callback(animation_callback)

    vis.create_window(window_name="banana", height=1080, width=1920)
    vis.get_render_option().point_size = 4
    vis.get_render_option().mesh_show_back_face = False

    ctr = vis.get_view_control()
    ctr.set_constant_z_near(0)
    ctr.set_constant_z_far(1000)

    # set the viewer's pose in the back of the first frame's pose
    param = ctr.convert_to_pinhole_camera_parameters()
    init_pose = np.eye(4)
    init_pose[:3, 3] += 2*normalize(init_pose[:3, 2])
    init_pose[:3, 2] *= -1
    init_pose[:3, 1] *= -1
    init_pose = np.linalg.inv(init_pose)

    param.extrinsic = init_pose
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    vis.destroy_window()



def normalize(x):
    return x / np.linalg.norm(x)

class SLAMFrontend:
    def __init__(self, output, init_pose, cam_scale=1, save_rendering=False,
                 near=0, estimate_c2w_list=None, gt_c2w_list=None):
        self.queue = Queue()
        self.p = Process(target=draw_trajectory, args=(
            self.queue, output, init_pose, cam_scale, save_rendering,
            near, estimate_c2w_list, gt_c2w_list))

    def update_pose(self, index, pose, gt=False):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()

        pose[:3, 2] *= -1
        self.queue.put_nowait(('pose', index, pose, gt))
        
    def update_mesh(self, path):
        self.queue.put_nowait(('mesh', path))

    def update_cam_trajectory(self, c2w_list, gt):
        self.queue.put_nowait(('traj', c2w_list, gt))

    def reset(self):
        self.queue.put_nowait(('reset', ))

    def start(self):
        self.p.start()
        return self

    def join(self):
        self.p.join()
    


if __name__ == "__main__":

    exp_name = "1"
    step = 451
    test = False
    if test:
        vis_cameras()
        exit()
    pose_l, pose_r = read_pose(f'D:\\datasets\\kinect\\appleUmbrella\\{exp_name}\\traj_{step}.txt')
    N = step //5
    
    is_gt = False
    cam_scale = 0.3
    init_pos = from_quaernion_and_loc_to_4_by_4_matrix(pose_r[0],pose_l[0])
    estimate_c2w = [from_quaernion_and_loc_to_4_by_4_matrix(pose_r[i],pose_l[i]) for i in range(N)]
    # np.save("estimate_c2w.npy", estimate_c2w)
    # exit()
    estimate_c2w = np.array(estimate_c2w)
    print(estimate_c2w.shape)
    output = f"D:\\datasets\\kinect\\appleUmbrella\\{exp_name}"
    frontend = SLAMFrontend(output, init_pose=init_pos, cam_scale=0.05,
                            save_rendering=True, near=0,
                            estimate_c2w_list=estimate_c2w, gt_c2w_list=estimate_c2w).start()
    
    for i in range(0, N):
        # show every second frame for speed up
        # if args.vis_input_frame and i % 2 == 0:
        #     idx, gt_color, gt_depth, gt_c2w = frame_reader[i]
        #     depth_np = gt_depth.numpy()
        #     color_np = (gt_color.numpy()*255).astype(np.uint8)
        #     depth_np = depth_np/np.max(depth_np)*255
        #     depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
        #     depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        #     color_np = np.clip(color_np, 0, 255)
        #     whole = np.concatenate([color_np, depth_np], axis=0)
        #     H, W, _ = whole.shape
        #     whole = cv2.resize(whole, (W//4, H//4))
        #     cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])
        #     cv2.waitKey(1)
        time.sleep(0.1)
        meshfile = f'{output}/result/final_mesh.ply'
        if  i== 0:
            frontend.update_mesh(meshfile)
        frontend.update_pose(i//10, estimate_c2w[i], gt=False)
        # if not args.no_gt_traj:
        #     frontend.update_pose(1, gt_c2w_list[i], gt=True)
        # the visualizer might get stucked if update every frame
        # with a long sequence (10000+ frames)
        if i % 10 == 0 or i == N-1:
            frontend.update_cam_trajectory(i, gt=False)
            # if not args.no_gt_traj:
            #     frontend.update_cam_trajectory(i, gt=True)

    