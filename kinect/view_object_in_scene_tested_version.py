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
import trimesh
from view_camera_in_the_scene import read_pose

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def from_quaernion_and_loc_to_4_by_4_matrix(quaternion,loc):
    quaternion = Tensor(quaternion)
    loc = Tensor(loc)
    pose = np.eye(4)
    qua_order = [3, 0, 1, 2]
    quaternion = quaternion[qua_order]
    pose[:3, :3] = pytorch3d.transforms.quaternion_to_matrix(quaternion)
    pose[0:3, 3] = loc.reshape(3)
    return pose

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

def draw_posed_3d_box( ob_in_cam, bbox, line_color=(0,255,0), linewidth=2):
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def to_homo(pts):
        assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
        homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
        return homo

    def draw_line3d(start,end,img):
        pts = np.stack((start,end),axis=0).reshape(-1,3)
        pts = (ob_in_cam@to_homo(pts).T).T[:,:3]   #(2,3)
        # projected = (K@pts.T).T
        # uv = np.round(projected[:,:2]/projected[:,2].reshape(-1,1)).astype(int)   #(2,2)
        # img = cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)
        return img

    for y in [ymin,ymax]:
        for z in [zmin,zmax]:
            start = np.array([xmin,y,z])
            end = start+np.array([xmax-xmin,0,0])
            img = draw_line3d(start,end,img)

    for x in [xmin,xmax]:
        for z in [zmin,zmax]:
            start = np.array([x,ymin,z])
            end = start+np.array([0,ymax-ymin,0])
            img = draw_line3d(start,end,img)

    for x in [xmin,xmax]:
        for y in [ymin,ymax]:
            start = np.array([x,y,zmin])
            end = start+np.array([0,0,zmax-zmin])
            img = draw_line3d(start,end,img)

    return img

def create_object(filepath):
    mesh = o3d.io.read_triangle_mesh(filepath)
    # mesh = trimesh.load(filepath)
    # # mesh.as_open3d()
    # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    # pts = (ob_in_cam@to_homo(pts).T).T[:,:3]   #(2,3)
    # center_pose = pose@np.linalg.inv(to_origin) 
    # # !!!!! need pose in ob_cam!!!
    
    # # after in inverse, the object is centered at the origin 
    # vis = draw_posed_3d_box( ob_in_cam=center_pose, bbox=bbox)

    # print(to_origin, extents)
    # exit()
    # # bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # cam_points = scale * np.array([
    #     [0,   0,   0],
    #     [-1,  -1, 1.5],
    #     [1,  -1, 1.5],
    #     [1,   1, 1.5],
    #     [-1,   1, 1.5],
    #     [-0.5, 1, 1.5],
    #     [0.5, 1, 1.5],
    #     [0, 1.2, 1.5]])
    # cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4],
    #                       [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])
    # points = []
    # for cam_line in cam_lines:
    #     begin_points, end_points = cam_points[cam_line[0]
    #                                           ], cam_points[cam_line[1]]
    #     t_vals = np.linspace(0., 1., 100)
    #     begin_points, end_points
    #     point = begin_points[None, :] * \
    #         (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
    #     points.append(point)
    # points = np.concatenate(points)
    # camera_actor = o3d.geometry.PointCloud(
    # points=o3d.utility.Vector3dVector(points))
    # red_level = float(i)/90
    # color = (0.0, 0.0, 0.0) if is_gt else (red_level, .0, .0)
    # camera_actor = o3d.geometry.PointCloud(
    #     points=o3d.utility.Vector3dVector(points))
    # camera_actor.paint_uniform_color(color)

    return mesh



def filter_pose_by_1_over_N(poses,N=5):
    # poses = np.array(poses)
    poses = poses[:-(N-1):N] # get every Nth element and remove the last N-1 elements
    print(len(poses))
    return poses


def read_obj_pose(file_path):
    files = os.listdir(file_path)
    pose= []
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(file_path, file), 'r') as f:
                lines = f.readlines()
                pose_matrix = np.eye(4)
                for i,line in enumerate(lines):
                    nums = line.split()
                    nums = [float(num) for num in nums]
                    pose_matrix[i] = nums
                pose.append(pose_matrix)
    return pose


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
        object_file = "D:\\datasets\\kinect\\appleUmbrella\\track_vis\\banana\\banana_mesh.obj"
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
                        cam_actor = create_object(object_file)
                        print('cam_actor:', cam_actor)
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
        pose = read_obj_pose(f'D:\\datasets\\kinect\\appleUmbrella\\track_vis\\ob_in_cam')
        print(len(pose))
        print(pose[0])
        print(pose[89])
        exit()
    
    pose = read_obj_pose(f'D:\\datasets\\kinect\\appleUmbrella\\track_vis\\ob_in_cam')

    N = step //5
    pose_l, pose_r = read_pose(f'D:\\datasets\\kinect\\appleUmbrella\\{exp_name}\\traj_{step}.txt')
    init_cam_pos = from_quaernion_and_loc_to_4_by_4_matrix(pose_r[0],pose_l[0])
    estimate_camera_c2w = [from_quaernion_and_loc_to_4_by_4_matrix(pose_r[i],pose_l[i]) for i in range(N)]
    for i in range(N):
        estimate_camera_c2w[i][:3, 2] *= -1
    
    
    
    is_gt = False
    cam_scale = 0.3
    init_pos = np.matmul(init_cam_pos , pose[0])
    estimate_c2w =  [np.matmul(estimate_camera_c2w[i],pose[i]) for i in range(N)]
    # init_pos = pose[0]
    # estimate_c2w =  [pose[i]for i in range(N)]

    estimate_c2w = np.array(estimate_c2w)
    print(estimate_c2w.shape)
    # print(estimate_c2w)
    # exit(0)

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

    