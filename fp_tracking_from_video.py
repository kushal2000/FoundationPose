import os
import time
import argparse
import numpy as np
import cv2
import pyzed.sl as sl
import trimesh
import nvdiffrast.torch as dr
import imageio
# Bring in FoundationPose and utilities (draw_xyz_axis, depth2xyzmap, etc.)
from estimater import *
from generate_mask import generate_binary_mask_box
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from VOT import Cutie, Tracker_2D  
from utils.kalman_filter_6d import KalmanFilter6D
from scipy.spatial.transform import Rotation
import rospy
from geometry_msgs.msg import PoseStamped
import imageio


def construct_camera_intrinsics(camera_params, target_width, target_height, camera_upsidedown=False):
    """
    Build a 3x3 intrinsics matrix K scaled to the target resolution.
    Optionally account for a 180-degree upside-down mounting by flipping the principal point.
    """
    fx_orig = camera_params.fx
    fy_orig = camera_params.fy
    cx_orig = camera_params.cx
    cy_orig = camera_params.cy

    orig_width = camera_params.image_size.width
    orig_height = camera_params.image_size.height

    scale_x = target_width / orig_width
    scale_y = target_height / orig_height

    fx = fx_orig * scale_x
    fy = fy_orig * scale_y
    cx = cx_orig * scale_x
    cy = cy_orig * scale_y

    if camera_upsidedown:
        cx = target_width - cx
        cy = target_height - cy

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)

    return K


def init_zed(serial_number: str, exposure: int, gain: int):
    zed = sl.Camera()
    input_type = sl.InputType()
    init_params = sl.InitParameters(input_t=input_type)
    init_params.svo_real_time_mode = True
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Depth in mm
    init_params.set_from_serial_number(int(serial_number))

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    # Manual exposure/gain (optional)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)

    runtime_parameters = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    return zed, runtime_parameters, image_mat, depth_mat

def select_mask_with_sam(rgb):
    """
    Use SAM box-based interactive selection to produce a binary mask.
    The SAM helper expects BGR input; convert from RGB.
    """
    bgr = rgb[..., ::-1]
    mask = generate_binary_mask_box(bgr, polygon_refinement=True)
    if mask is None:
        return None
    # Ensure binary uint8 mask (0/1)
    mask = (mask > 0).astype(np.uint8)
    return mask


def pose_matrix_to_posestamped(T, frame_id):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    quat_wxyz = trimesh.transformations.quaternion_from_matrix(T)
    msg.pose.orientation.x = float(quat_wxyz[1])
    msg.pose.orientation.y = float(quat_wxyz[2])
    msg.pose.orientation.z = float(quat_wxyz[3])
    msg.pose.orientation.w = float(quat_wxyz[0])
    return msg

def adjust_pose_to_image_point(
        ob_in_cam: torch.Tensor,
        K: torch.Tensor,
        x: float = -1.,
        y: float = -1.,
) -> torch.Tensor:
    """
    Adjusts the 6D pose(s) so that the projection matches the given 2D coordinate (x, y).

    Parameters:
    - ob_in_cam: Original 6D pose(s) as [4,4] or [B,4,4] tensor.
    - K: Camera intrinsic matrix (3x3 tensor).
    - x, y: Desired 2D coordinates on the image plane.

    Returns:
    - ob_in_cam_new: Adjusted pose(s) in same shape as input (tensor).
    """
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype

    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)  # [1, 4, 4]

    B = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)

    for i in range(B):
        R = ob_in_cam[i, :3, :3]
        t = ob_in_cam[i, :3, 3]

        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)

        ob_in_cam_new[i, :3, :3] = R
        ob_in_cam_new[i, :3, 3] = t_new

    return ob_in_cam_new if is_batched else ob_in_cam_new[0]


def get_pose_xy_from_image_point(
        ob_in_cam: torch.Tensor, 
        K: torch.Tensor, 
        x: float = -1., 
        y: float = -1.,
) -> tuple:
    """
    Computes new (tx, ty) in camera space such that the projection matches image point (x, y).

    Parameters:
    - ob_in_cam: 4x4 pose tensor.
    - K: 3x3 intrinsic matrix tensor.
    - x, y: Desired image coordinates.

    Returns:
    - tx, ty: New x/y in camera coordinate system.
    """

    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()  # [1, 4, 4]
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1. or y == -1.:
        return x, y
    
    t = ob_in_cam_new[:3, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    tz = t[2]

    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty

def get_mat_from_6d_pose_arr(pose_arr):
    # 提取位移 (xyz)
    xyz = pose_arr[:3]
    
    # 提取欧拉角
    euler_angles = pose_arr[3:]
    
    # 从欧拉角生成旋转矩阵
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()
    
    # 创建 4x4 变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    
    return transformation_matrix
    
def get_6d_pose_arr_from_mat(pose):
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='video_dir', help='Directory to save tracking results')
    parser.add_argument('--mesh', type=str, default='object_meshes/hammer_1.obj', help='Path to object mesh file (in meters).')
    parser.add_argument('--est_refine_iter', type=int, default=5, help='Refinement iterations for initial registration')
    parser.add_argument('--track_refine_iter', type=int, default=2, help='Refinement iterations per tracking step')
    parser.add_argument('--debug', type=int, default=0, help='Debug level for FoundationPose')
    parser.add_argument('--activate_2d_tracker', action='store_true', help='Activate 2D tracker')
    parser.add_argument('--activate_kalman_filter', action='store_true', help='Activate Kalman filter')
    parser.add_argument('--kf_measurement_noise_scale', type=float, default=0.05, help='Measurement noise scale for Kalman filter')
    args = parser.parse_args()

    print("Press 'r' key to reset pose tracking, 'q' or ESC to quit.")
    # Initialize camera
    T_RC = np.array([
        [0.88371235251068714, -0.23263011879228926, 0.40612277189381119, -0.45520504226664316],
        [-0.46598658048948349, -0.35631805180996939, 0.80987280035698606, -1.51357156913606095],
        [-0.04369193087682233, -0.90494235937159795, -0.42328517738189414, 0.68518932829980028],
        [0.00000000000000000, 0.00000000000000000, 0.00000000000000000, 1.00000000000000000]
    ])
    T_RC = np.array([
        [0.95527630647288930, -0.17920451516639435, 0.23522950502752071, -0.50020504226664309],
        [-0.28890230754832508, -0.39580744250644329, 0.87170632964878869, -1.43857156913606077],
        [-0.06310812138518884, -0.90067874972183481, -0.42987806970668574, 1.02018932829980047],
        [0.00000000000000000, 0.00000000000000000, 0.00000000000000000, 1.00000000000000000]
    ])
    save_folder = os.path.join(f'{args.video_dir}/')

    rgb_images_array = [os.path.join(args.video_dir, f'rgb/frame_{i:04d}.png') for i in range(len(os.listdir(args.video_dir)))]
    depth_images_array = [os.path.join(args.video_dir, f'depth/frame_{i:04d}.png') for i in range(len(os.listdir(args.video_dir)))]
    try:
        if args.debug >= 1:
            cv2.namedWindow('FoundationPose Live', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('FoundationPose Live', 640, 360)

        for i, (rgb, depth) in enumerate(zip(rgb_images_array, depth_images_array)):
            if i == 0:
                zed, runtime_parameters, image_mat, depth_mat = init_zed(args.serial_number, args.exposure, args.gain)
                camera_info = zed.get_camera_information()
                left_cam_params = camera_info.camera_configuration.calibration_parameters.left_cam
                K = construct_camera_intrinsics(left_cam_params, args.width, args.height, args.camera_upsidedown)
                mesh = trimesh.load(args.mesh, process=False)
                scorer = ScorePredictor()
                refiner = PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()
                est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                    mesh=mesh, scorer=scorer, refiner=refiner, debug=args.debug, glctx=glctx, debug_dir=args.save_dir)
                if args.activate_kalman_filter:
                    kf = KalmanFilter6D(args.kf_measurement_noise_scale)
                if args.activate_2d_tracker:
                    tracker_2D = Cutie()
                else:
                    tracker_2D = None

                # Get first frame and perform registration via ROI mask
                mask = select_mask_with_sam(rgb)
                if mask.sum() == 0:
                    print('Empty ROI selected. Exiting.')
                    return

                print('Registering initial pose...')
                t0 = time.time()
                pose = est.register(K=K, rgb=rgb, 
                                    depth=depth, 
                                    ob_mask=mask.astype(bool), 
                                    iteration=args.est_refine_iter)
                print(f'Initial registration done in {time.time()-t0:.3f}s')
                pose_pub.publish(pose_matrix_to_posestamped(pose, args.frame_id))

                frame_idx = 0
                mask_visualization_path = os.path.join(args.save_dir, f'mask_visualization_{frame_idx}.png')
                bbox_visualization_path = os.path.join(args.save_dir, f'bbox_visualization_{frame_idx}.png')
                if args.activate_kalman_filter:
                        kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))
                if args.activate_2d_tracker:
                    tracker_2D.initialize(rgb, init_info={"mask": (mask).astype(bool)}, 
                        mask_visualization_path=mask_visualization_path, bbox_visualization_path=bbox_visualization_path)

            else:
                mask_visualization_path = None
                bbox_visualization_path = None
                if args.activate_2d_tracker:
                    bbox_2d = tracker_2D.track(
                        rgb,
                        mask_visualization_path=mask_visualization_path,
                        bbox_visualization_path=bbox_visualization_path
                    )
                if args.activate_2d_tracker:
                    if not args.activate_kalman_filter:
                        est.pose_last = adjust_pose_to_image_point(ob_in_cam=est.pose_last, K=K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
                    else:
                        # using kf to estimate the 6d estimation of the last pose
                        kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, get_6d_pose_arr_from_mat(est.pose_last))
                        measurement_xy = np.array(get_pose_xy_from_image_point(ob_in_cam=est.pose_last, K=K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2))
                        kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
                        est.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(est.pose_last.device)

                pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter)
                if args.activate_2d_tracker and args.activate_kalman_filter:
                    # use kf to predict from last pose, and update kf status
                    kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)     # kf is alway one step behind

                robot_frame_pose = pose_matrix_to_posestamped(T_RC @ pose, 'robot_frame')
                offset_x = 0.005
                offset_z = 0.01
                offset_y = -0.00
                robot_frame_pose.pose.position.z += offset_z
                robot_frame_pose.pose.position.y += offset_y
                robot_frame_pose.pose.position.x += offset_x
                
                if args.debug >= 1:
                    if pose is not None:
                        vis = draw_xyz_axis(rgb, ob_in_cam=pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                        vis_bgr = vis[..., ::-1]
                    else:
                        vis_bgr = rgb[..., ::-1]
                    cv2.imshow('FoundationPose Live', vis_bgr)
    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

