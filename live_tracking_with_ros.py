import os
import time
import argparse
import numpy as np
import cv2
import pyzed.sl as sl
import trimesh
import nvdiffrast.torch as dr
import imageio
from estimater import *
from generate_mask import generate_binary_mask_box
from scipy.spatial.transform import Rotation
import rospy
from geometry_msgs.msg import PoseStamped


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


def read_rgbd_frame(zed, runtime_parameters, image_mat, depth_mat, width, height, camera_upsidedown=False):
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None, None

    # Retrieve color (left) image
    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    img = image_mat.get_data()
    # Convert to RGB from BGR(A) if needed
    if img.ndim == 3 and img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: replicate channel if grayscale
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img[..., :3]

    if camera_upsidedown:
        img_rgb = cv2.flip(img_rgb, -1)
    img_rgb = cv2.resize(img_rgb, (width, height))

    # Retrieve depth in millimeters, convert to meters (float32)
    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    depth_mm = depth_mat.get_data()
    if camera_upsidedown:
        depth_mm = cv2.flip(depth_mm, -1)
    depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m = (depth_mm.astype(np.float32) / 1000.0)

    # Sanitize depth
    depth_m[(depth_m < 0.001) | (~np.isfinite(depth_m))] = 0.0

    return img_rgb, depth_m


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


def main():
    parser = argparse.ArgumentParser(
        description="Live 6D object pose tracking with ROS publishing."
    )
    parser.add_argument('--mesh_path', type=str, required=True,
                        help='Path to object mesh file (.obj, in meters)')
    parser.add_argument('--calibration', type=str, required=True,
                        help='Path to 4x4 camera-to-robot transform (.npy or .txt)')
    parser.add_argument('--serial_number', type=str, default='15107',
                        help='ZED camera serial number')
    parser.add_argument('--camera_upsidedown', action='store_true',
                        help='Whether camera is mounted upside down')
    parser.add_argument('--width', type=int, default=960, help='Working image width')
    parser.add_argument('--height', type=int, default=540, help='Working image height')
    parser.add_argument('--exposure', type=int, default=25, help='Camera exposure value')
    parser.add_argument('--gain', type=int, default=40, help='Camera gain value')
    parser.add_argument('--fps', type=float, default=40.0, help='Target FPS for tracking loop')
    parser.add_argument('--est_refine_iter', type=int, default=5,
                        help='Refinement iterations for initial registration')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                        help='Refinement iterations per tracking step')
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug level (0=off, 1=show vis)')
    parser.add_argument('--save_dir', type=str, default='live_tracking_results',
                        help='Directory to save tracking results')
    parser.add_argument('--frame_id', type=str, default='camera_frame',
                        help='TF frame id for the camera frame')
    args = parser.parse_args()

    # Load camera-to-robot calibration
    if args.calibration.endswith('.npy'):
        T_RC = np.load(args.calibration)
    else:
        T_RC = np.loadtxt(args.calibration)
    T_RC = T_RC.reshape(4, 4)

    rospy.init_node('foundationpose_live_tracking', anonymous=True)
    pose_pub = rospy.Publisher('camera_frame/current_object_pose', PoseStamped, queue_size=1)
    robot_pose_pub = rospy.Publisher('robot_frame/current_object_pose', PoseStamped, queue_size=1)

    if args.debug >= 1:
        cv2.namedWindow('FoundationPose Live', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FoundationPose Live', 640, 360)

    # Initialize camera
    zed, runtime_parameters, image_mat, depth_mat = init_zed(args.serial_number, args.exposure, args.gain)

    try:
        # Build intrinsics at target resolution
        camera_info = zed.get_camera_information()
        left_cam_params = camera_info.camera_configuration.calibration_parameters.left_cam
        K = construct_camera_intrinsics(left_cam_params, args.width, args.height, args.camera_upsidedown)

        # Load mesh
        mesh = trimesh.load(args.mesh_path, process=False)

        # Initialize FoundationPose components
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=scorer, refiner=refiner,
            debug=args.debug, glctx=glctx, debug_dir=args.save_dir
        )

        # Get first frame and perform registration via SAM mask
        rgb, depth = read_rgbd_frame(zed, runtime_parameters, image_mat, depth_mat,
                                     args.width, args.height, args.camera_upsidedown)

        mask = select_mask_with_sam(rgb)
        if mask.sum() == 0:
            print('Empty ROI selected. Exiting.')
            return

        print('Registering initial pose...')
        t0 = time.time()
        pose = est.register(K=K, rgb=rgb, depth=depth,
                            ob_mask=mask.astype(bool),
                            iteration=args.est_refine_iter)
        print(f'Initial registration done in {time.time()-t0:.3f}s')
        pose_pub.publish(pose_matrix_to_posestamped(pose, args.frame_id))

        target_frame_time = 1.0 / args.fps if args.fps > 0 else 0.0

        vis_array = []
        while not rospy.is_shutdown():
            loop_start = time.time()
            rgb, depth = read_rgbd_frame(zed, runtime_parameters, image_mat, depth_mat,
                                         args.width, args.height, args.camera_upsidedown)

            pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter)

            pose_pub.publish(pose_matrix_to_posestamped(pose, args.frame_id))
            robot_frame_pose = pose_matrix_to_posestamped(T_RC @ pose, 'robot_frame')
            robot_pose_pub.publish(robot_frame_pose)

            # Visualization
            if args.debug >= 1:
                if pose is not None:
                    vis = draw_xyz_axis(rgb, ob_in_cam=pose, scale=0.1, K=K,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    vis_bgr = vis[..., ::-1]
                    vis_array.append(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
                else:
                    vis_bgr = rgb[..., ::-1]
                cv2.imshow('FoundationPose Live', vis_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # q or ESC
                    break

            # Pace to target FPS
            elapsed = time.time() - loop_start
            tracking_fps = 1.0 / elapsed
            print(f"Tracking FPS: {tracking_fps:.1f}")
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        zed.close()
        cv2.destroyAllWindows()
        if vis_array:
            os.makedirs(args.save_dir, exist_ok=True)
            imageio.mimsave(f'{args.save_dir}/vis.mp4', vis_array, fps=10)


if __name__ == '__main__':
    main()
