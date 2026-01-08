import argparse
import time
from typing import Optional, Tuple

import numpy as np
import cv2
import pyzed.sl as sl
import viser
from viser.extras import ViserUrdf
import yourdfpy
from scipy.spatial.transform import Rotation as R
import rospy
from sensor_msgs.msg import JointState


# C is camera frame (OpenCV convention: X right, Y down, Z forward)
# ZED_CAMERA_T_R_C maps 3D points from camera frame C to robot base frame R:
# X_R = R_RC * X_C + t_RC
ZED_CAMERA_T_R_C = np.eye(4)
ZED_CAMERA_T_R_C[:3, :3] = np.array(
    [
        [0.9543812680846684, 0.08746057618774912, -0.2854943830305726],
        [0.29537672607257903, -0.41644924520026877, 0.8598387150313551],
        [-0.043691930876822334, -0.904942359371598, -0.42328517738189414],
    ]
)
ZED_CAMERA_T_R_C[:3, 3] = np.array(
    [0.5947949577333569, -0.9635715691360609, 0.6851893282998003]
)



def construct_camera_intrinsics(
    left_cam_params: sl.CameraParameters,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    fx_orig = left_cam_params.fx
    fy_orig = left_cam_params.fy
    cx_orig = left_cam_params.cx
    cy_orig = left_cam_params.cy

    orig_w = left_cam_params.image_size.width
    orig_h = left_cam_params.image_size.height

    sx = target_width / orig_w
    sy = target_height / orig_h

    fx = fx_orig * sx
    fy = fy_orig * sy
    cx = cx_orig * sx
    cy = cy_orig * sy

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def init_zed(serial_number: str, width: int, height: int, exposure: int, gain: int):
    zed = sl.Camera()
    input_type = sl.InputType()
    init_params = sl.InitParameters(input_t=input_type)
    init_params.svo_real_time_mode = True
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.set_from_serial_number(int(serial_number))

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)

    runtime_parameters = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    # Precompute intrinsics at our working resolution
    cam_info = zed.get_camera_information()
    left_params = cam_info.camera_configuration.calibration_parameters.left_cam
    K = construct_camera_intrinsics(left_params, width, height)
    return zed, runtime_parameters, image_mat, depth_mat, K


def read_rgbd_frame(
    zed: sl.Camera,
    runtime_parameters: sl.RuntimeParameters,
    image_mat: sl.Mat,
    depth_mat: sl.Mat,
    width: int,
    height: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None, None

    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    img = image_mat.get_data()
    if img.ndim == 3 and img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img[..., :3]
    img_rgb = cv2.resize(img_rgb, (width, height))

    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    depth_mm = depth_mat.get_data()
    depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m = (depth_mm.astype(np.float32) / 1000.0)
    depth_m[(depth_m < 0.001) | (~np.isfinite(depth_m))] = 0.0

    return img_rgb, depth_m


def depth_to_points(
    depth_m: np.ndarray,
    K: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    stride: int = 2,
    max_depth_m: float = 5.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = depth_m.shape
    v_coords, u_coords = np.indices((h, w))
    if stride > 1:
        v_coords = v_coords[::stride, ::stride]
        u_coords = u_coords[::stride, ::stride]
        depth = depth_m[::stride, ::stride]
        if rgb is not None:
            colors = rgb[::stride, ::stride, :]
        else:
            colors = None
    else:
        depth = depth_m
        colors = rgb

    z = depth.reshape(-1)
    valid = (z > 0.0) & (z < max_depth_m)
    z = z[valid]

    u = u_coords.reshape(-1)[valid]
    v = v_coords.reshape(-1)[valid]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_c = np.stack([x, y, z], axis=1)

    cols = colors.reshape(-1, 3)[valid]
    return pts_c, cols


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    R_rc = T[:3, :3]
    t_rc = T[:3, 3]
    return (pts @ R_rc.T) + t_rc[None, :]


def rotmat_to_wxyz(Rm: np.ndarray) -> Tuple[float, float, float, float]:
    xyzw = R.from_matrix(Rm).as_quat()
    wxyz = (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))
    return wxyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial_number", type=str, default="15107")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--exposure", type=int, default=25)
    parser.add_argument("--gain", type=int, default=40)
    parser.add_argument("--stride", type=int, default=2, help="Point sampling stride")
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    parser.add_argument(
        "--urdf_path",
        type=str,
        # default="/juno/u/kedia/sapg/assets/urdf/kuka_allegro_description/iiwa14_real.urdf",
        default="/juno/u/kedia/sapg/assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
    )
    parser.add_argument(
        "--joint_cfg",
        type=float,
        nargs="*",
        default=[-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.308],
        help="First 7 joint angles (rad) for the arm; others set to 0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8082,
        help="Viser server port",
    )
    args = parser.parse_args()

    T_rc = ZED_CAMERA_T_R_C.copy()

    server = viser.ViserServer(port=args.port)

    # T_rc = np.array([
    #     [0.92758014494119889, -0.19411036827083383, 0.31924322959246304, -0.53020504226664311],
    #     [-0.37106076306599645, -0.37868257306711106, 0.84788762166230092, -1.41357156913606086],
    #     [-0.04369193087682233, -0.90494235937159795, -0.42328517738189414, 1.02018932829980036],
    #     [0.00000000000000000, 0.00000000000000000, 0.00000000000000000, 1.00000000000000000]
    # ])

    T_rc = np.array([
        [0.95527630647288930, -0.17920451516639435, 0.23522950502752071, -0.47020504226664311],
        [-0.28890230754832508, -0.39580744250644329, 0.87170632964878869, -1.46857156913606079],
        [-0.06310812138518884, -0.90067874972183481, -0.42987806970668574, 1.02018932829980047],
        [0.00000000000000000, 0.00000000000000000, 0.00000000000000000, 1.00000000000000000]
    ])
    
    # Add calibration sliders for T_rc adjustment
    rotation_folder = server.gui.add_folder("Rotation (degrees)")
    with rotation_folder:
        slider_rot_x = server.gui.add_slider(
            "Rotate X", min=-10.0, max=10.0, step=0.1, initial_value=0.0
        )
        slider_rot_y = server.gui.add_slider(
            "Rotate Y", min=-10.0, max=10.0, step=0.1, initial_value=0.0
        )
        slider_rot_z = server.gui.add_slider(
            "Rotate Z", min=-10.0, max=10.0, step=0.1, initial_value=0
        )
    # -1.04, -0.54, 0
    translation_folder = server.gui.add_folder("Translation (meters)")
    with translation_folder:
        slider_trans_x = server.gui.add_slider(
            "Translate X", min=-0.1, max=0.1, step=0.001, initial_value=0
        )
        slider_trans_y = server.gui.add_slider(
            "Translate Y", min=-0.1, max=0.1, step=0.001, initial_value=0
        )
        slider_trans_z = server.gui.add_slider(
            "Translate Z", min=-0.1, max=0.1, step=0.001, initial_value=0
        )
    
    # Add button to print current calibrated T_rc
    print_button = server.gui.add_button("Print Calibrated T_rc")
    
    def get_adjusted_T_rc():
        """Compute T_rc with slider adjustments applied."""
        # Start with the base transformation
        T_adjusted = T_rc.copy()
        
        # Apply rotation adjustments (in degrees, convert to radians)
        rot_x_rad = np.deg2rad(slider_rot_x.value)
        rot_y_rad = np.deg2rad(slider_rot_y.value)
        rot_z_rad = np.deg2rad(slider_rot_z.value)
        
        # Create rotation matrices for each axis
        R_x = R.from_euler('x', rot_x_rad).as_matrix()
        R_y = R.from_euler('y', rot_y_rad).as_matrix()
        R_z = R.from_euler('z', rot_z_rad).as_matrix()
        
        # Combined adjustment rotation
        R_adjust = R_z @ R_y @ R_x
        
        # Apply rotation adjustment to the base rotation
        T_adjusted[:3, :3] = R_adjust @ T_adjusted[:3, :3]
        
        # Apply translation adjustments
        T_adjusted[0, 3] += slider_trans_x.value
        T_adjusted[1, 3] += slider_trans_y.value
        T_adjusted[2, 3] += slider_trans_z.value
        
        return T_adjusted
    
    @print_button.on_click
    def _(_):
        """Print the current calibrated T_rc when button is clicked."""
        T_calibrated = get_adjusted_T_rc()
        print("\n" + "="*60)
        print("CALIBRATED T_RC (ZED_CAMERA_T_R_C):")
        print("="*60)
        print("T_RC = np.array([")
        for i in range(4):
            row_str = "    [" + ", ".join(f"{T_calibrated[i, j]:.17f}" for j in range(4)) + "]"
            if i < 3:
                row_str += ","
            print(row_str)
        print("])")
        print("="*60 + "\n")

    # Frames: robot base at origin; camera pose per extrinsics
    server.scene.add_frame(
        "/robot_base",
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.002,
    )
    server.scene.add_grid(
        "/grid",
        position=(0.0, 0.0, 0.53),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    # Load URDF and place at robot base
    urdf = yourdfpy.URDF.load(
        args.urdf_path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_collision_meshes=False,
        root_node_name="/robot",
    )
    # Build joint config: first 7 from args, rest zeros
    actuated = list(viser_urdf.get_actuated_joint_limits().keys())
    num_j = len(actuated)
    cfg = np.zeros((num_j,), dtype=np.float64)
    for i in range(min(7, num_j, len(args.joint_cfg))):
        cfg[i] = float(args.joint_cfg[i])
    allegro_joints = [0.03580158647006279, 1.190307500756139, 0.04091241471899582, -0.0020815739716152164, -0.003517249230515697, 1.2851153897506231, 0.044026046173861466, 0.014320749234448864, -0.026443060708318096, 1.3508007502819834, 0.019888673216377658, 0.0169404863577189, 1.3616900779442058, 0.01507557136958743, 0.1047496180391897, 0.009729167245470401]
    for i in range(len(allegro_joints)):
        # cfg[i+7] = float(allegro_joints[i])
        cfg[i+7] = 0.0
    viser_urdf.update_cfg(cfg)

    
    def iiwa_joint_states_callback(msg):

        for i in range(7):
            cfg[i] = float(msg.position[i])
        viser_urdf.update_cfg(cfg)
        # print(f"IIWA joint states: {cfg}")
        # print(f"(msg): {msg.position}")
    # read rostopic /iiwa/joint_states
    rospy.init_node('viser_compare_pointclouds')
    rospy.Subscriber("/iiwa/joint_states", JointState, iiwa_joint_states_callback, queue_size=1)
    # Prepare ZED
    zed, rt_params, image_mat, depth_mat, K = init_zed(
        args.serial_number, args.width, args.height, args.exposure, args.gain
    )

    pcd_handle = None
    print(f"Viser at http://localhost:{args.port}")
    try:
        while True:
            rgb, depth = read_rgbd_frame(
                zed, rt_params, image_mat, depth_mat, args.width, args.height
            )
            pts_c, cols = depth_to_points(
                depth, K, rgb=rgb, stride=args.stride, max_depth_m=args.max_depth_m
            )
            # Use adjusted T_rc based on calibration sliders
            T_rc_adjusted = get_adjusted_T_rc()
            pts_r = transform_points(T_rc_adjusted, pts_c)

            pcd_handle = server.scene.add_point_cloud(
                "/zed_points_robot_frame",
                points=pts_r.astype(np.float32),
                colors=cols.astype(np.uint8),
                point_size=0.005,
            )

            # Small sleep to keep loop responsive without spinning at 100% CPU
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        zed.close()


if __name__ == "__main__":
    main()
