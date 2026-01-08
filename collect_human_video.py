import os
import cv2
import pyzed.sl as sl
import shutil
import argparse
import time
import numpy as np
import datetime


def create_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    
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

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save captured frames')
    parser.add_argument('--serial_number', type=str, default='15107', help='ZED camera serial number')
    parser.add_argument('--camera_upsidedown', action='store_true', help='Whether camera is mounted upside down')
    parser.add_argument('--width', type=int, default=960, help='Image width')
    parser.add_argument('--height', type=int, default=540, help='Image height')
    parser.add_argument('--exposure', type=int, default=25, help='Camera exposure value')
    parser.add_argument('--gain', type=int, default=40, help='Camera gain value')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second for capture (default: 10)')
    
    args = parser.parse_args()
    
    # Calculate target frame time from FPS
    target_frame_time = 1.0 / args.fps if args.fps > 0 else 0.1

    # Create a directory to save frames
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(args.save_dir, current_time)

    # delete the save directory if it exists
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
    os.makedirs(save_directory)

    rgb_directory = os.path.join(save_directory, "rgb")
    depth_directory = os.path.join(save_directory, "depth")

    rgb_images_array = []
    depth_images_array = []

    create_directory(rgb_directory)
    create_directory(depth_directory)
    
    zed, runtime_parameters, image_mat, depth_mat = init_zed(args.serial_number, args.exposure, args.gain)
    camera_info = zed.get_camera_information()
    left_cam_params = camera_info.camera_configuration.calibration_parameters.left_cam
    K = construct_camera_intrinsics(left_cam_params, args.width, args.height, args.camera_upsidedown)
    
    # Save camera intrinsics to file
    intrinsics_path = os.path.join(save_directory, "cam_K.txt")
    np.savetxt(intrinsics_path, K)
    print(f"Camera intrinsics saved to {intrinsics_path}")
    print(f"K matrix:\n{K}")

    frame_count = 0

    try:
        while True:
            frame_start_time = time.time()
            print(f'Frame: {frame_count}')
            rgb_image, depth_image = read_rgbd_frame(zed, runtime_parameters, image_mat, depth_mat, args.width, args.height, args.camera_upsidedown)
            rgb_images_array.append(rgb_image)
            depth_images_array.append(depth_image)
            frame_count += 1
            # Calculate elapsed time and sleep to match target FPS
            frame_elapsed_time = time.time() - frame_start_time
            sleep_time = target_frame_time - frame_elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Missed time at frame: {frame_count}")

    except KeyboardInterrupt:
        print("Stopping frame capture")
        # store rgb and depth images to the folder
        min_length = min(len(rgb_images_array), len(depth_images_array))

        for i in range(min_length):
            rgb_image = rgb_images_array[i]
            depth_image = depth_images_array[i]
            rgb_filename = os.path.join(
                rgb_directory, f"frame_{i:04d}.png"
            )
            depth_filename = os.path.join(
                depth_directory, f"frame_{i:04d}.png"
            )
            cv2.imwrite(rgb_filename, rgb_image)
            cv2.imwrite(depth_filename, depth_image)
            print(f'Stored frame {i} to {rgb_filename} and {depth_filename}')

    finally:
        # Release the camera
        zed.close()
        cv2.destroyAllWindows()
        print(f"Captured {frame_count} frames to {save_directory}")