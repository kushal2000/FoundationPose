# FoundationPose (SimToolReal Fork)

Fork of [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) adapted for the [SimToolReal](https://simtoolreal.github.io/) perception pipeline. This repo provides three user-facing scripts for recording RGB-D videos, extracting 6D object poses, and running real-time pose tracking with ROS.

## Installation

### 1. Create Conda Environment

```bash
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**PyTorch3D** must be installed from source (not available on PyPI for all CUDA versions):
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

**ZED SDK** (required for `record_video.py` and `live_tracking_with_ros.py`):
- Install from [stereolabs.com](https://www.stereolabs.com/developers/release)
- Then: `pip install pyzed`

**ROS** (required for `live_tracking_with_ros.py`):
- Install ROS Noetic in the same conda environment, or source it before running.

### 3. Build C++ Extensions

```bash
bash build_all_conda.sh
```

This builds `mycpp` (pose clustering used by FoundationPose).

### 4. Download Model Weights

Download the FoundationPose pretrained weights from the [original repo](https://github.com/NVlabs/FoundationPose#model-weights) and place them in `weights/`:

```
weights/
  ├── 2023-10-28-18-33-37/   # Scorer model
  └── 2024-01-11-20-02-45/   # Refiner model
```

## Scripts

### Script 1: Record Video

Record an RGB-D video from a ZED stereo camera.

```bash
python record_video.py \
    --save_dir recordings/ \
    --serial_number 15107 \
    --fps 30
```

Press `Ctrl+C` to stop recording. Output directory structure:
```
recordings/<timestamp>/
  ├── rgb/          # RGB frames as PNGs
  ├── depth/        # Depth frames as 16-bit PNGs (mm)
  ├── cam_K.txt     # 3x3 camera intrinsics
  └── rgb.mp4       # RGB video
```

**Options:** `--width` (960), `--height` (540), `--exposure` (25), `--gain` (40), `--camera_upsidedown`

### Script 2: Extract Poses

Extract 6D object poses from a recorded RGB-D video and an object mesh.

```bash
python extract_poses.py \
    --video_dir recordings/<timestamp>/ \
    --mesh_path /path/to/object.obj \
    --output_path poses.json
```

On the first frame, an interactive window opens where you click 4 corners of a bounding box around the object for SAM-based segmentation. FoundationPose then tracks the object through all frames.

**Output format** (`poses.json`):
```json
{
  "poses_cam": [
    [x, y, z, qx, qy, qz, qw],
    ...
  ],
  "poses_robot": [...]  // only if --calibration is provided
}
```

**Options:** `--calibration <T_RC.npy>` (camera-to-robot transform), `--est_refine_iter` (5), `--track_refine_iter` (2), `--debug` (0/1/2)

### Script 3: Live Tracking with ROS

Run real-time 6D pose tracking from a ZED camera and publish poses to ROS topics.

```bash
python live_tracking_with_ros.py \
    --mesh_path /path/to/object.obj \
    --calibration /path/to/T_RC.npy
```

**Published ROS topics:**
- `camera_frame/current_object_pose` (`PoseStamped`) -- pose in camera frame
- `robot_frame/current_object_pose` (`PoseStamped`) -- pose in robot frame (via `T_RC`)

**Options:** `--serial_number` (15107), `--fps` (40), `--camera_upsidedown`, `--width` (960), `--height` (540), `--debug` (0/1)

## Configuration

### Camera Intrinsics

Camera intrinsics are automatically read from the ZED SDK for live scripts (`record_video.py`, `live_tracking_with_ros.py`). For offline processing (`extract_poses.py`), they are loaded from `cam_K.txt` in the video directory.

### Camera-to-Robot Calibration

The `--calibration` argument accepts a 4x4 homogeneous transform `T_RC` (robot-from-camera) as a `.npy` or `.txt` file. This transform converts poses from camera frame to robot frame: `pose_robot = T_RC @ pose_cam`.

### Object Meshes

Object meshes should be `.obj` files with units in **meters**. The mesh origin defines the object coordinate frame for the estimated poses.
