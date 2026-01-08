# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from scipy.spatial.transform import Rotation
from generate_mask import save_box_mask_to_file
import nvdiffrast.torch as dr
import copy
import shutil


def main(rgbd_frames_directory, object_meshes, camera_to_robot, est_refine_iter=5, track_refine_iter=2, debug=0):
  """
  Main function for FoundationPose object tracking.
  
  Args:
    rgbd_frames_directory: Directory containing RGBD frames
    object_meshes: List of object mesh file paths
    camera_intrinsics_path: Path to camera intrinsics file
    camera_to_robot: Path to transformation from camera to robot file
    est_refine_iter: Number of refinement iterations for estimation (default: 5)
    track_refine_iter: Number of refinement iterations for tracking (default: 2)
    debug: Debug level (default: 1)
  """
  code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  set_logging_format()
  set_seed(0)

  save_directory = rgbd_frames_directory

  reader = YcbineoatReader(video_dir=rgbd_frames_directory, shorter_side=None, zfar=np.inf)

  if camera_to_robot:
    cal_transform = np.load(camera_to_robot, allow_pickle=True)
    try:
      transform_arr = np.atleast_1d(cal_transform)
      transform_arr = transform_arr[0]['agent1']['tcr']
      cal_transform = np.array(transform_arr)
      # convert 3x4 calibration matrix to 4x4 homogeneous
      cal_transform = np.vstack([cal_transform, np.array([0,0,0,1])])
    except:
      pass
  else:
      cal_transform = None
  

  all_poses = []
  debug_dir = f'{save_directory}/output'
  os.system(f'rm -rf {debug_dir}/*')

  for j, object_mesh in enumerate(object_meshes):
    mesh_file = object_mesh
    mesh = trimesh.load(mesh_file)

    os.system(f'mkdir -p {debug_dir}/track_vis_{j} {debug_dir}/ob_in_cam_{j}')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    poses = []

    for i in range(len(reader.color_files)-1):
      start_time = time.time()
      logging.info(f'i:{i}')
      # read_start_time = time.time()
      color = reader.get_color(i)
      depth = reader.get_depth(i)
      # read_end_time = time.time()
      # read_fps = 1.0 / (read_end_time - read_start_time)
      # print(f"Read FPS: {read_fps}")
      if i==0:
        save_box_mask_to_file(save_directory)
        mask = reader.get_mask(0).astype(bool)
        # register_start_time = time.time()
        breakpoint()
        pose = est.register(K=reader.K, 
                            rgb=color, 
                            depth=depth, 
                            ob_mask=mask, 
                            iteration=est_refine_iter)
        # register_end_time = time.time()
        # register_fps = 1.0 / (register_end_time - register_start_time)
        # print(f"Register FPS: {register_fps}")
        # breakpoint()
        if debug>=3:
          m = mesh.copy()
          m.apply_transform(pose)
          m.export(f'{debug_dir}/model_tf.obj')
          xyz_map = depth2xyzmap(depth, reader.K)
          valid = depth>=0.001
          pcd = toOpen3dCloud(xyz_map[valid], color[valid])
          o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
      else:
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)
      orig_pos = copy.deepcopy(pose)
      # write_start_time = time.time()
      os.makedirs(f'{debug_dir}/ob_in_cam_{j}', exist_ok=True)
      np.savetxt(f'{debug_dir}/ob_in_cam_{j}/{reader.id_strs[i]}.txt', pose.copy().reshape(4,4))
      if cal_transform is not None:
          pose_robot = cal_transform @ pose.copy().reshape(4,4)
          os.makedirs(f'{debug_dir}/ob_in_robot_{j}', exist_ok=True)
          np.savetxt(f'{debug_dir}/ob_in_robot_{j}/{reader.id_strs[i]}.txt', pose_robot)
      # write_end_time = time.time()
      # write_fps = 1.0 / (write_end_time - write_start_time)
      # print(f"Write FPS: {write_fps}")
      xyz = pose_robot[:3, 3]
      rot = pose_robot[:3, :3]
      quat = Rotation.from_matrix(rot).as_quat()
      quat = np.roll(quat, 1)
      oned_pose = np.concatenate((xyz, quat))
      poses.append(oned_pose)
      
      if debug>=1:
        center_pose = orig_pos@np.linalg.inv(to_origin)
        print(f"Orig pose: {Rotation.from_matrix(orig_pos[:3, :3]).as_euler('xyz', degrees=True)}")
        print(f"rot robot: {Rotation.from_matrix(rot).as_euler('xyz', degrees=True)}")
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)


      if debug>=2:
        os.makedirs(f'{debug_dir}/track_vis_{j}', exist_ok=True)
        imageio.imwrite(f'{debug_dir}/track_vis_{j}/{reader.id_strs[i]}.png', vis)
      # end_time = time.time()
      # tracking_fps = 1.0 / (end_time - start_time)
      # print(f"Tracking FPS: {tracking_fps}")
    all_poses.append(poses)

  if debug>=1:
    all_poses = np.array(all_poses)
    # breakpoint()
    np.save(f'{debug_dir}/poses_to_origin.npy', all_poses)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  parser.add_argument('--rgbd_frames_directory', type=str, required=True, help='Directory containing RGBD frames')
  parser.add_argument('--object_meshes', type=str, nargs='+', required=True, help='List of object mesh file paths')
  parser.add_argument('--camera_to_robot', type=str, default=f'{code_dir}/transforms_kitchen_new.npy', help='Path to camera-to-robot transformation file')
  args = parser.parse_args()
  main(
    args.rgbd_frames_directory,
    args.object_meshes,
    args.camera_to_robot
  ) 