#! /bin/bash
export PROJECT_ROOT="/juno/u/kedia/FoundationPose"
export TESTCASE="lego_20fps"
cd $PROJECT_ROOT
python src/obj_pose_track.py \
--rgb_seq_path $PROJECT_ROOT/$TESTCASE/color \
--depth_seq_path $PROJECT_ROOT/$TESTCASE/depth \
--mesh_path $PROJECT_ROOT/$TESTCASE/mesh/1x4.stl \
--init_mask_path $PROJECT_ROOT/$TESTCASE/0_mask.png \
--pose_output_path $PROJECT_ROOT/$TESTCASE/pose.npy \
--mask_visualization_path $PROJECT_ROOT/$TESTCASE/mask_visualization \
--bbox_visualization_path $PROJECT_ROOT/$TESTCASE/bbox_visualization \
--pose_visualization_path $PROJECT_ROOT/$TESTCASE/pose_visualization \
--cam_K "[[426.8704833984375, 0.0, 423.89471435546875], [0.0, 426.4277648925781, 243.5056915283203], [0.0, 0.0, 1.0]]" \
--activate_2d_tracker \
--apply_scale 0.01 \
--force_apply_color \
--apply_color "[0, 159, 237]" \
--est_refine_iter 10 \
--track_refine_iter 3