#!/bin/bash

mkdir data

camera_params="../../st_handeye_eval/camera_info.yaml"
calib_test_dir="../../st_handeye_eval/cctag_test1"


command="../build/calibrate_cctag -u"
command="$command -c $camera_params"
command="$command --visual_inf_scale 1e-6"
command="$command --handpose_inf_scale_trans 1e-3"
command="$command --handpose_inf_scale_rot 1"
command="$command --robust_kernel_handpose Huber"
command="$command --robust_kernel_projection Huber"
command="$command --robust_kernel_handpose_delta 1e-3"
command="$command --robust_kernel_projection_delta 1e-3"
command="$command --save_hand2eye_visp data/hand2eye_visp.csv"
command="$command --save_hand2eye_dq data/hand2eye_dq.csv"
command="$command --save_hand2eye_graph data/hand2eye_graph.csv"
command="$command $calib_test_dir"

$command

