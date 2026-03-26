#!/bin/bash

ARGS="--ros-args -p perf_logging:=true -p filter_imu:=false"
ARGS=""
./Examples/ROS2/stereo_inertial_ros2 Vocabulary/ORBvoc.txt cyperstereo_C72.yaml ${ARGS} | tee run_ros2.log
