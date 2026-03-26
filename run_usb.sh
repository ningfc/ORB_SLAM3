#!/bin/bash

CONFIG=~/Code/CyperstereoSDK/ORB_SLAM3/cyperstereo_C72.yaml
DATA_ROOT=~/Code
if [ $# -gt 0 ]; then
    DATA_ROOT=$1
fi

./build/usb_camera ~/Code/ORB_SLAM3/Vocabulary/ORBvoc.txt ${CONFIG} ${DATA_ROOT}/left/ ${DATA_ROOT}/right/ ${DATA_ROOT}/imu/imu.csv

