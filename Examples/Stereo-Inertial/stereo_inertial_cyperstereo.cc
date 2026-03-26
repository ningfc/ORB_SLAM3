/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez,
* José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
* University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms
* of the GNU General Public License as published by the Free Software Foundation, either
* version 3 of the License, or (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <System.h>

#include "cyperstereo_api.h"

using namespace std;

bool b_continue_session = true;

void exit_loop_handler(int s) {
    cout << "Finishing session" << endl;
    b_continue_session = false;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        cerr << endl
             << "Usage: ./stereo_inertial_cyperstereo path_to_vocabulary path_to_settings (trajectory_file_name)"
             << endl;
        return 1;
    }

    string file_name;
    if (argc == 4) {
        file_name = string(argv[argc - 1]);
    }

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    const double g = 9.7887;

    std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
    if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device)) {
        return 0;
    }

    cyperstereo::FrameInfo frame_info{};
    cyperstereo::uvc::set_device_mode(
        *cyperstereo_device, 752, 480, static_cast<int>(cyperstereo::Format::YUYV), 60,
        [&frame_info](const void *data, std::function<void()> continuation) {
            cyperstereo::SetStreamData(frame_info, data, continuation);
        });
    cyperstereo::uvc::start_streaming(*cyperstereo_device, 0);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true, 0, file_name);
    float imageScale = SLAM.GetImageScale();

    while (b_continue_session && !SLAM.isShutDown()) {
        try {
            cyperstereo::WaitForStream(frame_info);
        } catch (const std::exception &e) {
            std::cerr << "Cyperstereo stream timeout: " << e.what() << std::endl;
            continue;
        }

        double timestamp = 0.0;
        cv::Mat left_image;
        cv::Mat right_image;
        cyperstereo::IMUStreamData imu_data{};

        {
            std::lock_guard<std::mutex> lock(frame_info.mtx);
            timestamp = frame_info.framestream.image_timestamp;
            frame_info.framestream.left_image.copyTo(left_image);
            frame_info.framestream.right_image.copyTo(right_image);
            imu_data = frame_info.framestream.imu;
        }

        if (left_image.empty() || right_image.empty()) {
            continue;
        }

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vImuMeas.reserve(imu_data.imu_count + 1);
        for (int i = 0; i <= imu_data.imu_count; ++i) {
            double imu_timestamp = imu_data.imu_timestamp[i];
            double gyro_x = imu_data.gyro_x[i];
            double gyro_y = imu_data.gyro_y[i];
            double gyro_z = imu_data.gyro_z[i];
            double acc_x = imu_data.acc_x[i] * g;
            double acc_y = imu_data.acc_y[i] * g;
            double acc_z = imu_data.acc_z[i] * g;
            vImuMeas.emplace_back(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_timestamp);
        }

        if (imageScale != 1.f) {
            int width = left_image.cols * imageScale;
            int height = left_image.rows * imageScale;
            cv::resize(left_image, left_image, cv::Size(width, height));
            cv::resize(right_image, right_image, cv::Size(width, height));
        }

        SLAM.TrackStereo(left_image, right_image, timestamp, vImuMeas);
    }

    cyperstereo::uvc::stop_streaming(*cyperstereo_device);
    SLAM.Shutdown();

    return 0;
}
