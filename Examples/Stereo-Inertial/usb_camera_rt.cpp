/**
* Realtime version of usb_camera.cpp:
* - initialize ORB-SLAM3 IMU_STEREO
* - receive stereo frames + IMU from Cyperstereo SDK
* - assemble per-frame IMU measurements and call TrackStereo
* - save trajectory on exit
*/

#include <signal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <System.h>

#include "cyperstereo_api.h"

using namespace std;

namespace {

constexpr double kGravity = 9.7887;
constexpr int kWidth = 752;
constexpr int kHeight = 480;

volatile sig_atomic_t g_running = 1;

void SignalHandler(int) {
    g_running = 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        cerr << "Usage: ./usb_camera_rt path_to_vocabulary path_to_settings [fps]" << endl;
        return 1;
    }

    int fps = 60;
    if (argc == 4) {
        fps = std::max(1, std::atoi(argv[3]));
    }

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = SignalHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, nullptr);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true);
    const float imageScale = SLAM.GetImageScale();

    std::shared_ptr<cyperstereo::uvc::device> device{nullptr};
    if (!cyperstereo::FindCyperstereoDevices(device)) {
        cerr << "No Cyperstereo device found." << endl;
        return 1;
    }

    cyperstereo::FrameInfo frame_info{};
    cyperstereo::uvc::set_device_mode(
        *device, kWidth, kHeight, static_cast<int>(cyperstereo::Format::YUYV), fps,
        [&frame_info](const void* data, std::function<void()> continuation) {
            cyperstereo::SetStreamData(frame_info, data, continuation);
        });
    cyperstereo::uvc::start_streaming(*device, 0);

    cv::namedWindow("left", cv::WINDOW_NORMAL);
    cv::namedWindow("right", cv::WINDOW_NORMAL);

    std::deque<ORB_SLAM3::IMU::Point> imu_buf;
    bool has_prev_imu = false;
    ORB_SLAM3::IMU::Point prev_imu(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.0);
    double last_imu_t_seen = -std::numeric_limits<double>::infinity();

    int frame_count = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (g_running && !SLAM.isShutDown()) {
        try {
            cyperstereo::WaitForStream(frame_info);
        } catch (const std::exception& e) {
            cerr << "WaitForStream timeout/error: " << e.what() << endl;
            continue;
        }

        double image_timestamp = 0.0;
        cv::Mat left_image;
        cv::Mat right_image;
        cyperstereo::IMUStreamData imu_data{};

        {
            std::lock_guard<std::mutex> lock(frame_info.mtx);
            image_timestamp = frame_info.framestream.image_timestamp;
            frame_info.framestream.left_image.copyTo(left_image);
            frame_info.framestream.right_image.copyTo(right_image);
            imu_data = frame_info.framestream.imu;
        }

        if (left_image.empty() || right_image.empty()) {
            continue;
        }

        for (int i = 0; i <= imu_data.imu_count; ++i) {
            const double t = imu_data.imu_timestamp[i];
            if (!std::isfinite(t) || t <= last_imu_t_seen) {
                continue;
            }
            const float gx = static_cast<float>(imu_data.gyro_x[i]);
            const float gy = static_cast<float>(imu_data.gyro_y[i]);
            const float gz = static_cast<float>(imu_data.gyro_z[i]);
            const float ax = static_cast<float>(imu_data.acc_x[i] * kGravity);
            const float ay = static_cast<float>(imu_data.acc_y[i] * kGravity);
            const float az = static_cast<float>(imu_data.acc_z[i] * kGravity);
            imu_buf.emplace_back(ax, ay, az, gx, gy, gz, t);
            last_imu_t_seen = t;
        }

        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vImuMeas.reserve(64);

        if (has_prev_imu && prev_imu.t < image_timestamp) {
            vImuMeas.push_back(prev_imu);
        }

        while (!imu_buf.empty() && imu_buf.front().t <= image_timestamp) {
            const ORB_SLAM3::IMU::Point imu = imu_buf.front();
            if (vImuMeas.empty() || imu.t > vImuMeas.back().t) {
                vImuMeas.push_back(imu);
            }
            prev_imu = imu;
            has_prev_imu = true;
            imu_buf.pop_front();
        }

        // Add one sample after image timestamp as right boundary for preintegration.
        if (!imu_buf.empty() && imu_buf.front().t > image_timestamp) {
            const ORB_SLAM3::IMU::Point& imu_next = imu_buf.front();
            if (vImuMeas.empty() || imu_next.t > vImuMeas.back().t) {
                vImuMeas.push_back(imu_next);
            }
        }

        if (vImuMeas.size() < 2) {
            if (frame_count % 30 == 0) {
                cout << std::fixed << std::setprecision(6)
                     << "insufficient IMU for frame ts=" << image_timestamp
                     << " imu_count=" << vImuMeas.size() << endl;
            }
            continue;
        }

        if (imageScale != 1.f) {
            int width = left_image.cols * imageScale;
            int height = left_image.rows * imageScale;
            cv::resize(left_image, left_image, cv::Size(width, height));
            cv::resize(right_image, right_image, cv::Size(width, height));
        }

        SLAM.TrackStereo(left_image, right_image, image_timestamp, vImuMeas);

        std::string overlay = "ts=" + std::to_string(image_timestamp) +
                              " imu=" + std::to_string(vImuMeas.size());
        cv::putText(left_image, overlay, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(255), 2);
        cv::imshow("left", left_image);
        cv::imshow("right", right_image);
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }

        ++frame_count;
        if (frame_count % 100 == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(now - t0).count();
            if (sec > 0.0) {
                cout << std::fixed << std::setprecision(2)
                     << "effective_fps=" << (100.0 / sec) << endl;
            }
            t0 = now;
        }
    }

    cyperstereo::uvc::stop_streaming(*device);
    SLAM.Shutdown();
    SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectoryUTM.txt");
    cv::destroyAllWindows();
    return 0;
}
