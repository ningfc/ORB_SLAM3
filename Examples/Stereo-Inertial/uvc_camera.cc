// Copyright 2018 Slightech Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <mutex>
#include "string"
#include <queue>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Thirdparty/usb/uvc/uvc.h"

#include <System.h>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <condition_variable>

#define g 9.8f
#define BMI088_ACCEL_24G_SEN 0.000732421875f
#define BMI088_ACCEL_12G_SEN 0.0003662109375f
#define BMI088_ACCEL_6G_SEN 0.00018310546875f
#define BMI088_ACCEL_3G_SEN 0.000091552734375f 
#define BMI088_GYRO_2000_SEN 0.0010652644178602f 
#define BMI088_GYRO_1000_SEN 0.0005326322089301215f 
#define BMI088_GYRO_500_SEN 0.0002663161044650608f 
#define BMI088_GYRO_250_SEN 0.0001331580522325304f 
#define BMI088_GYRO_125_SEN 0.00006657902611626519f 
float BMI088_ACCEL_SEN = BMI088_ACCEL_3G_SEN;
float BMI088_GYRO_SEN = BMI088_GYRO_2000_SEN;


std::queue<pair<double,std::vector<Eigen::Vector3d>> > IMU;
std::queue<pair<double,std::vector<cv::Mat>> > IMAGE;
std::mutex m_buf;
double current_time = -1;
std::condition_variable con;
// ORB_SLAM3::System SLAM("../Vocabulary/ORBvoc.txt","../usb_camera.yaml",ORB_SLAM3::System::IMU_STEREO, true, 0, "");

ORB_SLAM3::System *SLAM;

void InputIMU( const double timestamp, const Eigen::Vector3d& accl, const Eigen::Vector3d& gyro)
{
    m_buf.lock();
    std::vector<Eigen::Vector3d> IMUTEMP;
    IMUTEMP.push_back(accl);
    IMUTEMP.push_back(gyro);
    IMU.push(make_pair(timestamp,IMUTEMP));
    m_buf.unlock();
    con.notify_one();
}

void InputIMAGE(const cv::Mat& cam0_img,
                const cv::Mat& cam1_img,
                double time)
{
    m_buf.lock();
    std::vector<cv::Mat> IMAGETEMP;
    IMAGETEMP.push_back(cam0_img);
    IMAGETEMP.push_back(cam1_img);
    IMAGE.push(make_pair(time,IMAGETEMP));
    m_buf.unlock();
    con.notify_one();

}


std::vector<std::pair<std::vector<std::pair<double,std::vector<Eigen::Vector3d>> >, std::vector<std::pair<double,std::vector<cv::Mat>> > >>
getMeasurements()
{
    std::vector<std::pair<std::vector<std::pair<double,std::vector<Eigen::Vector3d>> >, std::vector<std::pair<double,std::vector<cv::Mat>> > >>  measurements;
    while (true)
    {

        if(IMAGE.empty()||IMU.empty())
        {
          //cout<<"wait for data"<<endl;
          return measurements;
        }

        if (!(IMU.back().first > IMAGE.front().first))
        {
            // cout<<"wait for imu, only should happen at the beginning";
            return measurements;
        }
        if (!(IMU.front().first < IMAGE.front().first))
        {
            cout<<"throw img, only should happen at the beginning";
            IMAGE.pop();
            continue;
        }
        std::vector<std::pair<double,std::vector<Eigen::Vector3d>> > IMUs;
        while (IMU.front().first < IMAGE.front().first)
        {
            IMUs.emplace_back(IMU.front());
            IMU.pop();
        }
        IMUs.emplace_back(IMU.front());

        if (IMUs.empty())
           cout<<"no imu between two image";

        vector<pair<double,std::vector<cv::Mat>> > IMAGES;
        IMAGES.push_back(IMAGE.front());
        IMAGE.pop();
        measurements.push_back(make_pair(IMUs,IMAGES));

    }
    cout<<measurements.size()<<endl;
    return measurements;
}

void process()
{
    
    while (true)
       {
        std::vector<std::pair<std::vector<std::pair<double,std::vector<Eigen::Vector3d>> >, std::vector<std::pair<double,std::vector<cv::Mat>> > >> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        
        for (auto &measurement : measurements)
        {
           std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
           vImuMeas.clear();
           auto img = measurement.second.front();
           double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
           for (auto &imu_msg : measurement.first)
           {
               double t = imu_msg.first;
               double img_t = img.first;
               if (t <= img_t)
               {
                   if (current_time < 0)
                       current_time = t;
                   double dt = t - current_time;
                   current_time = t;
                   dx = imu_msg.second.front()[0];
                   dy = imu_msg.second.front()[1];
                   dz = imu_msg.second.front()[2];
                   rx = imu_msg.second.back()[0];
                   ry = imu_msg.second.back()[1];
                   rz = imu_msg.second.back()[2];
                   vImuMeas.push_back(ORB_SLAM3::IMU::Point(dx, dy, dz, rx, ry, rz, current_time));
                  // std::cout.setf(std::ios::fixed, std::ios::floatfield);
                  // std::cout.precision(6);
                  // cout<<"imu: "<< current_time << " " << dx << " " << dy << " " << dz <<  " " << rx << " " << ry << " " << rz <<endl;

               }
               else
               {
                   double dt_1 = img_t - current_time;
                   double dt_2 = t - img_t;
                   current_time = img_t;
                   double w1 = dt_2 / (dt_1 + dt_2);
                   double w2 = dt_1 / (dt_1 + dt_2);
                   dx = w1 * dx + w2 * imu_msg.second.front()[0];
                   dy = w1 * dy + w2 * imu_msg.second.front()[1];
                   dz = w1 * dz + w2 * imu_msg.second.front()[2];
                   rx = w1 * rx + w2 * imu_msg.second.back()[0];
                   ry = w1 * ry + w2 * imu_msg.second.back()[1];
                   rz = w1 * rz + w2 * imu_msg.second.back()[2];
                   vImuMeas.push_back(ORB_SLAM3::IMU::Point(dx, dy, dz, rx, ry, rz, current_time));
                  //  std::cout.setf(std::ios::fixed, std::ios::floatfield);
                  //  std::cout.precision(6);
                  //  cout<<"imu: "<< current_time << " " << dx << " " << dy << " " << dz <<  " " << rx << " " << ry << " " << rz <<endl;
               }
            }
          //  cv::imshow("left ",img.second.front());
          //  cv::imshow("right ",img.second.back());
          //  cv::waitKey(1);
          //  std::cout.precision(6);
          //  cout<<"imagetime"<<img.first<<endl;
          SLAM->TrackStereo(img.second.front(), img.second.back(), img.first, vImuMeas);
          // SLAM->TrackStereo(img.second.front(), img.second.back(), img.first);
        }
       }
}




struct frame {
  const void *data = nullptr;
  std::function<void()> continuation = nullptr;
  frame() {
  }
  ~frame() {
    data = nullptr;
    if (continuation) {
      continuation();
      continuation = nullptr;
    }
  }
};

enum class Format : int {
    /** Greyscale, 8 bits per pixel */
    GREY,
    /** YUV 4:2:2, 16 bits per pixel */
    YUYV,
    /** BGR 8:8:8, 24 bits per pixel */
    BGR888,
    /** RGB 8:8:8, 24 bits per pixel */
    RGB888,
};


// MYNTEYE_USE_NAMESPACE

size_t nFrames = 0;
int main(int argc, char *argv[]) {
  ORB_SLAM3::System SLAM_("../Vocabulary/ORBvoc.txt","../usb_camera.yaml",ORB_SLAM3::System::IMU_STEREO, true, 0, "");
  SLAM = &SLAM_;

  std::thread measurement_process{process};

  std::vector<std::shared_ptr<mynteye::uvc::device>> mynteye_devices;

  auto context = mynteye::uvc::create_context();
  auto devices = mynteye::uvc::query_devices(context);
  if (devices.size() <= 0) {
    std::cout << "No devices :(" << std::endl;
    return 1;
  }

  for (auto &&device : devices) {
    auto vid = mynteye::uvc::get_vendor_id(*device);
    if (vid == MYNTEYE_VID) {
      mynteye_devices.push_back(device);
    }
  }

  // std::string dashes(80, '-');

  size_t n = mynteye_devices.size();
  if (n <= 0) {
    std::cout << "No MYNT EYE devices :(" << std::endl;
    return 1;
  }

  std::cout  << "usb devices: " << std::endl;
  for (size_t i = 0; i < n; i++) {
    auto device = mynteye_devices[i];
    auto name = mynteye::uvc::get_video_name(*device);
    auto vid = mynteye::uvc::get_vendor_id(*device);
    auto pid = mynteye::uvc::get_product_id(*device);
    std::cout << "  index: " << i << ", name: " << name << ", vid: 0x"
              << std::hex << vid << ", pid: 0x" << std::hex << pid << std::endl;
  }

  std::shared_ptr<mynteye::uvc::device> device = nullptr;
  if (n <= 1) {
    device = mynteye_devices[0];
    std::cout << "Only one MYNT EYE device, select index: 0" << std::endl;
  } else {
    while (true) {
      size_t i;
      std::cout << "There are " << n << " MYNT EYE devices, select index: " << std::endl;
      std::cin >> i;
      if (i >= n) {
        std::cout << "Index out of range :(" << std::endl;
        continue;
      }
      device = mynteye_devices[i];
      break;
    }
  }

  std::mutex mtx;
  std::condition_variable cv;

  std::shared_ptr<frame> frame = nullptr;
  const auto frame_ready = [&frame]() { return frame != nullptr; };
  const auto frame_empty = [&frame]() { return frame == nullptr; };

  mynteye::uvc::set_device_mode(
      *device, 752, 480, static_cast<int>(Format::YUYV), 60,
// #else
//       *device, 1280, 400, static_cast<int>(Format::BGR888), 20,
// #endif
      [&mtx, &cv, &frame, &frame_ready](
          const void *data, std::function<void()> continuation) {
        // reinterpret_cast<const std::uint8_t *>(data);
        std::unique_lock<std::mutex> lock(mtx);
        if (frame == nullptr) {
          frame = std::make_shared<struct frame>();
        } else {
          if (frame->continuation) {
            frame->continuation();
          }
        }
        frame->data = data;  // not copy here
        frame->continuation = continuation;
        if (frame_ready())
          cv.notify_one();
      });


  mynteye::uvc::start_streaming(*device, 0);

  double tic, fps = 0;
  int fps_count = 0;
  double tic_total = 0;
  cv::Mat left(480, 752, CV_8U);
	cv::Mat right(480, 752, CV_8U);
  cv::Mat imu(1, 752 * 2, CV_8U);
  unsigned char* left_p = left.ptr<unsigned char>(0, 0);
	unsigned char* right_p = right.ptr<unsigned char>(0, 0);
  unsigned char* imu_p = imu.ptr<unsigned char>(0, 0);
  double acc_x[4];
	double acc_y[4];
	double acc_z[4];
	double gyro_x[4];
	double gyro_y[4];
	double gyro_z[4];
  double temperature[4];
	double imu_timestamp[4];
  double image_sensor_time = 0;
  double imu_sensor_time[4] = {0};
  double last_image_sensor_time = 0;
  double last_imu_sensor_time = 0;
  int last_imu_count_s = 0;
  int last_image_count_s = 0;
  double last_temperature;
  tic = static_cast<double>(cv::getTickCount());

  while (true) {
    std::unique_lock<std::mutex> lock(mtx);
    if (frame_empty()) {
      if (!cv.wait_for(lock, std::chrono::seconds(2), frame_ready))
        throw std::runtime_error("Timeout waiting for frame.");
    }

    cv::Mat img(480, 752, CV_8UC2, const_cast<void *>(frame->data));
    unsigned char* frame_p = img.ptr<unsigned char>(0, 0);
    for (int i = 0; i < 752 * 480; i++) {
      left_p[i] = frame_p[2 * i];
      right_p[i] = frame_p[2 * i + 1];
    }
		for (int i = 752 * 479 * 2, j = 0; i < 752 * 480 * 2; ++i, ++j) {
			imu_p[j] = frame_p[i];
		}

    //image data
    // double count = ((int16_t)((imu.at<uchar>(0, 1)) << 8) | imu.at<uchar>(0, 0));
    // double image_count_begin_ms = ((int16_t)((imu.at<uchar>(0, 3)) << 8) | imu.at<uchar>(0, 2));
    // double image_count_begin_s = ((int16_t)((imu.at<uchar>(0, 5)) << 8) | imu.at<uchar>(0, 4));
    double image_count_ms = ((int16_t)((imu.at<uchar>(0, 7)) << 8) | imu.at<uchar>(0, 6));
    double image_count_s = ((int16_t)((imu.at<uchar>(0, 9)) << 8) | imu.at<uchar>(0, 8));
    if (image_count_s < last_image_count_s)
      image_count_s += 43200;
    image_sensor_time = image_count_s + image_count_ms / 10000;
    last_image_count_s = image_count_s;
    if (image_sensor_time - last_image_sensor_time < 0.015) {
      std::cout << "image warn image_sensor_time " << image_sensor_time << std::endl;
      std::cout << "image warn last_image_sensor_time " << last_image_sensor_time << std::endl;
    }
    if (image_sensor_time - last_image_sensor_time > 0.025) {
      std::cout << "image warn image_sensor_time " << image_sensor_time
                << std::endl;
      std::cout << "image warn last_image_sensor_time "
                << last_image_sensor_time << std::endl;
    }
    last_image_sensor_time = image_sensor_time;
    
    // vImuMeas.clear();
    //imu data
    for (int i = 0; i < 4; ++i) {
      double imu_count_ms = ((int16_t)((imu.at<uchar>(0, 11 + i * 18)) << 8) | imu.at<uchar>(0, 10 + i * 18));
      double imu_count_s = ((int16_t)((imu.at<uchar>(0, 13 + i * 18)) << 8) | imu.at<uchar>(0, 12 + i * 18));
      if (imu_count_s < last_imu_count_s)
        imu_count_s += 43200;
      imu_sensor_time[i] = imu_count_s + imu_count_ms / 10000;
      acc_x[i] = ((int16_t)((imu.at<uchar>(0, 15 + i * 18)) << 8) | imu.at<uchar>(0, 14 + i * 18))* BMI088_ACCEL_SEN;
		  acc_y[i] = ((int16_t)((imu.at<uchar>(0, 17 + i * 18)) << 8) | imu.at<uchar>(0, 16 + i * 18))* BMI088_ACCEL_SEN;
		  acc_z[i] = ((int16_t)((imu.at<uchar>(0, 19 + i * 18)) << 8) | imu.at<uchar>(0, 18 + i * 18))* BMI088_ACCEL_SEN;

      gyro_x[i] = ((int16_t)((imu.at<uchar>(0, 21 + i * 18)) << 8) | imu.at<uchar>(0, 20 + i * 18))* BMI088_GYRO_SEN;
		  gyro_y[i] = ((int16_t)((imu.at<uchar>(0, 23 + i * 18)) << 8) | imu.at<uchar>(0, 22 + i * 18))* BMI088_GYRO_SEN;
		  gyro_z[i] = ((int16_t)((imu.at<uchar>(0, 25 + i * 18)) << 8) | imu.at<uchar>(0, 24 + i * 18))* BMI088_GYRO_SEN;
      temperature[i] = ((int16_t)((int16_t)((imu.at<uchar>(0, 27 + i * 18)) << 8) | imu.at<uchar>(0, 26 + i * 18)));
      if (temperature[i] > 1023)
        temperature[i] = temperature[i] - 2048;
      else
        temperature[i] = temperature[i];
      temperature[i] = temperature[i] * 0.125 + 23;
      if (i == 3 && abs(imu_sensor_time[3] - imu_sensor_time[2]) > 0.01) {
	      continue;
      }	
      if (imu_sensor_time[i] - last_imu_sensor_time > 0.0075) {
        std::cout << "imu warn imu_sensor_time " << imu_sensor_time[i]
                  << std::endl;
        std::cout << "imu warn last_imu_sensor_time " << last_imu_sensor_time
                  << std::endl;
      }       
      last_imu_sensor_time = imu_sensor_time[i];
      last_imu_count_s = imu_count_s;
      // std::ofstream foutC("./imu/imu.csv", std::ios::app);
      // foutC.setf(std::ios::fixed, std::ios::floatfield);
      // foutC.precision(4);
      // foutC << imu_sensor_time[i] << ",";
      // foutC.precision(6);
      // foutC << gyro_x[i] << ","
      //       << gyro_y[i] << ","
      //       << gyro_z[i] << ","
      //       << acc_x[i] * 9.8 << ","
      //       << acc_y[i] * 9.8 << ","
      //       << acc_z[i] * 9.8 
      //       << std::endl;
      // foutC.close();
      // std::cout << " imu_sensor_time " << imu_sensor_time[i] << std::endl;
		  // std::cout << " acc_x: " << acc_x[i] << std::endl;
		  // std::cout << " acc_y: " << acc_y[i] << std::endl;
		  // std::cout << " acc_z: " << acc_z[i] << std::endl;
		  // std::cout << " gyro_x: " << gyro_x[i] << std::endl;
		  // std::cout << " gyro_y: " << gyro_y[i] << std::endl;
		  // std::cout << " gyro_z: " << gyro_z[i] << std::endl;
      // std::cout << " temperature: " << temperature[i] << std::endl;
      Eigen::Vector3d acc(acc_x[i] * 9.82, acc_y[i] * 9.82, acc_z[i] * 9.82);
      Eigen::Vector3d gyro(gyro_x[i], gyro_y[i], gyro_z[i]);
      InputIMU(imu_sensor_time[i], acc, gyro);
      // vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc_x[i] * 9.8, acc_y[i] * 9.8, acc_z[i] * 9.8, gyro_x[i], gyro_y[i], gyro_z[i], imu_sensor_time[i]));    
    }
    if (fps_count % 6 == 0) {
      // cv::imwrite("./left/" + std::to_string(static_cast<int>(image_sensor_time * 10000)) + ".png", left);
      // cv::imwrite("./right/" + std::to_string(static_cast<int>(image_sensor_time * 10000)) + ".png", right);
      cv::Mat IMAGELEFT=left.clone();
      cv::Mat IMAGERIGHT=right.clone();
      InputIMAGE(IMAGELEFT, IMAGERIGHT, image_sensor_time);
    }
    
    // SLAM.TrackStereo(left, right, image_sensor_time, vImuMeas);

    frame = nullptr;

    fps_count++;
    if (fps_count == 60) {
      tic_total = static_cast<double>(cv::getTickCount() - tic);
      tic = static_cast<double>(cv::getTickCount());
      fps_count = 0;
      // std::cout << "fps" << 60.0 * cv::getTickFrequency() / tic_total << std::endl;
    }
  }

  // SLAM.Shutdown();
  // SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
  // SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");

  mynteye::uvc::stop_streaming(*device);
  // cv::destroyAllWindows();
  return 0;
}



