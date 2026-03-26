/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira,
* Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós,
* University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it
* under the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License along
* with ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
*/

#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <opencv2/core/core.hpp>

#include "System.h"
#include "ImuTypes.h"

using std::placeholders::_1;
using std::placeholders::_2;

class StereoInertialNode : public rclcpp::Node
{
public:
    StereoInertialNode(ORB_SLAM3::System* pSLAM)
        : Node("stereo_inertial_orbslam3"),
          mpSLAM(pSLAM)
    {
        left_topic_ = this->declare_parameter<std::string>("left_topic", "/cam0/image_raw");
        right_topic_ = this->declare_parameter<std::string>("right_topic", "/cam1/image_raw");
        imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/imu0");
        sync_queue_size_ = this->declare_parameter<int>("sync_queue_size", 10);
        max_stereo_queue_ = this->declare_parameter<int>("max_stereo_queue", 20);
        use_clahe_ = this->declare_parameter<bool>("use_clahe", false);
        debug_timestamps_ = this->declare_parameter<bool>("debug_timestamps", true);
        drop_zero_timestamps_ = this->declare_parameter<bool>("drop_zero_timestamps", true);
        max_stereo_dt_ = this->declare_parameter<double>("max_stereo_dt", 1e-4);
        max_imu_delay_ = this->declare_parameter<double>("max_imu_delay", 0.02);
        perf_logging_ = this->declare_parameter<bool>("perf_logging", false);

        filter_imu_ = this->declare_parameter<bool>("filter_imu", true);
        max_ready_queue_ = this->declare_parameter<int>("max_ready_queue", 10);

        rclcpp::SensorDataQoS qos;

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos,
            std::bind(&StereoInertialNode::GrabImu, this, _1));

        left_sub_.subscribe(this, left_topic_, qos.get_rmw_qos_profile());
        right_sub_.subscribe(this, right_topic_, qos.get_rmw_qos_profile());

        sync_ = std::make_shared<Sync>(SyncPolicy(sync_queue_size_), left_sub_, right_sub_);
        sync_->registerCallback(std::bind(&StereoInertialNode::GrabStereo, this, _1, _2));

        if (use_clahe_)
        {
            clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
        }

        RCLCPP_INFO(this->get_logger(),
                    "ORB-SLAM3 Stereo-Inertial ROS2 node started. Left: %s Right: %s IMU: %s",
                    left_topic_.c_str(), right_topic_.c_str(), imu_topic_.c_str());

        if (debug_timestamps_)
        {
            RCLCPP_INFO(this->get_logger(),
                        "Timestamp debug enabled. Clock type: %d",
                        static_cast<int>(this->get_clock()->get_clock_type()));
        }

        running_.store(true);
        processing_thread_ = std::thread(&StereoInertialNode::ProcessQueue, this);
        consumer_thread_ = std::thread(&StereoInertialNode::ConsumerLoop, this);
    }

    ~StereoInertialNode() override
    {
        running_.store(false);
        // Wake consumer if waiting
        ready_cv_.notify_all();
        if (processing_thread_.joinable())
        {
            processing_thread_.join();
        }
        if (consumer_thread_.joinable())
        {
            consumer_thread_.join();
        }
    }

private:
    using SyncPolicy = message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;

    static bool IsFinite(double v)
    {
        return std::isfinite(v);
    }

    static bool IsFiniteVec3(const geometry_msgs::msg::Vector3& v)
    {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    }

    static bool IsFiniteImuPoint(const ORB_SLAM3::IMU::Point& p)
    {
        return std::isfinite(p.t) &&
               std::isfinite(p.a.x()) && std::isfinite(p.a.y()) && std::isfinite(p.a.z()) &&
               std::isfinite(p.w.x()) && std::isfinite(p.w.y()) && std::isfinite(p.w.z());
    }

    static bool IsSupportedImageEncoding(const std::string& enc)
    {
        using namespace sensor_msgs::image_encodings;
        return enc == MONO8 || enc == MONO16 || enc == BGR8 || enc == RGB8;
    }

    static bool IsReasonableImu(const sensor_msgs::msg::Imu::ConstSharedPtr& imu)
    {
        const double ax = imu->linear_acceleration.x;
        const double ay = imu->linear_acceleration.y;
        const double az = imu->linear_acceleration.z;
        const double gx = imu->angular_velocity.x;
        const double gy = imu->angular_velocity.y;
        const double gz = imu->angular_velocity.z;
        const double a_norm = std::sqrt(ax * ax + ay * ay + az * az);
        const double g_norm = std::sqrt(gx * gx + gy * gy + gz * gz);
        // Reasonable thresholds: accel < 200 m/s^2, gyro < 2000 rad/s
        if (!IsFinite(a_norm) || !IsFinite(g_norm)) return false;
        if (a_norm > 200.0 || g_norm > 2000.0) return false;
        return true;
    }

    static bool IsImageNonBlank(const cv::Mat& m)
    {
        if (m.empty()) return false;
        cv::Scalar mean, stddev;
        cv::meanStdDev(m, mean, stddev);
        // If standard deviation is very small, image is likely blank or stuck
        const double thresh = 1.0;
        return std::isfinite(stddev[0]) && stddev[0] >= thresh;
    }

    void GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
    {
        const double t_imu = rclcpp::Time(msg->header.stamp).seconds();
        if (!IsFinite(t_imu))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "IMU stamp is NaN/Inf. Dropping.");
            return;
        }
        if (debug_timestamps_ && t_imu > 1e9)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "IMU stamp is very large (%.9f). Possibly nanoseconds-as-seconds?",
                                 t_imu);
        }
        if (drop_zero_timestamps_ && t_imu <= 0.0)
        {
            if (debug_timestamps_)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "IMU stamp is zero/invalid (%.9f). Dropping.", t_imu);
            }
            return;
        }

        if (last_imu_stamp_recv_ >= 0.0 && t_imu <= last_imu_stamp_recv_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "IMU time is non-increasing: prev=%.9f now=%.9f (delta=%.9f). Dropping sample.",
                                 last_imu_stamp_recv_, t_imu, t_imu - last_imu_stamp_recv_);
            return;
        }

        if (!IsFiniteVec3(msg->linear_acceleration) || !IsFiniteVec3(msg->angular_velocity))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "IMU data contains NaN/Inf. Dropping.");
            return;
        }
        last_imu_stamp_recv_ = t_imu;

        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buf_.push_back(msg);
        // printf("Received IMU with t=%.6f, acc=(%.3f, %.3f, %.3f) gyr=(%.3f, %.3f, %.3f) buf=%zu\n",
        //        t_imu,
        //        msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
        //        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
        //        imu_buf_.size());
        if (imu_buf_.size() > 2000)
        {
            imu_buf_.pop_front();
        }
        if (debug_timestamps_)
        {
            const double t_now = this->get_clock()->now().seconds();
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "IMU recv: stamp=%.6f now=%.6f buf=%zu",
                                 t_imu, t_now, imu_buf_.size());
        }
    }

    void GrabStereo(const sensor_msgs::msg::Image::ConstSharedPtr left_msg,
                    const sensor_msgs::msg::Image::ConstSharedPtr right_msg)
    {
        const double t_left = rclcpp::Time(left_msg->header.stamp).seconds();
        const double t_right = rclcpp::Time(right_msg->header.stamp).seconds();

        if (!IsFinite(t_left) || !IsFinite(t_right))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Image stamp is NaN/Inf. Dropping.");
            return;
        }

        if (debug_timestamps_ && (t_left > 1e9 || t_right > 1e9))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "Image stamp is very large (left=%.9f right=%.9f). Possibly nanoseconds-as-seconds?",
                                 t_left, t_right);
        }

        if (drop_zero_timestamps_ && (t_left <= 0.0 || t_right <= 0.0))
        {
            if (debug_timestamps_)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "Image stamp is zero/invalid (left=%.9f right=%.9f). Dropping.",
                                     t_left, t_right);
            }
            return;
        }

        if (last_img_stamp_recv_ >= 0.0 && t_left < last_img_stamp_recv_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Image time went backwards: prev=%.9f now=%.9f (delta=%.9f)",
                                 last_img_stamp_recv_, t_left, t_left - last_img_stamp_recv_);
        }
        last_img_stamp_recv_ = t_left;

        if (std::abs(t_left - t_right) > max_stereo_dt_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "Stereo timestamps not aligned: left %.6f right %.6f",
                                 t_left, t_right);
            return;
        }

        if (left_msg->width == 0 || left_msg->height == 0 ||
            right_msg->width == 0 || right_msg->height == 0)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Image has zero width/height. Dropping.");
            return;
        }

        if (left_msg->width != right_msg->width || left_msg->height != right_msg->height)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Left/right image size mismatch: left=%ux%u right=%ux%u. Dropping.",
                                 left_msg->width, left_msg->height,
                                 right_msg->width, right_msg->height);
            return;
        }

        if (left_msg->encoding != right_msg->encoding)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Left/right encoding mismatch: left=%s right=%s. Dropping.",
                                 left_msg->encoding.c_str(), right_msg->encoding.c_str());
            return;
        }

        if (!IsSupportedImageEncoding(left_msg->encoding))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Unsupported image encoding: %s. Dropping.",
                                 left_msg->encoding.c_str());
            return;
        }

        if (debug_timestamps_)
        {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Stereo recv: left=%.6f right=%.6f dt=%.6f queue=%zu",
                                 t_left, t_right, std::abs(t_left - t_right),
                                 stereo_queue_.size());
        }

        std::lock_guard<std::mutex> lock(stereo_mutex_);
        if (stereo_queue_.size() >= static_cast<size_t>(max_stereo_queue_))
        {
            stereo_queue_.pop_front();
        }
        stereo_queue_.push_back({left_msg, right_msg, t_left, this->get_clock()->now().seconds()});
    }

    void ProcessQueue()
    {
        while (rclcpp::ok() && running_.load())
        {
            StereoPair pair;
            bool drop_stereo_pair = false;
            const char* drop_reason = nullptr;
            {
                std::lock_guard<std::mutex> lock(stereo_mutex_);
                if (stereo_queue_.empty())
                {
                    if (debug_timestamps_)
                    {
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                                             "Stereo queue empty. IMU buf=%zu",
                                             imu_buf_.size());
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                pair = stereo_queue_.front();
                if (debug_timestamps_)
                {
                    const double now = this->get_clock()->now().seconds();
                    const double enq_delay = now - pair.enqueued;
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                         "Stereo pair queued %.6f s ago (enqueue_delay=%.6f) stereo_queue=%zu imu_buf=%zu",
                                         pair.timestamp, enq_delay, stereo_queue_.size(), imu_buf_.size());
                }
            }

            std::vector<ORB_SLAM3::IMU::Point> imu_meas;
            std::chrono::steady_clock::time_point proc_start;
            if (perf_logging_) proc_start = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(imu_mutex_);
                if (imu_buf_.empty())
                {
                    if (debug_timestamps_)
                    {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                             "IMU buf empty. Waiting. Stereo queue=%zu",
                                             stereo_queue_.size());
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                const double t_imu_last = rclcpp::Time(imu_buf_.back()->header.stamp).seconds();
                if (pair.timestamp > t_imu_last)
                {
                    if (debug_timestamps_)
                    {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                             "IMU not up to image. img=%.6f imu_last=%.6f lag=%.6f",
                                             pair.timestamp, t_imu_last, pair.timestamp - t_imu_last);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                const double t_imu_first = rclcpp::Time(imu_buf_.front()->header.stamp).seconds();
                if (pair.timestamp < t_imu_first)
                {
                    // This image is older than the earliest IMU still available.
                    // No IMU integration slice can be built for it anymore, so drop it.
                    drop_stereo_pair = true;
                    drop_reason = "stereo frame older than earliest IMU";
                }

                if (drop_stereo_pair)
                {
                    // Defer actual pop to outside imu lock.
                }
                else
                {

                    // Keep one boundary IMU sample from the previous frame interval.
                    // ORB-SLAM3 preintegration benefits from having measurements spanning
                    // [t_prev_frame, t_current_frame], not only strictly-new samples.
                    double last_added_t = -std::numeric_limits<double>::infinity();
                    if (prev_imu_msg_)
                    {
                        const double t_prev = rclcpp::Time(prev_imu_msg_->header.stamp).seconds();
                        if (t_prev < pair.timestamp && IsFinite(t_prev) &&
                            IsFiniteVec3(prev_imu_msg_->linear_acceleration) &&
                            IsFiniteVec3(prev_imu_msg_->angular_velocity) &&
                            (!filter_imu_ || IsReasonableImu(prev_imu_msg_)))
                        {
                            cv::Point3f acc(prev_imu_msg_->linear_acceleration.x,
                                            prev_imu_msg_->linear_acceleration.y,
                                            prev_imu_msg_->linear_acceleration.z);
                            cv::Point3f gyr(prev_imu_msg_->angular_velocity.x,
                                            prev_imu_msg_->angular_velocity.y,
                                            prev_imu_msg_->angular_velocity.z);
                            imu_meas.emplace_back(acc, gyr, t_prev);
                            last_added_t = t_prev;
                        }
                    }

                    if (pair.timestamp - t_imu_last > max_imu_delay_)
                    {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                             "IMU is too old for image. img=%.6f imu_last=%.6f delay=%.6f",
                                             pair.timestamp, t_imu_last, pair.timestamp - t_imu_last);
                    }

                    if (debug_timestamps_)
                    {
                        const double imu_earliest = rclcpp::Time(imu_buf_.front()->header.stamp).seconds();
                        const double imu_latest = rclcpp::Time(imu_buf_.back()->header.stamp).seconds();
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                             "IMU buf: size=%zu earliest=%.6f latest=%.6f img=%.6f",
                                             imu_buf_.size(), imu_earliest, imu_latest, pair.timestamp);
                    }

                    while (!imu_buf_.empty())
                    {
                        const double t_imu = rclcpp::Time(imu_buf_.front()->header.stamp).seconds();
                        if (t_imu > pair.timestamp)
                        {
                            break;
                        }
                        const auto& imu = imu_buf_.front();
                        if (t_imu > last_added_t)
                        {
                            if (!IsFiniteVec3(imu->linear_acceleration) || !IsFiniteVec3(imu->angular_velocity))
                            {
                                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                                     "Dropping IMU sample with NaN/Inf at t=%.6f", t_imu);
                            }
                            else if (filter_imu_ && !IsReasonableImu(imu))
                            {
                                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                                     "Dropping IMU sample with unreasonable magnitude at t=%.6f", t_imu);
                            }
                            else
                            {
                                cv::Point3f acc(imu->linear_acceleration.x,
                                                imu->linear_acceleration.y,
                                                imu->linear_acceleration.z);
                                cv::Point3f gyr(imu->angular_velocity.x,
                                                imu->angular_velocity.y,
                                                imu->angular_velocity.z);
                                imu_meas.emplace_back(acc, gyr, t_imu);
                                last_imu_time_ = t_imu;
                                last_added_t = t_imu;
                            }
                        }
                        prev_imu_msg_ = imu;
                        imu_buf_.pop_front();
                    }

                    // Add one valid boundary sample strictly after image timestamp,
                    // like ORB-SLAM3's internal preintegration queue logic expects.
                    // Keep that valid sample in the deque (do not pop) for next frame.
                    while (!imu_buf_.empty())
                    {
                        const auto& imu_next = imu_buf_.front();
                        const double t_next = rclcpp::Time(imu_next->header.stamp).seconds();

                        if (!IsFinite(t_next))
                        {
                            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                                 "Dropping IMU sample with NaN/Inf timestamp at front of future queue.");
                            imu_buf_.pop_front();
                            continue;
                        }

                        if (t_next <= pair.timestamp)
                        {
                            // Safety: shouldn't normally happen here, consume and continue.
                            prev_imu_msg_ = imu_next;
                            imu_buf_.pop_front();
                            continue;
                        }

                        if (!IsFiniteVec3(imu_next->linear_acceleration) || !IsFiniteVec3(imu_next->angular_velocity))
                        {
                            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                                 "Dropping future-boundary IMU sample with NaN/Inf at t=%.6f", t_next);
                            imu_buf_.pop_front();
                            continue;
                        }

                        if (filter_imu_ && !IsReasonableImu(imu_next))
                        {
                            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                                 "Dropping future-boundary IMU sample with unreasonable magnitude at t=%.6f", t_next);
                            imu_buf_.pop_front();
                            continue;
                        }

                        if (t_next > last_added_t)
                        {
                            cv::Point3f acc(imu_next->linear_acceleration.x,
                                            imu_next->linear_acceleration.y,
                                            imu_next->linear_acceleration.z);
                            cv::Point3f gyr(imu_next->angular_velocity.x,
                                            imu_next->angular_velocity.y,
                                            imu_next->angular_velocity.z);
                            imu_meas.emplace_back(acc, gyr, t_next);
                            last_added_t = t_next;
                        }
                        break;
                    }
                }
            }

            if (drop_stereo_pair)
            {
                {
                    std::lock_guard<std::mutex> lock(stereo_mutex_);
                    if (!stereo_queue_.empty()) stereo_queue_.pop_front();
                }
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                     "Dropping stale stereo frame at %.6f (%s)",
                                     pair.timestamp,
                                     drop_reason ? drop_reason : "unknown reason");
                continue;
            }
            // printf("Processing stereo frame at %.6f with %zu IMU measurements\n", pair.timestamp, imu_meas.size());

            if (imu_meas.size() < 2)
            {
                if (debug_timestamps_)
                {
                    const double t_now = this->get_clock()->now().seconds();
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                         "Insufficient IMU slice (<2). img=%.6f imu_count=%zu last_imu=%.6f now=%.6f",
                                         pair.timestamp, imu_meas.size(), last_imu_time_, t_now);
                }

                bool cannot_improve = false;
                {
                    std::lock_guard<std::mutex> ilock(imu_mutex_);
                    if (!imu_buf_.empty())
                    {
                        const double t_front = rclcpp::Time(imu_buf_.front()->header.stamp).seconds();
                        // If front IMU is already in the future, this frame won't get more <=timestamp samples.
                        // Boundary-after-timestamp sample has already been attempted in this iteration.
                        cannot_improve = (t_front > pair.timestamp);
                    }
                }

                if (cannot_improve)
                {
                    std::lock_guard<std::mutex> lock(stereo_mutex_);
                    if (!stereo_queue_.empty()) stereo_queue_.pop_front();
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                         "Dropping stereo frame %.6f due to persistent insufficient IMU slice",
                                         pair.timestamp);
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            bool imu_slice_valid = true;
            for (size_t i = 0; i < imu_meas.size(); ++i)
            {
                if (!IsFiniteImuPoint(imu_meas[i]))
                {
                    imu_slice_valid = false;
                    break;
                }
                if (i > 0 && !(imu_meas[i].t > imu_meas[i - 1].t))
                {
                    imu_slice_valid = false;
                    break;
                }
            }

            if (!imu_slice_valid)
            {
                {
                    std::lock_guard<std::mutex> lock(stereo_mutex_);
                    if (!stereo_queue_.empty()) stereo_queue_.pop_front();
                }
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                     "Dropping stereo frame %.6f due to invalid/non-monotonic IMU slice",
                                     pair.timestamp);
                continue;
            }
            if (debug_timestamps_)
            {
                const double imu_first = imu_meas.front().t;
                const double imu_last = imu_meas.back().t;
                const double t_now = this->get_clock()->now().seconds();
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "IMU slice: count=%zu range=[%.6f, %.6f] img=%.6f img-imu_last=%.6f now=%.6f",
                                     imu_meas.size(), imu_first, imu_last, pair.timestamp,
                                     pair.timestamp - imu_last, t_now);
            }

            // Package ready frame and push to ready_queue for consumer thread
            ReadyFrame rf;
            rf.left = pair.left;
            rf.right = pair.right;
            rf.timestamp = pair.timestamp;
            rf.imu_meas = std::move(imu_meas);

            {
                std::lock_guard<std::mutex> rlock(ready_mutex_);
                if (static_cast<int>(ready_queue_.size()) >= max_ready_queue_)
                {
                    RCLCPP_WARN(this->get_logger(), "ready_queue overflow, dropping oldest ready frame");
                    ready_queue_.pop_front();
                }
                ready_queue_.push_back(std::move(rf));
            }
            ready_cv_.notify_one();

            // Remove from stereo_queue
            {
                std::lock_guard<std::mutex> lock(stereo_mutex_);
                if (!stereo_queue_.empty()) stereo_queue_.pop_front();
            }
        }
    }

    void ConsumerLoop()
    {
        while (rclcpp::ok() && running_.load())
        {
            ReadyFrame frame;
            {
                std::unique_lock<std::mutex> lock(ready_mutex_);
                ready_cv_.wait(lock, [&]() { return !ready_queue_.empty() || !running_.load(); });
                if (!running_.load() && ready_queue_.empty()) break;
                frame = std::move(ready_queue_.front());
                ready_queue_.pop_front();
            }

            // Perform conversion and TrackStereo in consumer thread
            cv_bridge::CvImageConstPtr cv_left;
            cv_bridge::CvImageConstPtr cv_right;
            try
            {
                cv_left = cv_bridge::toCvShare(frame.left, sensor_msgs::image_encodings::MONO8);
                cv_right = cv_bridge::toCvShare(frame.right, sensor_msgs::image_encodings::MONO8);
            }
            catch (const cv_bridge::Exception& e)
            {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception in consumer: %s", e.what());
                continue;
            }

            cv::Mat im_left = cv_left->image.clone();
            cv::Mat im_right = cv_right->image.clone();

            if (use_clahe_ && clahe_)
            {
                clahe_->apply(im_left, im_left);
                clahe_->apply(im_right, im_right);
            }

            if (im_left.empty() || im_right.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Converted image empty in consumer. Dropping.");
                continue;
            }

            if (!IsImageNonBlank(im_left) || !IsImageNonBlank(im_right))
            {
                RCLCPP_WARN(this->get_logger(), "Image appears blank/constant in consumer. Dropping.");
                continue;
            }

            if (mpSLAM)
            {
                if (perf_logging_)
                {
                    auto t0 = std::chrono::steady_clock::now();
                    mpSLAM->TrackStereo(im_left, im_right, frame.timestamp, frame.imu_meas);
                    auto t1 = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    RCLCPP_INFO(this->get_logger(), "Consumer TrackStereo time: %.3f ms img=%.6f imu_count=%zu", ms, frame.timestamp, frame.imu_meas.size());
                }
                else
                {
                    mpSLAM->TrackStereo(im_left, im_right, frame.timestamp, frame.imu_meas);
                }
            }
        }
    }

    ORB_SLAM3::System* mpSLAM;

    std::string left_topic_;
    std::string right_topic_;
    std::string imu_topic_;
    int sync_queue_size_ = 10;
    int max_stereo_queue_ = 20;
    bool use_clahe_ = false;
    bool debug_timestamps_ = true;
    bool drop_zero_timestamps_ = true;
    double max_stereo_dt_ = 1e-4;
    double max_imu_delay_ = 0.02;
    bool filter_imu_ = true;
    bool perf_logging_ = false;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
    std::shared_ptr<Sync> sync_;

    std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buf_;
    std::mutex imu_mutex_;

    struct StereoPair
    {
        sensor_msgs::msg::Image::ConstSharedPtr left;
        sensor_msgs::msg::Image::ConstSharedPtr right;
        double timestamp = 0.0;
        double enqueued = 0.0;
    };

    struct ReadyFrame
    {
        sensor_msgs::msg::Image::ConstSharedPtr left;
        sensor_msgs::msg::Image::ConstSharedPtr right;
        double timestamp = 0.0;
        std::vector<ORB_SLAM3::IMU::Point> imu_meas;
    };

    std::deque<StereoPair> stereo_queue_;
    std::mutex stereo_mutex_;

    std::deque<ReadyFrame> ready_queue_;
    std::mutex ready_mutex_;
    std::condition_variable ready_cv_;
    std::thread consumer_thread_;
    int max_ready_queue_ = 10;

    std::atomic<bool> running_{false};
    std::thread processing_thread_;
    double last_imu_time_ = -1.0;
    double last_img_stamp_recv_ = -1.0;
    double last_imu_stamp_recv_ = -1.0;
    sensor_msgs::msg::Imu::ConstSharedPtr prev_imu_msg_;

    cv::Ptr<cv::CLAHE> clahe_;
};

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: stereo_inertial_ros2 path_to_vocabulary path_to_settings" << std::endl;
        return 1;
    }

    rclcpp::init(argc, argv);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true);

    auto node = std::make_shared<StereoInertialNode>(&SLAM);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    SLAM.Shutdown();
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM_Format.txt");
    SLAM.SaveTrajectoryTUM("FrameTrajectory_TUM_Format.txt");

    rclcpp::shutdown();
    return 0;
}
