#pragma once
#include "filterKalman.hpp"
#include <Eigen/Core>
#include <algorithm>

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <System.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <vector>

class Kalman2D;
class ImageStabilization {
public:
    ImageStabilization();
    ImageStabilization(ORB_SLAM3::System* slam);

    ~ImageStabilization();

    void stabilizeImage(const cv::Mat& im);
    void updateGraph(int state, Eigen::Quaternionf q);
    static cv::Mat updateFrameSize(const cv::Mat& image);
    cv::Mat shiftImage(const cv::Mat& prev, const cv::Mat current);

    static void quatToRotation(const Eigen::Quaternionf& quat, cv::Mat& rotation);

    static void rotateImage(const Eigen::Quaternionf& from, const Eigen::Quaternionf& to, const cv::Mat& current, cv::Mat& stabilized, ORB_SLAM3::System* SLAM);

    static Eigen::Vector3f averageCameraPosition(const std::vector<Eigen::Vector3f>& positions);

    static Eigen::Quaternionf average_quaternions(const std::vector<Eigen::Quaternionf>& quats);

private:
    ORB_SLAM3::System* SLAM = nullptr;
    camera_stabilization::Kalman2D* kalmanFilter_;
    std::vector<Eigen::Quaternionf> quats_;
    std::vector<Eigen::Vector3f> positions_;
    int prevState = 1;
    size_t counterState2 = 0;
    size_t counterState3 = 0;
    cv::VideoWriter writer;
};