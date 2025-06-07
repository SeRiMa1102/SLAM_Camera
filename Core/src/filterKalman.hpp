#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace camera_stabilization {

class Kalman2D {
public:
    Kalman2D(float process_noise = 1e-2, float measurement_noise = 1e-1,
        float error = 1.0);

    void reinitFilter(float process_noise = 1e-2, float measurement_noise = 1e-1,
        float error = 1.0);

    cv::Point2f update(const cv::Point2f& measurement_pt);

private:
    cv::KalmanFilter kf;
    cv::Mat measurement;
};

cv::Point2f computeShift(const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2);
}