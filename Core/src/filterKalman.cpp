#include "filterKalman.hpp"

namespace camera_stabilization {

Kalman2D::Kalman2D(float process_noise, float measurement_noise,
    float error)
{
    reinitFilter(process_noise, measurement_noise, error);
}

void Kalman2D::reinitFilter(float process_noise, float measurement_noise,
    float error)
{
    kf = cv::KalmanFilter(4, 2, 0); // state: x, y, dx, dy | measurement: x, y
    // [x, y, dx, dy]
    kf.transitionMatrix
        = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    measurement = cv::Mat::zeros(2, 1, CV_32F);
    cv::setIdentity(kf.measurementMatrix);
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-4; // усилить сглаживание
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1.0; // больше игнорировать шум
    kf.errorCovPost
        = cv::Mat::eye(4, 4, CV_32F) * error;
    cv::setIdentity(kf.processNoiseCov,
        cv::Scalar::all(process_noise));
    cv::setIdentity(kf.measurementNoiseCov,
        cv::Scalar::all(measurement_noise));
    cv::setIdentity(kf.errorCovPost,
        cv::Scalar::all(error));
}

cv::Point2f Kalman2D::update(const cv::Point2f& measurement_pt)
{
    measurement.at<float>(0) = measurement_pt.x;
    measurement.at<float>(1) = measurement_pt.y;
    cv::Mat prediction = kf.predict();
    kf.correct(measurement);
    return cv::Point2f(prediction.at<float>(0),
        prediction.at<float>(1));
}

cv::Point2f computeShift(const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2)
{
    cv::Point2f total_shift(0, 0);
    for (size_t i = 0; i < pts1.size(); i++) {
        total_shift += pts2[i] - pts1[i];
    }
    return pts1.size() > 0 ? total_shift * (1.0f / pts1.size())
                           : cv::Point2f(0, 0);
}

}
