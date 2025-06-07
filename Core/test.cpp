#include <Eigen/Core>
#include <algorithm>

#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <System.h>
#include <iostream>
#include <vector>

Eigen::Vector3f averageCameraPosition(const std::vector<Eigen::Vector3f>& positions)
{
    if (positions.empty()) {
        return Eigen::Vector3f::Zero(); // Возвращает ноль, если нет элементов
    }

    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (const auto& pos : positions) {
        sum += pos;
    }

    return sum / positions.size();
}

Eigen::Quaternionf average_quaternions(const std::vector<Eigen::Quaternionf>& quats)
{
    if (quats.empty())
        throw std::runtime_error("Empty quaternions!");

    // Align signs
    Eigen::Quaternionf q0 = quats[0];
    std::vector<Eigen::Quaternionf> aligned_quats = quats;
    for (size_t i = 0; i < aligned_quats.size(); ++i) {
        if (q0.coeffs().dot(aligned_quats[i].coeffs()) < 0) {
            aligned_quats[i].coeffs() *= -1.0f;
        }
    }

    // Build A
    Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
    for (const auto& q : aligned_quats) {
        Eigen::Vector4f v = q.coeffs(); // [x, y, z, w]
        A += v * v.transpose();
    }
    A /= aligned_quats.size();

    // Eigenvector
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> eigensolver(A);
    Eigen::Vector4f avg_vec = eigensolver.eigenvectors().col(3); // max eigenvalue

    Eigen::Quaternionf avg_quat(avg_vec(3), avg_vec(0), avg_vec(1), avg_vec(2)); // (w, x, y, z)
    avg_quat.normalize();
    return avg_quat;
}

int main(int argc, char** argv)
{
    std::cout << "argc = " << argc << std::endl;
    if (argc != 5) {
        cerr << endl
             << "Usage: ./camera_stabilization path_to_vocabulary path_to_settings "
                "video.mp4"
                "trajectory_file_name"
             << endl;
        return 1;
    }

    cv::VideoCapture cap(argv[argc - 2]);
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть видео" << std::endl;
        return -1;
    }

    string file_name = string(argv[argc - 1]);
    cout << "file name: " << file_name << endl;

    cout << endl
         << "-------" << endl;
    cout.precision(17);

    int fps = 30;
    float dT = 1.f / fps;

    // Create SLAM system. It initializes all system threads and gets ready to
    // process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    float imageScale = SLAM.GetImageScale();
    // SLAM.ActivateLocalizationMode();

    std::cout << "imageScale = " << imageScale << std::endl;

    double time_to_track = 0.f;
    std::vector<Eigen::Quaternionf> quats_;
    std::vector<Eigen::Vector3f> positions_;

    // Main loop
    cv::Mat im;
    size_t frame_id = 0;
    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Frame", 640, 480); // Задай нужный размер сразу!
    Eigen::Quaternionf q_smooth(-0.999977410, 0.006577891, -0.000561112, 0.001298228);
    Eigen::Vector3f p_smooth(-0.044727251, 0.007203315, 0.007359035);
    while (true) {
        // Read image from file
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        cap >> im;
        if (im.empty()) {
            std::cout << "End of file\n";
            break;
        }

        if (imageScale != 1.f) {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        double timestamp = frame_id++ * dT;

        SLAM.TrackMonocular(im, timestamp);

        cout << "timestamp = " << timestamp << ", SLAM.GetTrackingState() = " << SLAM.GetTrackingState() << endl;
        Sophus::SE3f pose = SLAM.GetTracking()->mCurrentFrame.GetPose();
        Eigen::Quaternionf q;
        Eigen::Vector3f trans;
        int result = SLAM.GetCurrentPose(q, trans);
        if (result > 1) {
            positions_.push_back(trans);
            quats_.push_back(q);
        }

        if (result == 2) {

            Eigen::Quaternionf q_delta = q_smooth * q.inverse();
            Eigen::Matrix3f R_delta = q_delta.toRotationMatrix();
            Eigen::Matrix3f K = SLAM.GetSettings()->camera1()->toK_(); // камера-интринсика
            Eigen::Matrix3f H = K * R_delta * K.inverse();

            Eigen::Vector3f t_corr = trans - p_smooth;
            float fx = K(0, 0);
            float fy = K(1, 1);
            float dx = fx * t_corr.x() / t_corr.z();
            float dy = fy * t_corr.y() / t_corr.z();
            cv::Mat warp_mat = (cv::Mat_<double>(2, 3) << 1, 0, -dx, 0, 1, -dy);
            cv::Mat stabilized;
            cv::warpAffine(im, stabilized, warp_mat, im.size());

            // cv::Mat H_cv;
            // cv::eigen2cv(H, H_cv);
            // cv::Mat stabilized;
            // cv::warpPerspective(im, stabilized, H_cv, im.size());
            cv::imshow("Frame", stabilized);

        } else {

            cv::imshow("Frame", im);
        }
        cv::waitKey(1);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                            .count();

        if (ttrack < dT) {
            time_to_track += dT;
        } else {
            time_to_track += ttrack;
        }
        usleep(50000);
    }
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    // const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
    const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
    SLAM.SaveTrajectoryEuRoC(f_file);
    // SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);

    Eigen::Quaternionf resQ = average_quaternions(quats_);
    Eigen::Vector3f resP = averageCameraPosition(positions_);
    std::cout << std::fixed;
    std::cout
        << "Size = " << quats_.size() << std::endl;

    int i = 0;
    for (const auto& q : quats_) {
        std::cout << i++ << ": ";
        std::cout << setprecision(6) << q.x() << " " << q.y() << " "
                  << q.z() << " " << q.w() << endl;
    }

    std::cout << "Average quaternion = ";
    std::cout << setprecision(9) << resQ.x() << " " << resQ.y() << " "
              << resQ.z() << " " << resQ.w() << endl;

    std::cout << "Average position = ";
    std::cout << setprecision(9) << resP.x() << " " << resP.y() << " "
              << resP.z() << endl;
    return 0;
}
