#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <System.h>
#include <iostream>
#include <vector>

// Аргумент: вектор кватернионов
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

    // Main loop
    cv::Mat im;
    size_t frame_id = 0;
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

        // Pass the image to the SLAM system
        // cout << "time_to_track = " << time_to_track << endl;
        // std::cout << "Frame size: " << im.cols << "x" << im.rows << std::endl;
        double timestamp = frame_id++ * dT;

        SLAM.TrackMonocular(im, timestamp);

        cout << "timestamp = " << timestamp << ", SLAM.GetTrackingState() = " << SLAM.GetTrackingState() << endl;
        Sophus::SE3f pose = SLAM.GetTracking()->mCurrentFrame.GetPose();
        Eigen::Quaternionf q;
        int result = SLAM.GetCurrentPose(q);
        if (result > 1) {
            quats_.push_back(q);
        }
        // Eigen::Matrix3f R = pose.rotationMatrix(); // 3x3 rotation matrix
        // Eigen::Vector3f t = pose.translation(); // translation vector

        // std::cout << "Pose rotation matrix R:\n"
        //           << R << std::endl;
        // std::cout << "Pose translation vector t:\n"
        //           << t.transpose() << std::endl;

        // Если хочешь вывести как 4x4 матрицу:
        Eigen::Matrix4f T
            = pose.matrix();
        // std::cout << "Full transformation matrix T (4x4):\n"
        //           << T << std::endl;
        // SLAM.TrackMonocular(im, time_to_track);
        // cout << "Make monocular" << std::endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                            .count();

        if (ttrack < dT) {

            time_to_track += dT;
        } else {
            time_to_track += ttrack;
        }
    }
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    // const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
    const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
    SLAM.SaveTrajectoryEuRoC(f_file);
    // SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);

    Eigen::Quaternionf resQ = average_quaternions(quats_);
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

    return 0;
}
