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
#include <vector>

#include "image_stabilization.hpp"

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
    ImageStabilization stabilizerObject(&SLAM);

    double time_to_track = 0.f;

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

        double timestamp = frame_id++ * dT;

        SLAM.TrackMonocular(im, timestamp);

        // cout << "timestamp = " << timestamp << ", SLAM.GetTrackingState() = " << SLAM.GetTrackingState() << endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                            .count();

        stabilizerObject.stabilizeImage(im);
        if (ttrack < dT) {
            time_to_track += dT;
        } else {
            time_to_track += ttrack;
        }
        // usleep(50000);
    }
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    // const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
    const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
    SLAM.SaveTrajectoryEuRoC(f_file);
    // SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);

    // Eigen::Quaternionf resQ = average_quaternions(quats_);
    // Eigen::Vector3f resP = averageCameraPosition(positions_);
    // std::cout << std::fixed;
    // std::cout
    //     << "Size = " << quats_.size() << std::endl;

    // int i = 0;
    // // for (const auto& q : quats_) {
    // //     std::cout << i++ << ": ";
    // //     std::cout << setprecision(6) << q.x() << " " << q.y() << " "
    // //               << q.z() << " " << q.w() << endl;
    // // }

    // std::cout << "Average quaternion = ";
    // std::cout << setprecision(9) << resQ.w() << ", " << resQ.x() << ", " << resQ.y() << ", "
    //           << resQ.z() << endl;

    // std::cout << "Average position = ";
    // std::cout << setprecision(9) << resP.x() << " " << resP.y() << " "
    //           << resP.z() << endl;
    return 0;
}
