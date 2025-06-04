#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>

#include <System.h>

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

    std::cout << "imageScale = " << imageScale << std::endl;

    double time_to_track = 0.f;

    // Main loop
    cv::Mat im;
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
        cout << "time_to_track = " << time_to_track << endl;
        // std::cout << "Frame size: " << im.cols << "x" << im.rows << std::endl;
        SLAM.TrackMonocular(im, time_to_track);
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
    const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
    const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
    SLAM.SaveTrajectoryEuRoC(f_file);
    SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);

    return 0;
}
