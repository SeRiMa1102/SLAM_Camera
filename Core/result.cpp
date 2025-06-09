#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#define PROCCESSED 0

int main()
{
#if PROCCESSED
    cv::VideoCapture cap("/home/rinat/test/results/proccessed.mp4");
#else
    cv::VideoCapture cap("/home/rinat/test/results/initial.mp4");
#endif

    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Frame", 640, 480);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video\n";
        return -1;
    }

    cv::Mat prevGray, gray, frame;
    std::vector<cv::Point2f> pointsPrev, pointsNext;

    // Захват первого кадра
    cap >> frame;
    cv::cvtColor(frame, prevGray, cv::COLOR_BGR2GRAY);

// ===== 1. Выбор начальной точки вручную =====
#if PROCCESSED
    cv::Point2f initialPoint(1050, 2350); // или полученная с mouse click
    // cv::Point2f initialPoint(900, 2110); // или полученная с mouse click
#else
    cv::Point2f initialPoint(1050, 2350); // или полученная с mouse click
    // cv::Point2f initialPoint(1080 - 310, 800); // или полученная с mouse click
#endif
    pointsPrev.push_back(initialPoint);
    cv::circle(frame, initialPoint, 30, cv::Scalar(0, 255, 0), 20); // зелёный круг
#if !PROCCESSED
    // cv::flip(frame, frame, -1);
#endif

    cv::imshow("Frame", frame);
    // cv::waitKey(30);
    cv::waitKey(5000);

    std::vector<float> displacementHistory;
    float totalDisplacement = 0.0f;
    size_t counter = 0;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<uchar> status;
        std::vector<float> err;

        // ===== 2. Optical flow tracking =====
        cv::calcOpticalFlowPyrLK(prevGray, gray, pointsPrev, pointsNext, status, err);

        // cv::circle(frame, initialPoint, 30, cv::Scalar(0, 0, 255), 20);

        if (status[0]) {
            std::cout << "Tracking = " << counter++ << "\n";
            float dx = pointsNext[0].x - initialPoint.x;
            float dy = pointsNext[0].y - initialPoint.y;
            float displacement = std::sqrt(dx * dx + dy * dy);
            displacementHistory.push_back(displacement);

            // Для визуализации
            // cv::circle(frame, initialPoint, 30, cv::Scalar(0, 255, 0), 20); // зелёный круг
            cv::circle(frame, pointsNext[0], 30, cv::Scalar(0, 255, 0), 20);
        }

// Показать кадр (опционально)
#if !PROCCESSED
        // cv::flip(frame, frame, -1);
#endif
        cv::imshow("Frame", frame);
        if (cv::waitKey(30) == 27)
            break;

        prevGray = gray.clone();
        pointsPrev = pointsNext;
    }

// ===== 3. Сохранение графика в файл =====
#if PROCCESSED
    std::ofstream file("/home/rinat/SLAM_Camera/displacement_proccessed.csv");
#else
    std::ofstream file("/home/rinat/SLAM_Camera/displacement.csv");
#endif

    for (size_t i = 0; i < displacementHistory.size(); ++i) {
        file << i << "," << displacementHistory[i] << "\n";
    }

    std::cout << "Данные сохранены в displacement.csv\n";
    return 0;
}
