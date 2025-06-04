#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./calibrate <image_dir> <rows> <cols>\n";
        return 1;
    }

    std::string img_dir = argv[1];
    int rows = std::stoi(argv[2]);
    int cols = std::stoi(argv[3]);

    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;
    std::vector<cv::Point3f> objp;

    float square_size = 24.0f; // мм

    for (int i = 0; i < rows * cols; ++i)
        objp.emplace_back((i % cols) * square_size, (i / cols) * square_size, 0.0f);


    cv::Size patternSize(cols, rows);

    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        std::cout << img.cols << " " << img.rows << std::endl;
        if (img.empty()) continue;

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img, patternSize, corners);

        if (found) {
            std::cout <<found << std::endl;
            cv::cornerSubPix(img, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            imgpoints.push_back(corners);
            objpoints.push_back(objp);
        }
    }

    if (imgpoints.empty()) {
        std::cerr << "No chessboard corners were found. Check your images and pattern size.\n";
        return 1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    cv::calibrateCamera(objpoints, imgpoints, cv::Size(640, 480),
                        cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "Camera matrix:\n" << cameraMatrix << "\n";
    std::cout << "Distortion coeffs:\n" << distCoeffs << "\n";

    cv::FileStorage fs("calib.yaml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs.release();

    std::cout << "Calibration saved to calib.yaml\n";
    return 0;
}
