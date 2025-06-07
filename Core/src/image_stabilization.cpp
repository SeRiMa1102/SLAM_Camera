#include "image_stabilization.hpp"

constexpr size_t numberToAdjustState2 = 10;
ImageStabilization::ImageStabilization() {}
ImageStabilization::ImageStabilization(ORB_SLAM3::System* slam)
    : SLAM(slam)
{
    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Frame", 640, 480); // Задай нужный размер сразу!
}

ImageStabilization::~ImageStabilization()
{
    Eigen::Quaternionf resQ = average_quaternions(quats_);
    Eigen::Vector3f resP = averageCameraPosition(positions_);
    std::cout << std::fixed;
    // std::cout
    //     << "Size = " << quats_.size() << std::endl;

    // int i = 0;
    // for (const auto& q : quats_) {
    //     std::cout << i++ << ": ";
    //     std::cout << setprecision(6) << q.x() << " " << q.y() << " "
    //               << q.z() << " " << q.w() << endl;
    // }

    std::cout << "Average quaternion = ";
    std::cout << setprecision(9) << resQ.w() << ", " << resQ.x() << ", " << resQ.y() << ", "
              << resQ.z() << endl;

    // std::cout << "Average position = ";
    // std::cout << setprecision(9) << resP.x() << " " << resP.y() << " "
    //           << resP.z() << endl;
}

void ImageStabilization::stabilizeImage(const cv::Mat& im)
{
    Eigen::Quaternionf q_smooth(0.982242703, 0.004156070, -0.162135080, 0.094309561);

    Eigen::Quaternionf q;
    Eigen::Vector3f trans;
    int result = SLAM->GetCurrentPose(q, trans);
    if (result != 3) {
        std::cout << "state = " << result << std::endl;
    } else if (result == 3) {
        std::cout << "\033[31mstate = " << result << "\033[0m" << std::endl;
    }

    updateGraph(result, q);

    // if ((result == 2) || (result == 3)) {
    if (result == 2) {
        if (counterState2 < numberToAdjustState2) {
            cv::flip(im, im, -1);
            cv::imshow("Frame", im);
        } else {
            cv::Mat stabilized;
            rotateImage(quats_.back(), q_smooth, im, stabilized, SLAM);
            cv::flip(stabilized, stabilized, -1);
            cv::imshow("Frame", stabilized);
            // cv::imshow("Frame", im);
        }

    } else {
        counterState2 = 0;
        cv::flip(im, im, -1);
        cv::imshow("Frame", im);
    }
    cv::waitKey(1);
}

void ImageStabilization::updateGraph(int state, Eigen::Quaternionf q)
{

    if (state == 2) {
        counterState2++;
        quats_.push_back(q);
    }
    if (state == 3) {
        counterState3++;
        if (quats_.size() > 2) {
            Eigen::Quaternionf q_prev2 = quats_[quats_.size() - 2];
            Eigen::Quaternionf q_prev1 = quats_[quats_.size() - 1];
            Eigen::Quaternionf delta_q = q_prev2.inverse() * q_prev1;
            Eigen::Quaternionf q_pred = q_prev1 * delta_q;
            quats_.push_back(q);
        }
    }
}

void ImageStabilization::quatToRotation(const Eigen::Quaternionf& quat, cv::Mat& rotation)
{
    Eigen::Matrix3f R_eigen = quat.toRotationMatrix();

    cv::Mat R(3, 3, CV_64F); // <--- тут CV_64F
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R.at<double>(i, j) = static_cast<double>(R_eigen(i, j)); // приведение типов
    rotation = R;
}

void ImageStabilization::rotateImage(const Eigen::Quaternionf& from, const Eigen::Quaternionf& to, const cv::Mat& current, cv::Mat& stabilized, ORB_SLAM3::System* SLAM)
{
    cv::Mat rotFrom;
    quatToRotation(from, rotFrom);
    cv::Mat rotTo;
    quatToRotation(to, rotTo);

    cv::Mat R1 = rotTo * rotFrom.t();
    cv::Mat H;
    cv::Mat K_mat = cv::Mat(SLAM->GetSettings()->camera1()->toK());
    K_mat.convertTo(K_mat, CV_64F);
    H = K_mat * R1.inv() * K_mat.inv();
    // std::cout << "det(R1) = " << cv::determinant(R1) << std::endl;

    // Вычисляем куда попадут углы изображения после трансформации
    cv::Size img_size = current.size();
    std::vector<cv::Point2f> corners = {
        { 0, 0 },
        { (float)img_size.width, 0 },
        { (float)img_size.width, (float)img_size.height },
        { 0, (float)img_size.height }
    };

    std::vector<cv::Point2f> warped_corners;
    cv::perspectiveTransform(corners, warped_corners, H);

    // Находим границы нового изображения
    cv::Rect bbox = cv::boundingRect(warped_corners);

    // Строим смещающую матрицу, чтобы изображение не вышло за границы
    cv::Mat offset = (cv::Mat_<double>(3, 3) << 1, 0, -bbox.x, 0, 1, -bbox.y, 0, 0, 1);

    // Общая гомография с учётом смещения
    cv::Mat H_total = offset * H;

    // Применяем трансформацию
    cv::warpPerspective(current, stabilized, H_total, bbox.size());

    // // Показываем результат
    // std::string windowName = "Last warped";
    // cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    // cv::imshow(windowName, result1);
    // cv::imwrite("/home/rinat/Photogrammetry/build/rectified_2.jpg", result1);
}

Eigen::Vector3f ImageStabilization::averageCameraPosition(const std::vector<Eigen::Vector3f>& positions)
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

Eigen::Quaternionf ImageStabilization::average_quaternions(const std::vector<Eigen::Quaternionf>& quats)
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
