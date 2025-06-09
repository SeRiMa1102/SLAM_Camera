#include "image_stabilization.hpp"
#include "filterKalman.hpp"
#include "gms_matcher.h"
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>

constexpr size_t numberToStabilize = 30;
constexpr size_t numberToApproximate = 5;
constexpr size_t neededNumberOfInliers = 50;
ImageStabilization::ImageStabilization() {}
ImageStabilization::ImageStabilization(ORB_SLAM3::System* slam)
    : SLAM(slam)
    , kalmanFilter_(new camera_stabilization::Kalman2D())
{
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 20.0; // Частота кадров
    cv::Size frameSize(1080 * 2, 1920 * 2); // Размер кадра (ширина, высота)
    // cv::Size frameSize(1080 * 2 * 2, 1920 * 2); // Размер кадра (ширина, высота)

    writer = cv::VideoWriter("/home/rinat/SLAM_Camera/build/output.mp4", fourcc, fps, frameSize, true);
    if (!writer.isOpened()) {
        std::cerr << "Ошибка открытия файла для записи видео!" << std::endl;
        exit(-1);
        // обработка ошибки
    }
    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Frame", 640, 480);

    // Eigen::Quaternionf q_smooth(1.0f, 0.0f, 0.0f, 0.0f); // (w, x, y, z)
    averageCurrentQuaternion_ = Eigen::Quaternionf(0.982242703, 0.004156070, -0.162135080, 0.094309561);
}

ImageStabilization::~ImageStabilization()
{
    delete kalmanFilter_;
    writer.release();

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
    static cv::Mat previosImage;

    Eigen::Quaternionf q;
    Eigen::Vector3f trans;
    int result = SLAM->GetCurrentPose(q, trans);
    if (result != 3) {
        std::cout << "state = " << result << std::endl;
    } else if (result == 3) {
        std::cout << "\033[31mstate = " << result << "\033[0m" << std::endl;
    }

    updateGraph(result, q);
    cv::Mat update;

    update = im.clone();
    static bool flag = false;
    if (quats_.size() == numberToStabilize) {
        std::cout << "Stabilization\n";
        cv::Mat stabilizedPrev = previosImage.clone();

        if (flag == false) {
            cv::imwrite("/home/rinat/SLAM_Camera/build/first.jpg", previosImage);
            // cv::flip(update, update, -1);
            cv::imwrite("/home/rinat/SLAM_Camera/build/second.jpg", update);
            // flag = true;
            // cv::flip(update, update, -1);
            std::cout << "first filtered\n";
        }
        // cv::Mat stabilizedPrev;
        averageCurrentQuaternion_ = average_quaternions(quats_);
        rotateImage(quats_.back(), averageCurrentQuaternion_, im, update, SLAM);
        rotateImage(quats_[quats_.size() - 2], averageCurrentQuaternion_, previosImage, stabilizedPrev, SLAM);
        if (flag == false) {
            cv::imwrite("/home/rinat/SLAM_Camera/build/first_1.jpg", stabilizedPrev);
            cv::imwrite("/home/rinat/SLAM_Camera/build/second_1.jpg", update);
            flag = true;
            std::cout << "first filtered\n";
        }
        update = shiftImage(stabilizedPrev, update);
        // update = stabilizedPrev.clone();
    }

    cv::flip(update, update, -1);

    update = updateFrameSize(update);
    //////
    // cv::Mat img1 = im.clone();

    // cv::flip(img1, img1, -1);
    // img1 = updateFrameSize(img1);

    // cv::Mat resultConcat;
    // cv::hconcat(update, img1, resultConcat);
    // // cv::imshow("Frame", resultConcat);
    // // writer.write(resultConcat);
    // cv::imshow("Frame", img1);
    // writer.write(img1);
    ///////
    cv::imshow("Frame", update);
    writer.write(update);
    cv::waitKey(1);
    previosImage = im.clone();
}

void ImageStabilization::updateGraph(int state, Eigen::Quaternionf q)
{
    // TODO:add moving average
    if (state == 1) {
        reinitFiltraion();
        return;
    }

    if (state == 2) {
        counterState2++;
        std::cout << "counter++\n";
        quats_.push_back(q);
        if (counterState2 > numberToStabilize) {
            quats_.erase(quats_.begin());
        }
        return;
    }

    if (state == 3) {
        counterState3++;
        if (counterState3 > numberToApproximate) {
            reinitFiltraion();
            return;
        }

        if (quats_.size() > 2) {
            Eigen::Quaternionf q_prev2 = quats_[quats_.size() - 2];
            Eigen::Quaternionf q_prev1 = quats_[quats_.size() - 1];
            Eigen::Quaternionf delta_q = q_prev2.inverse() * q_prev1;
            Eigen::Quaternionf q_pred = q_prev1 * delta_q;
            quats_.push_back(q_pred);
            return;
        }
    }
}

cv::Mat ImageStabilization::updateFrameSize(const cv::Mat& image)
{
    constexpr int target_width = 1080 * 2; // cols (X)
    constexpr int target_height = 1920 * 2; // rows (Y)

    // Создаем черное изображение нужного размера
    cv::Mat frame_expanded(target_height, target_width, image.type(), cv::Scalar(0, 0, 0));

    // Вычисляем смещения для центрирования
    int offset_x = (target_width - image.cols) / 2;
    int offset_y = (target_height - image.rows) / 2;

    // Если исходное изображение больше холста — делаем crop, чтобы не вывалилось за пределы
    int copy_width = std::min(image.cols, target_width);
    int copy_height = std::min(image.rows, target_height);

    // Вычисляем область исходника, если он слишком большой
    int src_x = (image.cols > target_width) ? (image.cols - target_width) / 2 : 0;
    int src_y = (image.rows > target_height) ? (image.rows - target_height) / 2 : 0;

    cv::Rect roi_dst(
        std::max(offset_x, 0),
        std::max(offset_y, 0),
        copy_width,
        copy_height);
    cv::Rect roi_src(
        src_x,
        src_y,
        copy_width,
        copy_height);

    image(roi_src).copyTo(frame_expanded(roi_dst));

    return frame_expanded;
}

cv::Mat ImageStabilization::shiftImage(const cv::Mat& prev, const cv::Mat current)
{
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(1000);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> good_matches_gms;
    std::vector<cv::Point2f> points1, points2;
    std::vector<cv::KeyPoint> keypoints_prev;
    cv::Mat descriptors_prev;
    std::vector<cv::KeyPoint> keypoints_curr;
    cv::Mat descriptors_curr;
    int inliers = 0;
    cv::Mat warp;
    cv::Point2f shift;
    cv::Point2f smoothed_shift;

    if (prev.empty() || current.empty()) {
        std::cout << "Empty frame\n";
        kalmanFilter_->reinitFilter();
        return current;
    }

    // cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    detector->detectAndCompute(prev, cv::noArray(),
        keypoints_prev, descriptors_prev);
    detector->detectAndCompute(current, cv::noArray(),
        keypoints_curr, descriptors_curr);
    {
        // cv::Mat frame_expanded(frame.rows * 3, frame.cols * 3,
        //     frame.type(),
        //     cv::Scalar(0, 0, 0));
        // frame.copyTo(frame_expanded(
        //     cv::Rect(frame.cols, frame.rows, frame.cols,
        //         frame.rows))); // Копируем исходное изображение в центр
        // cv::Mat display(frame_expanded);
    }

    if (descriptors_prev.empty() || descriptors_curr.empty()) {
        std::cout << "Warning: descriptors` are empty."
                     "Skipping frame."
                  << std::endl;
        kalmanFilter_->reinitFilter();
        return current;
    }

    matcher->match(descriptors_prev, descriptors_curr, matches);
    // Filtering matches GMS
    inliers = camera_stabilization::filterMatchesGMS(matches, keypoints_prev,
        keypoints_curr, current.size(),
        current.size(), good_matches_gms,
        false);

    std::cout << "inliers = " << inliers << std::endl;
    if (inliers < neededNumberOfInliers) {
        std::cout << "!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@\n";
        kalmanFilter_->reinitFilter();
        return current;
    }

    for (const cv::DMatch& match : good_matches_gms) {
        points1.push_back(keypoints_prev[match.queryIdx].pt);
        points2.push_back(keypoints_curr[match.trainIdx].pt);
    }

    shift = camera_stabilization::computeShift(points1, points2);
    std::cout << "Shift = " << shift << std::endl;
    smoothed_shift = kalmanFilter_->update(shift);
    warp = (cv::Mat_<double>(2, 3) << 1, 0, -smoothed_shift.x,
        0, 1,
        -smoothed_shift.y);

    double expand_ratio = 1.5;
    int canvas_w = (double)current.cols * expand_ratio;
    int canvas_h = (double)current.rows * expand_ratio;
    cv::Mat canvas(canvas_h, canvas_w, current.type(), cv::Scalar(0, 0, 0));

    // ==== 2. Вставляем исходное изображение в центр canvas ====
    int offset_x = (canvas_w - current.cols) / 2;
    int offset_y = (canvas_h - current.rows) / 2;
    current.copyTo(canvas(cv::Rect(offset_x, offset_y, current.cols, current.rows)));

    // ==== 3. Делаем сдвиг (warpAffine) на canvas ====
    cv::Mat shifted_canvas;
    warp = (cv::Mat_<double>(2, 3) << 1, 0, -smoothed_shift.x, 0, 1, -smoothed_shift.y);
    cv::warpAffine(canvas, shifted_canvas, warp, canvas.size(),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // ==== 4. Возвращаем увеличенный сдвинутый canvas ====
    return shifted_canvas;
}
void ImageStabilization::reinitFiltraion()
{
    counterState2 = 0;
    counterState3 = 0;
    kalmanFilter_->reinitFilter();
    quats_.clear();
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

void ImageStabilization::rotateImage(
    const Eigen::Quaternionf& from,
    const Eigen::Quaternionf& to,
    const cv::Mat& current,
    cv::Mat& stabilized,
    ORB_SLAM3::System* SLAM)
{
    // Гомография по текущему кадру
    cv::Mat rotFrom, rotTo;
    quatToRotation(from, rotFrom);
    quatToRotation(to, rotTo);

    cv::Mat R1 = rotTo * rotFrom.t();
    cv::Mat H;
    cv::Mat K_mat = cv::Mat(SLAM->GetSettings()->camera1()->toK());
    K_mat.convertTo(K_mat, CV_64F);

    H = K_mat * R1.inv() * K_mat.inv();

    // Вычисляем, какой размер нужен для полного вписывания после warp
    cv::Size img_size = current.size();
    std::vector<cv::Point2f> corners = {
        { 0, 0 },
        { (float)img_size.width, 0 },
        { (float)img_size.width, (float)img_size.height },
        { 0, (float)img_size.height }
    };
    std::vector<cv::Point2f> warped_corners;
    cv::perspectiveTransform(corners, warped_corners, H);
    cv::Rect bbox = cv::boundingRect(warped_corners);

    // Делаем warpPerspective с учётом bbox
    cv::Mat warped_full;
    // Сместим гомографию, чтобы результат весь поместился в изображение bbox
    cv::Mat offset = (cv::Mat_<double>(3, 3) << 1, 0, -bbox.x, 0, 1, -bbox.y, 0, 0, 1);
    cv::Mat H_total = offset * H;
    cv::warpPerspective(current, warped_full, H_total, bbox.size(),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    stabilized = warped_full.clone();
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
