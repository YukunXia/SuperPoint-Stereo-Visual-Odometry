#include <odml_visual_odometry/feature_detection.hpp>

///////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////classic_class_functions_def///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void ClassicFeatureFrontEnd::initDetector() {
  switch (detector_type_) {
  case DetectorType::BRISK:
    detector_ = cv::BRISK::create();
    break;
  case DetectorType::ORB: {
    const int num_features = 2000;
    const float scale_factor = 1.2f;
    const int nlevels = 8;
    const int edge_threshold = 31;
    const int first_level = 0;
    const int WTA_K = 2;
    const cv::ORB::ScoreType score_type = cv::ORB::FAST_SCORE;
    const int patch_size = 31;
    const int fast_threshold = 20;
    detector_ = cv::ORB::create(num_features, scale_factor, nlevels,
                                edge_threshold, first_level, WTA_K, score_type,
                                patch_size, fast_threshold);
  } break;
  case DetectorType::AKAZE:
    detector_ = cv::AKAZE::create();
    break;
  case DetectorType::SIFT:
    detector_ = cv::SIFT::create();
    break;
  case DetectorType::FAST: {
    const int threshold = 10;
    const bool nonmaxSuppression = true;
    detector_ = cv::FastFeatureDetector::create(threshold, nonmaxSuppression);
  } break;
  case DetectorType::ShiTomasi: {
    const int max_corners = 1000;
    const double quality_level = 0.03;
    const double min_distance = 7.5;
    const int block_size = 5;
    const bool use_harris_detector = false;
    const double k = 0.04;
    detector_ =
        cv::GFTTDetector::create(max_corners, quality_level, min_distance,
                                 block_size, use_harris_detector, k);
  } break;
  default:
    ROS_ERROR("[initDetector] Detector is not implemented");
    break;
  }
}

void ClassicFeatureFrontEnd::initDescriptor() {
  switch (descriptor_type_) {
  case DescriptorType::BRISK: {
    // FAST/AGAST detection threshold score.
    const int threshold = 30;
    // detection octaves (use 0 to do single scale)
    const int octaves = 3;
    // apply this scale to the pattern used for sampling the neighbourhood of a
    // keypoint.
    const float pattern_scale = 1.0f;
    extractor_ = cv::BRISK::create(threshold, octaves, pattern_scale);
  } break;
  case DescriptorType::ORB:
    extractor_ = cv::ORB::create();
    break;
  case DescriptorType::AKAZE:
    extractor_ = cv::AKAZE::create();
    break;
  case DescriptorType::SIFT:
    extractor_ = cv::SIFT::create();
    break;
  default:
    ROS_ERROR("[initDescriptor] Decscriptor is not implemented");
    break;
  }
}

void ClassicFeatureFrontEnd::addStereoImagePair(
    const cv::Mat &img_l, const cv::Mat &img_r,
    const cv::Mat &projection_matrix_l, const cv::Mat &projection_matrix_r) {
  if (img_l.rows != img_r.rows || img_l.cols != img_l.cols) {
    ROS_ERROR("input images shape doesn't match! img_l's shape = (%d, %d), "
              "while img_r's shape = (%d, %d)",
              img_l.rows, img_l.cols, img_r.rows, img_r.cols);
    return;
  }

  const auto start = std::chrono::system_clock::now();

  projection_matrix_l_ = projection_matrix_l.clone();
  projection_matrix_r_ = projection_matrix_r.clone();

  images_dq.push_back(img_l);
  images_dq.push_back(img_r);

  keypoints_dq.push_back(detectKeypoints(img_l));
  keypoints_dq.push_back(detectKeypoints(img_r));
  ROS_INFO("%lu, %lu keypoints for img_l and img_r",
           keypoints_dq.end()[-2].size(), keypoints_dq.end()[-1].size());

  descriptors_dq.push_back(describeKeypoints(keypoints_dq.rbegin()[1], img_l));
  descriptors_dq.push_back(describeKeypoints(keypoints_dq.rbegin()[0], img_r));

  while (images_dq.size() > 4) {
    images_dq.pop_front();
    keypoints_dq.pop_front();
    descriptors_dq.pop_front();
  }

  assert(keypoints_dq.size() <= 4);
  assert(descriptors_dq.size() <= 4);

  const auto end = std::chrono::system_clock::now();
  ROS_INFO(
      "(pre, mid, post)processing detection of 1 image takes %.4f ms",
      (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() /
          1000.0f);
}