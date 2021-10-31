#pragma once

#include <ros/ros.h>

#include <opencv4/opencv2/opencv.hpp>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////Type_and_macro_definitions//////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

enum class DetectorType { ShiTomasi, BRISK, FAST, ORB, AKAZE, SIFT };
// static const std::string DetectorType_str[] = {"ShiTomasi", "BRISK", "FAST",
//                                                "ORB",       "AKAZE", "SIFT"};
const std::unordered_map<std::string, DetectorType> detector_name_to_type = {
    {"ShiTomasi", DetectorType::ShiTomasi}, {"BRISK", DetectorType::BRISK},
    {"FAST", DetectorType::FAST},           {"ORB", DetectorType::ORB},
    {"AKAZE", DetectorType::AKAZE},         {"SIFT", DetectorType::SIFT}};
enum class DescriptorType { BRISK, ORB, BRIEF, AKAZE, FREAK, SIFT };
// static const std::string DescriptorType_str[] = {"BRISK", "ORB",   "BRIEF",
//                                                  "AKAZE", "FREAK", "SIFT"};
const std::unordered_map<std::string, DescriptorType> descriptor_name_to_type =
    {{"BRISK", DescriptorType::BRISK}, {"ORB", DescriptorType::ORB},
     {"BRIEF", DescriptorType::BRIEF}, {"AKAZE", DescriptorType::AKAZE},
     {"FREAK", DescriptorType::FREAK}, {"SIFT", DescriptorType::SIFT}};
enum class MatcherType { BF, FLANN };
const std::unordered_map<std::string, MatcherType> matcher_name_to_type = {
    {"BF", MatcherType::BF},
    {"FLANN", MatcherType::FLANN},
};
enum class SelectorType { NN, KNN };
const std::unordered_map<std::string, SelectorType> selector_name_to_type = {
    {"NN", SelectorType::NN},
    {"KNN", SelectorType::KNN},
};

enum MatchType {
  CURR_LEFT_CURR_RIGHT = 0,
  PREV_LEFT_CURR_LEFT = 1,
  MATCH_TYPE_NUM = 2
};
static const std::string MatchType_str[] = {"CURR_LEFT_CURR_RIGHT",
                                            "PREV_LEFT_CURR_LEFT"};
std::array<std::pair<int, int>, MATCH_TYPE_NUM> match_type_to_positions = {
    std::pair<int, int>(-2, -1), std::pair<int, int>(-4, -2)};

enum ImagePosition {
  PREV_LEFT = -4,
  PREV_RIGHT = -3,
  CURR_LEFT = -2,
  CURR_RIGHT = -1
};

///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////Main_classes_def/////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

class ClassicFeatureFrontEnd {
public:
  ClassicFeatureFrontEnd()
      : detector_type_(DetectorType::ShiTomasi),
        descriptor_type_(DescriptorType::ORB), matcher_type_(MatcherType::BF),
        selector_type_(SelectorType::NN), cross_check_(true) {
    initDetector();
    initDescriptor();
    initMatcher();
  }

  ClassicFeatureFrontEnd(const DetectorType detector_type,
                         const DescriptorType descriptor_type,
                         const MatcherType matcher_type,
                         const SelectorType selector_type,
                         const bool cross_check)
      : detector_type_(detector_type), descriptor_type_(descriptor_type),
        matcher_type_(matcher_type), selector_type_(selector_type),
        cross_check_(cross_check) {
    initDetector();
    initDescriptor();
    initMatcher();
  }

  void initDetector();
  void initDescriptor();
  void initMatcher();

  void addStereoImagePair(const cv::Mat &img_l, const cv::Mat &img_r);

  inline std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat &img) {
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(img, keypoints);
    return keypoints;
  }

  inline cv::Mat describeKeypoints(std::vector<cv::KeyPoint> &keypoints,
                                   const cv::Mat &img) {
    cv::Mat descriptors;
    extractor_->compute(img, keypoints, descriptors);
    return descriptors;
  }

  // inline cv::Mat detectAndDescribeKeypoints(const cv::Mat &img) {
  //   std::vector<cv::KeyPoint> keypoints = detectKeypoints(img);
  //   return describeKeypoints(keypoints, img);
  // }

  void matchDescriptors(const MatchType match_type);

  void solve5PointsRANSAC(const MatchType match_type,
                          const cv::Mat &camera_matrix,
                          tf2::Transform &frame1_T_frame0);
  void solveStereoOdometry(const cv::Mat &projection_matrix_l,
                           const cv::Mat &projection_matrix_r,
                           tf2::Transform &cam0_curr_T_cam0_prev,
                           const float stereo_threshold = 4.0);

  cv::Mat visualizeMatches(const MatchType match_type);

  std::deque<cv::Mat> images_dq;
  std::deque<std::vector<cv::KeyPoint>> keypoints_dq;
  std::deque<cv::Mat> descriptors_dq;
  // std::array<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>,
  //            MATCH_TYPE_NUM>
  //     matches_keypoint_pairs;
  std::array<std::vector<cv::DMatch>, MATCH_TYPE_NUM> cv_DMatches_list;
  std::array<cv::Mat, MATCH_TYPE_NUM> matches_masks;

private:
  const int IMG_HEIGHT = 375;
  const int IMG_WIDTH = 1242;

  const DetectorType detector_type_;
  const DescriptorType descriptor_type_;
  const MatcherType matcher_type_;
  const float knn_threshold_ = 0.8;
  const SelectorType selector_type_;
  const bool cross_check_;

  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorExtractor> extractor_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;
};

///////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////Main_classes_functions_def////////////////////////////////
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

void ClassicFeatureFrontEnd::initMatcher() {
  // Reference: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
  if (matcher_type_ == MatcherType::BF) {
    int norm_type;
    if (descriptor_type_ == DescriptorType::AKAZE or
        descriptor_type_ == DescriptorType::BRISK or
        descriptor_type_ == DescriptorType::ORB) {
      norm_type = cv::NORM_HAMMING;
    } else if (descriptor_type_ == DescriptorType::SIFT) {
      norm_type = cv::NORM_L2;
    } else {
      ROS_ERROR("[initMatcher] Decscriptor is not implemented");
    }
    matcher_ = cv::BFMatcher::create(norm_type, cross_check_);
  } else if (matcher_type_ == MatcherType::FLANN) {
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  }
}

void ClassicFeatureFrontEnd::addStereoImagePair(const cv::Mat &img_l,
                                                const cv::Mat &img_r) {
  images_dq.push_back(img_l);
  images_dq.push_back(img_r);

  keypoints_dq.push_back(detectKeypoints(img_l));
  keypoints_dq.push_back(detectKeypoints(img_r));

  descriptors_dq.push_back(describeKeypoints(keypoints_dq.rbegin()[1], img_l));
  descriptors_dq.push_back(describeKeypoints(keypoints_dq.rbegin()[0], img_r));

  while (images_dq.size() > 4) {
    images_dq.pop_front();
    keypoints_dq.pop_front();
    descriptors_dq.pop_front();
  }

  // // In the future, some matches could be reused
  // for (auto &match_keypoint_pairs : matches_keypoint_pairs) {
  //   match_keypoint_pairs.first.clear();
  //   match_keypoint_pairs.second.clear();
  // }

  for (auto &mask : matches_masks) {
    mask.release();
  }

  for (auto &cv_DMatches : cv_DMatches_list) {
    cv_DMatches.clear();
  }

  assert(keypoints_dq.size() <= 4);
  assert(descriptors_dq.size() <= 4);
}

void ClassicFeatureFrontEnd::matchDescriptors(const MatchType match_type) {
  cv::Mat &descriptors0 =
      descriptors_dq.end()[match_type_to_positions[match_type].first];
  cv::Mat &descriptors1 =
      descriptors_dq.end()[match_type_to_positions[match_type].second];
  std::vector<cv::KeyPoint> &keypoints0 =
      keypoints_dq.end()[match_type_to_positions[match_type].first];
  std::vector<cv::KeyPoint> &keypoints1 =
      keypoints_dq.end()[match_type_to_positions[match_type].second];

  if (matcher_type_ == MatcherType::FLANN) {
    if (descriptors0.type() != CV_32F) {
      descriptors0.convertTo(descriptors0, CV_32F);
    }
    if (descriptors1.type() != CV_32F) {
      descriptors1.convertTo(descriptors1, CV_32F);
    }
  }

  std::vector<cv::DMatch> &cv_Dmatches = cv_DMatches_list[match_type];
  if (selector_type_ == SelectorType::NN) {
    matcher_->match(descriptors0, descriptors1, cv_Dmatches);
  } else if (selector_type_ == SelectorType::KNN) {
    // k nearest neighbors (k=2)
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors0, descriptors1, knn_matches, 2);
    for (const auto &knn_match : knn_matches) {
      if (knn_match[0].distance < knn_threshold_ * knn_match[1].distance) {
        cv_Dmatches.push_back(knn_match[0]);
      }
    }
  }
}

// for prev to curr matching, keypoints0 represent the prev pts, keypoints1
// represent the curr pts
// camera_matrix: 3x3 matrix
void ClassicFeatureFrontEnd::solve5PointsRANSAC(
    const MatchType match_type, const cv::Mat &camera_matrix,
    tf2::Transform &frame1_T_frame0) {
  std::vector<cv::Point2f> matched_keypoints0;
  std::vector<cv::Point2f> matched_keypoints1;

  // std::vector<cv::Point2f> &matched_keypoints0 =
  //     matches_keypoint_pairs[match_type].first;
  // std::vector<cv::Point2f> &matched_keypoints1 =
  //     matches_keypoint_pairs[match_type].second;
  // assert(matched_keypoints0.empty());
  // assert(matched_keypoints1.empty());

  std::vector<cv::KeyPoint> &keypoints0 =
      keypoints_dq.end()[match_type_to_positions[match_type].first];
  std::vector<cv::KeyPoint> &keypoints1 =
      keypoints_dq.end()[match_type_to_positions[match_type].second];
  const std::vector<cv::DMatch> &cv_Dmatches = cv_DMatches_list[match_type];
  matched_keypoints0.resize(cv_Dmatches.size());
  matched_keypoints1.resize(cv_Dmatches.size());
  size_t i = 0;
  for (const auto &match : cv_Dmatches) {
    matched_keypoints0[i] = keypoints0[match.queryIdx].pt;
    matched_keypoints1[i] = keypoints1[match.trainIdx].pt;
    ++i;
  }

  cv::Mat &mask = matches_masks[match_type];

  // cv::Mat mask, triangulated_points;
  // prob: 0.999
  // threshold: 1.0
  cv::Mat E = cv::findEssentialMat(matched_keypoints0, matched_keypoints1,
                                   camera_matrix, cv::RANSAC, 0.999, 1.0, mask);
  cv::Mat R, t;
  int recover_result =
      cv::recoverPose(E, matched_keypoints0, matched_keypoints1, camera_matrix,
                      R, t, 200.0f, mask);

  const tf2::Matrix3x3 R_tf2(
      R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
      R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
      R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
  frame1_T_frame0.setBasis(R_tf2);
  frame1_T_frame0.setOrigin(
      tf2::Vector3(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)));

  tf2::Quaternion q_tf2 = frame1_T_frame0.getRotation();
  ROS_INFO("From RANSAC, frame1_T_frame0.rotation(): axis = %.4f, %.4f, "
           "%.4f, and angle = %.4f",
           q_tf2.getAxis().getX(), q_tf2.getAxis().getY(),
           q_tf2.getAxis().getZ(), q_tf2.getAngle());
  ROS_INFO("From RANSAC, frame1_T_frame0.translation(): t = %.4f, %.4f, %.4f",
           t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
}

void ClassicFeatureFrontEnd::solveStereoOdometry(
    const cv::Mat &projection_matrix_l, const cv::Mat &projection_matrix_r,
    tf2::Transform &cam0_curr_T_cam0_prev, const float stereo_threshold) {
  const std::vector<cv::DMatch> &cv_Dmatches_curr_stereo =
      cv_DMatches_list[CURR_LEFT_CURR_RIGHT];
  const std::vector<cv::DMatch> &cv_Dmatches_inter_left =
      cv_DMatches_list[PREV_LEFT_CURR_LEFT];

  // prepare hashsets of curr left keypoint indices
  std::vector<int> indices_curr_left_from_curr_stereo;
  // TODO: Stereo epipolar constraint
  // TODO: handle edge case, disparity = 0
  std::transform(
      cv_Dmatches_curr_stereo.begin(), cv_Dmatches_curr_stereo.end(),
      std::back_inserter(indices_curr_left_from_curr_stereo),
      [](const cv::DMatch &cv_Dmatch) { return cv_Dmatch.queryIdx; });
  std::vector<std::pair<int, int>> index_pairs_curr_left_to_prev_left;
  std::transform(cv_Dmatches_inter_left.begin(), cv_Dmatches_inter_left.end(),
                 std::back_inserter(index_pairs_curr_left_to_prev_left),
                 [](const cv::DMatch &cv_Dmatch) -> std::pair<int, int> {
                   return {cv_Dmatch.trainIdx, cv_Dmatch.queryIdx};
                 });

  const std::unordered_set<int> set_of_indices_curr_left_from_curr_stereo(
      indices_curr_left_from_curr_stereo.begin(),
      indices_curr_left_from_curr_stereo.end());
  const std::unordered_map<int, int> map_of_indices_curr_left_to_prev_left(
      index_pairs_curr_left_to_prev_left.begin(),
      index_pairs_curr_left_to_prev_left.end());

  // extract common feature points in both matches
  std::vector<cv::Point2f> keypoints_curr_left;
  keypoints_curr_left.reserve(cv_Dmatches_curr_stereo.size());
  std::vector<cv::Point2f> keypoints_curr_right;
  keypoints_curr_right.reserve(cv_Dmatches_curr_stereo.size());
  std::vector<cv::Point2f> keypoints_prev_left;
  keypoints_prev_left.reserve(cv_Dmatches_curr_stereo.size());

  for (const auto &cv_Dmatch : cv_Dmatches_curr_stereo) {
    const int index_curr_left = cv_Dmatch.queryIdx;
    if (map_of_indices_curr_left_to_prev_left.find(index_curr_left) ==
        map_of_indices_curr_left_to_prev_left.end())
      continue;

    const cv::Point2f &keypoint_curr_left =
        keypoints_dq.end()[CURR_LEFT][index_curr_left].pt;
    const cv::Point2f &keypoint_curr_right =
        keypoints_dq.end()[CURR_RIGHT][cv_Dmatch.trainIdx].pt;

    if (std::abs(keypoint_curr_left.y - keypoint_curr_right.y) >
            stereo_threshold ||
        std::abs(keypoint_curr_left.x - keypoint_curr_right.x) < 0.25f)
      continue;

    const cv::Point2f &keypoint_prev_left =
        keypoints_dq
            .end()[PREV_LEFT]
                  [map_of_indices_curr_left_to_prev_left.at(index_curr_left)]
            .pt;

    keypoints_curr_left.push_back(keypoint_curr_left);
    keypoints_curr_right.push_back(keypoint_curr_right);
    keypoints_prev_left.push_back(keypoint_prev_left);
  }

  // triangulation
  cv::Mat curr_left_point4d;
  cv::triangulatePoints(projection_matrix_l, projection_matrix_r,
                        keypoints_curr_left, keypoints_curr_right,
                        curr_left_point4d);
  // curr_left_point4d: 64FC1 4xN => 64FC1 Nx4
  curr_left_point4d = curr_left_point4d.t();
  // curr_left_point4d: 64FC1 Nx4 => 64FC4 Nx1
  curr_left_point4d = curr_left_point4d.reshape(4, curr_left_point4d.rows);

  cv::Mat curr_left_point3d;
  // curr_left_point3d: 64FC3 Nx1
  cv::convertPointsFromHomogeneous(curr_left_point4d, curr_left_point3d);

  // PnP
  const cv::Mat intrinsics_l = projection_matrix_l.colRange(0, 3);
  cv::Mat inliers;
  const cv::Mat distortion = cv::Mat::zeros(4, 1, CV_32FC1);
  cv::Mat r_vec, t;
  cv::solvePnPRansac(curr_left_point3d, keypoints_prev_left, intrinsics_l,
                     distortion, r_vec, t, false, 500, 2.0, 0.999, inliers);

  cv::Mat R;
  cv::Rodrigues(r_vec, R);

  // TODO: use inliers to refine PnP

  // output
  tf2::Transform cam0_prev_T_cam0_curr;
  const tf2::Matrix3x3 R_tf2(
      R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
      R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
      R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
  cam0_prev_T_cam0_curr.setBasis(R_tf2);
  cam0_prev_T_cam0_curr.setOrigin(
      tf2::Vector3(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)));

  tf2::Quaternion q_tf2 = cam0_prev_T_cam0_curr.getRotation();
  ROS_INFO("From PnP, cam0_prev_T_cam0_curr.rotation():\n axis = %.4f, %.4f, "
           "%.4f, and angle = %.4f",
           q_tf2.getAxis().getX(), q_tf2.getAxis().getY(),
           q_tf2.getAxis().getZ(), q_tf2.getAngle());
  ROS_INFO(
      "From PnP, cam0_prev_T_cam0_curr.translation():\n t = %.4f, %.4f, %.4f",
      t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));

  cam0_curr_T_cam0_prev = cam0_prev_T_cam0_curr.inverse();
}

///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////Helper_functions/////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

// TODO: Add mask visualization (cautious for empty mask)
cv::Mat ClassicFeatureFrontEnd::visualizeMatches(const MatchType match_type) {
  assert(images_dq.size() >= 4);

  // image dq: prev_l, prev_r, curr_l, curr_r
  // eg. -3 and -1 for prev_left - curr_left
  const cv::Mat &image0 =
      images_dq.end()[match_type_to_positions[match_type].first];
  const cv::Mat &image1 =
      images_dq.end()[match_type_to_positions[match_type].second];
  std::vector<cv::KeyPoint> &keypoints0 =
      keypoints_dq.end()[match_type_to_positions[match_type].first];
  std::vector<cv::KeyPoint> &keypoints1 =
      keypoints_dq.end()[match_type_to_positions[match_type].second];

  const std::vector<cv::DMatch> cv_DMatches = cv_DMatches_list[match_type];

  std::vector<cv::DMatch> matches_dnsp;
  const int stride = std::ceil(static_cast<float>(cv_DMatches.size()) /
                               100.0f); // at most 100 points

  for (int i = 0; i < cv_DMatches.size(); i += stride) {
    matches_dnsp.push_back(cv_DMatches[i]);
  }

  cv::Mat matching_img = image1.clone();
  cv::drawMatches(image0, keypoints0, image1, keypoints1, matches_dnsp,
                  matching_img, cv::Scalar::all(-1), cv::Scalar::all(-1),
                  std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);

  return matching_img;
}