#pragma once

#include <ros/ros.h>
#include <tf2/LinearMath/Transform.h>

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
const std::string MatchType_str[] = {"CURR_LEFT_CURR_RIGHT",
                                            "PREV_LEFT_CURR_LEFT"};
const std::array<std::pair<int, int>, MATCH_TYPE_NUM> match_type_to_positions = {
    std::pair<int, int>(-2, -1), std::pair<int, int>(-4, -2)};

enum ImagePosition {
  PREV_LEFT = -4,
  PREV_RIGHT = -3,
  CURR_LEFT = -2,
  CURR_RIGHT = -1
};

///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Abstract_class_def/////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

class FeatureFrontEnd {
public:
  FeatureFrontEnd(const MatcherType matcher_type,
                  const SelectorType selector_type, const bool cross_check)
      : matcher_type_(matcher_type), selector_type_(selector_type),
        cross_check_(cross_check) {}
  virtual void initMatcher() = 0;
  virtual void addStereoImagePair(const cv::Mat &img_l,
                                  const cv::Mat &img_r) = 0;
  void solveStereoOdometry(const cv::Mat &projection_matrix_l,
                           const cv::Mat &projection_matrix_r,
                           tf2::Transform &cam0_curr_T_cam0_prev,
                           const float stereo_threshold = 4.0f);
  cv::Mat visualizeMatches(const MatchType match_type);

  std::deque<cv::Mat> images_dq;
  std::deque<std::vector<cv::KeyPoint>> keypoints_dq;
  std::deque<cv::Mat> descriptors_dq;
  std::array<std::vector<cv::DMatch>, MATCH_TYPE_NUM> cv_DMatches_list;

protected:
  const int IMG_HEIGHT = 375;
  const int IMG_WIDTH = 1242;

  const MatcherType matcher_type_;
  const float knn_threshold_ = 0.8;
  const SelectorType selector_type_;
  const bool cross_check_;

  cv::Ptr<cv::DescriptorMatcher> matcher_;

  cv::Mat r_vec_pred = cv::Mat::zeros(3, 1, CV_32FC1);
  cv::Mat t_pred = cv::Mat::zeros(3, 1, CV_32FC1);
};

///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////classic_classes_def//////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

class ClassicFeatureFrontEnd : public FeatureFrontEnd {
public:
  ClassicFeatureFrontEnd()
      : detector_type_(DetectorType::ShiTomasi),
        descriptor_type_(DescriptorType::ORB),
        FeatureFrontEnd(MatcherType::BF, SelectorType::NN, true) {
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
        FeatureFrontEnd(matcher_type, selector_type, cross_check) {
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

  void matchDescriptors(const MatchType match_type);

private:
  const DetectorType detector_type_;
  const DescriptorType descriptor_type_;

  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorExtractor> extractor_;
};

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


class NeuralNetworkFeatureFrontEnd : public FeatureFrontEnd {
public:
  void addStereoImagePair(const cv::Mat &img_l, const cv::Mat &img_r);

private:
};

