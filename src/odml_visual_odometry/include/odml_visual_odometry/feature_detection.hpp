#pragma once

#include <ros/ros.h>
#include <tf2/LinearMath/Transform.h>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv4/opencv2/opencv.hpp>
#define EIGEN_USE_THREADS
#include <eigen3/unsupported/Eigen/CXX11/ThreadPool>
#include <unsupported/Eigen/CXX11/Tensor>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <odml_visual_odometry/logging.h>

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////Type_and_macro_definitions//////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

enum class DetectorType {
  ShiTomasi,
  BRISK,
  FAST,
  ORB,
  AKAZE,
  SIFT,
  SuperPoint
};
// static const std::string DetectorType_str[] = {"ShiTomasi", "BRISK", "FAST",
//                                                "ORB",       "AKAZE", "SIFT"};
const std::unordered_map<std::string, DetectorType> detector_name_to_type = {
    {"ShiTomasi", DetectorType::ShiTomasi},
    {"BRISK", DetectorType::BRISK},
    {"FAST", DetectorType::FAST},
    {"ORB", DetectorType::ORB},
    {"AKAZE", DetectorType::AKAZE},
    {"SIFT", DetectorType::SIFT},
    {"SuperPoint", DetectorType::SuperPoint}};
enum class DescriptorType { BRISK, ORB, BRIEF, AKAZE, FREAK, SIFT, SuperPoint };
// static const std::string DescriptorType_str[] = {"BRISK", "ORB",   "BRIEF",
//                                                  "AKAZE", "FREAK", "SIFT",
//                                                  "SUPERPOINT"};
const std::unordered_map<std::string, DescriptorType> descriptor_name_to_type =
    {{"BRISK", DescriptorType::BRISK},
     {"ORB", DescriptorType::ORB},
     {"BRIEF", DescriptorType::BRIEF},
     {"AKAZE", DescriptorType::AKAZE},
     {"FREAK", DescriptorType::FREAK},
     {"SIFT", DescriptorType::SIFT},
     {"SuperPoint", DescriptorType::SuperPoint}};
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

enum ImagePosition {
  PREV_LEFT = -4,
  PREV_RIGHT = -3,
  CURR_LEFT = -2,
  CURR_RIGHT = -1,
  NUM_IMAGE_POSITIONS = 4
};
const std::map<int, std::string> ImagePosition_str = {
    {PREV_LEFT, "PREV_LEFT"},
    {PREV_RIGHT, "PREV_RIGHT"},
    {CURR_LEFT, "CURR_LEFT"},
    {CURR_RIGHT, "CURR_RIGHT"}};

enum MatchType {
  CURR_LEFT_CURR_RIGHT = 0,
  CURR_LEFT_PREV_LEFT = 1,
  PREV_LEFT_PREV_RIGHT = 2,
  MATCH_TYPE_NUM = 3
};
const std::string MatchType_str[] = {
    "CURR_LEFT_CURR_RIGHT", "CURR_LEFT_PREV_LEFT", "PREV_LEFT_PREV_RIGHT"};
const std::array<std::pair<int, int>, MATCH_TYPE_NUM> match_type_to_positions =
    {std::pair<int, int>(CURR_LEFT, CURR_RIGHT),
     std::pair<int, int>(CURR_LEFT, PREV_LEFT),
     std::pair<int, int>(PREV_LEFT, PREV_RIGHT)};

///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Abstract_class_def/////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

class FeatureFrontEnd {
public:
  FeatureFrontEnd(const DetectorType detector_type,
                  const DescriptorType descriptor_type,
                  const MatcherType matcher_type,
                  const SelectorType selector_type, const bool cross_check,
                  const float stereo_threshold, const float min_disparity,
                  const int refinement_degree, const bool verbose,
                  const int input_height, const int input_width)
      : detector_type_(detector_type), descriptor_type_(descriptor_type),
        matcher_type_(matcher_type), selector_type_(selector_type),
        cross_check_(cross_check), stereo_threshold_(stereo_threshold),
        min_disparity_(min_disparity), refinement_degree_(refinement_degree),
        verbose_(verbose), input_height_(input_height),
        input_width_(input_width) {}
  virtual ~FeatureFrontEnd(){};
  void initMatcher();
  void clearLagecyData();
  void preprocessImageImpl(cv::Mat &img, cv::Mat &projection_matrix);
  virtual void addStereoImagePair(cv::Mat &img_l, cv::Mat &img_r,
                                  const cv::Mat &projection_matrix_l,
                                  const cv::Mat &projection_matrix_r) = 0;
  void matchDescriptors(const MatchType match_type);
  void solveStereoOdometry(tf2::Transform &cam0_curr_T_cam0_prev);
  cv::Mat visualizeMatches(const MatchType match_type);
  cv::Mat visualizeInliers(const ImagePosition image_position);

  std::deque<cv::Mat> images_dq;
  std::deque<std::vector<cv::KeyPoint>> keypoints_dq;
  // // valid means that these keypoints passed both CURR_LEFT_CURR_RIGHT and
  // CURR_LEFT_PREV_LEFT std::deque<std::vector<cv::KeyPoint>>
  // keypoints_valid_dq;
  std::deque<cv::Mat> descriptors_dq;
  std::array<std::vector<cv::DMatch>, MATCH_TYPE_NUM> cv_DMatches_list;

  const bool verbose_;

protected:
  const DetectorType detector_type_;
  const DescriptorType descriptor_type_;
  const MatcherType matcher_type_;
  const float knn_threshold_ = 0.8;
  const SelectorType selector_type_;
  const bool cross_check_;
  const float stereo_threshold_;
  const float min_disparity_;
  // nums of nonlinear least square optmz factor per time frame
  const int refinement_degree_;
  // Cautious: fit the following number into use case
  constexpr static double TIME_INTERVAL = 0.1;
  constexpr static double MAX_ACCELERATION = 8.0;
  constexpr static int IGNORE_FRAME_COUNT = 10;
  const int input_height_;
  const int input_width_;

  cv::Ptr<cv::DescriptorMatcher> matcher_;

  cv::Mat projection_matrix_l_;
  cv::Mat projection_matrix_r_;

  cv::Mat r_vec_pred = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat t_vec_pred = cv::Mat::zeros(3, 1, CV_64FC1);
  int frame_count = 0;

  // std::array<std::unordered_map<int, int>, MATCH_TYPE_NUM> maps_of_indices;
  std::array<std::vector<int>, MATCH_TYPE_NUM> maps_of_indices;

  // will not be used if refinement_degree <= 2
  bool prev_left_points_3d_inited = false;
  cv::Mat prev_left_points_3d;
  // -1 means invalid, valide means in the last time frame, the points with
  // these indices passed both CURR_LEFT_CURR_RIGHT, CURR_LEFT_PREV_LEFT and
  // stereo checks. Valid indices are also the indices for the triangulation.
  std::vector<int> map_from_prev_left_matched_to_prev_valid_index;
  std::vector<int> map_from_curr_valid_to_prev_left_matched_index;
  std::vector<int> map_from_curr_left_matched_to_curr_valid_index;

  // for visualization only. matched indices at curr left. also maps from curr
  // valid indices to curr matched indices.
  std::vector<int> inliers_postmatching;
  // for both visualization and refinement. valid indices at curr frame
  std::vector<int> inliers_pnp;
};

///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////classic_classes_def//////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

class ClassicFeatureFrontEnd : public FeatureFrontEnd {
public:
  ClassicFeatureFrontEnd()
      : FeatureFrontEnd(DetectorType::ShiTomasi, DescriptorType::ORB,
                        MatcherType::BF, SelectorType::NN, true, 2.0f, 1.0f, 4,
                        true, 120, 392) {
    initDetector();
    initDescriptor();
    initMatcher();
  }

  ClassicFeatureFrontEnd(const DetectorType detector_type,
                         const DescriptorType descriptor_type,
                         const MatcherType matcher_type,
                         const SelectorType selector_type,
                         const bool cross_check, const float stereo_threshold,
                         const float min_disparity, const int refinement_degree,
                         const bool verbose, const int input_height,
                         const int input_width)
      : FeatureFrontEnd(detector_type, descriptor_type, matcher_type,
                        selector_type, cross_check, stereo_threshold,
                        stereo_threshold, refinement_degree, verbose,
                        input_height, input_width) {
    initDetector();
    initDescriptor();
    initMatcher();
  }

  void initDetector();
  void initDescriptor();

  void addStereoImagePair(cv::Mat &img_l, cv::Mat &img_r,
                          const cv::Mat &projection_matrix_l,
                          const cv::Mat &projection_matrix_r);

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

private:
  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorExtractor> extractor_;
};

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

enum TensorRtPrecision {
  TRT_FP32 = 0,
  TRT_FP16 = 1,
  NUM_TRT_PRECISION_CHOICES = 2
};
const std::unordered_map<std::string, TensorRtPrecision>
    trt_precision_string2enum = {
        {"FP32", TRT_FP32},
        {"FP16", TRT_FP16},
};
const std::array<std::string, NUM_TRT_PRECISION_CHOICES>
    trt_precision_enum2string = {"FP32", "FP16"};

class SuperPointFeatureFrontEnd : public FeatureFrontEnd {
public:
  SuperPointFeatureFrontEnd()
      : FeatureFrontEnd(DetectorType::SuperPoint, DescriptorType::SuperPoint,
                        MatcherType::BF, SelectorType::NN, true, 2.0f, 1.0f, 4,
                        true, 120, 392),
        model_name_prefix_("superpoint_pretrained"), model_batch_size_(2),
        machine_name_("laptop"), trt_precision_(TRT_FP32),
        input_size_(2 * 120 * 392),
        output_det_size_(2 * output_det_channel_ * 49 * 15),
        output_desc_size_(2 * output_desc_channel_ * 49 * 15),
        output_width_(49), output_height_(15), conf_thresh_(0.015),
        dist_thresh_(4), num_threads_(12), border_remove_(4) {

    initMatcher();
    initPointers();
    loadTrtEngine();
  }

  SuperPointFeatureFrontEnd(
      const MatcherType matcher_type, const SelectorType selector_type,
      const bool cross_check, const std::string model_name_prefix,
      const int model_batch_size, const std::string machine_name,
      const TensorRtPrecision trt_precision, const int input_height,
      const int input_width, const float conf_thresh, const int dist_thresh,
      const int num_threads, const int border_remove,
      const float stereo_threshold, const float min_disparity,
      const int refinement_degree, const bool verbose)
      : FeatureFrontEnd(DetectorType::SuperPoint, DescriptorType::SuperPoint,
                        matcher_type, selector_type, cross_check,
                        stereo_threshold, min_disparity, refinement_degree,
                        verbose, input_height, input_width),
        model_name_prefix_(model_name_prefix),
        model_batch_size_(model_batch_size), machine_name_(machine_name),
        trt_precision_(trt_precision),
        input_size_(model_batch_size * input_height * input_width),
        output_det_size_(model_batch_size * output_det_channel_ * input_height *
                         input_width / 64),
        output_desc_size_(model_batch_size * output_desc_channel_ *
                          input_height * input_width / 64),
        output_width_(input_width / 8), output_height_(input_height / 8),
        conf_thresh_(conf_thresh), dist_thresh_(dist_thresh),
        num_threads_(num_threads), border_remove_(border_remove) {
    assert(input_height % 8 == 0 && input_width % 8 == 0);

    std::cout << "input_size_ = " << input_size_ << std::endl;
    std::cout << "output_det_size_ = " << output_det_size_ << std::endl;
    std::cout << "output_desc_size_ = " << output_desc_size_ << std::endl;

    initMatcher();
    initPointers();
    loadTrtEngine();
  }

  ~SuperPointFeatureFrontEnd();

  void initPointers() {
    input_data_ = std::unique_ptr<float[]>(new float[input_size_]);
    output_det_data_ = std::unique_ptr<float[]>(new float[output_det_size_]);
    output_desc_data_ = std::unique_ptr<float[]>(new float[output_desc_size_]);

    tpl_ptr_ =
        std::unique_ptr<Eigen::ThreadPool>(new Eigen::ThreadPool(num_threads_));
    dev_ptr_ = std::unique_ptr<Eigen::ThreadPoolDevice>(
        new Eigen::ThreadPoolDevice(tpl_ptr_.get(), num_threads_));
  }
  void loadTrtEngine();

  void preprocessImage(cv::Mat &img, cv::Mat &projection_matrix,
                       const int curr_batch);
  void runNeuralNetwork();
  void
  processOneHeatmap(const Eigen::Tensor<float, 3, Eigen::RowMajor> &heatmap,
                    const int curr_batch);
  void postprocessDetectionAndDescription();
  Eigen::VectorXf
  bilinearInterpolationDesc(const Eigen::Tensor<float, 4, Eigen::RowMajor>
                                &output_desc_tensor_transposed,
                            const int row, const int col, const int curr_batch);
  void debugOneBatchOutput();
  void addStereoImagePair(cv::Mat &img_l, cv::Mat &img_r,
                          const cv::Mat &projection_matrix_l,
                          const cv::Mat &projection_matrix_r);

  inline int getInputHeight() const { return input_height_; }
  inline int getInputWidth() const { return input_width_; }

private:
  const std::string model_name_prefix_;
  // this param is only for comparing the running time now. For better
  // performance, the higher the better. Uppper bound is the num of cameras.
  const int model_batch_size_;
  const std::string machine_name_;
  const TensorRtPrecision trt_precision_;

  // total IO ports, 1 input, 2 final outputs
  static constexpr int BUFFER_SIZE = 3;

  const int input_size_;

  // output det size = output det channel num * output width * output height
  const int output_det_size_;
  static constexpr int output_det_channel_ = 65;
  static constexpr int output_det_heatmap_factor_ = 8;
  // output desc size = output desc channel num * output width * output height
  const int output_desc_size_;
  static constexpr int output_desc_channel_ = 256;
  // width and height are shared by the detector and descriptor
  const int output_width_;
  const int output_height_;

  // postprocessing param
  const float conf_thresh_;
  const int dist_thresh_;
  const int border_remove_;
  static constexpr int max_keypoints_ = 1000;

  std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime *)>>
      runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine,
                  std::function<void(nvinfer1::ICudaEngine *)>>
      engine_;
  std::unique_ptr<nvinfer1::IExecutionContext,
                  std::function<void(nvinfer1::IExecutionContext *)>>
      context_;
  cudaStream_t stream_;
  // buffers for input and output data
  // std::vector<void *> buffers_ = std::vector<void *>(BUFFER_SIZE);
  std::array<void *, BUFFER_SIZE> buffers_;
  std::unique_ptr<float[]> input_data_;
  std::unique_ptr<float[]> output_det_data_;
  std::unique_ptr<float[]> output_desc_data_;

  // Performance engineering
  const int num_threads_;
  // Define a parallel device
  std::unique_ptr<Eigen::ThreadPool> tpl_ptr_;
  std::unique_ptr<Eigen::ThreadPoolDevice> dev_ptr_;
};
