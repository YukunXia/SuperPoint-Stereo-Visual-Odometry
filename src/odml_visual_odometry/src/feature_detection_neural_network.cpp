#include <odml_visual_odometry/feature_detection.hpp>

#include <fstream>
#include <ros/package.h>

#define EIGEN_USE_THREADS
#include <eigen3/Eigen/SparseCore>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                    \
  {                                                                            \
    cudaError_t error_code = callstr;                                          \
    if (error_code != cudaSuccess) {                                           \
      ROS_ERROR("CUDA error %d: %s at ", (int)error_code,                      \
                cudaGetErrorString(error_code));                               \
    }                                                                          \
  }
#endif

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////neural_network_classe_func_def//////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void SuperPointFeatureFrontEnd::loadTrtEngine() {
  const std::string model_name_full =
      ros::package::getPath("odml_visual_odometry") + "/models/" +
      model_name_prefix_ + "_" + std::to_string(input_height_) + "_" +
      std::to_string(input_width_) + "_" +
      trt_precision_enum2string.at(trt_precision_) + ".engine";

  std::ifstream engine_file(model_name_full, std::ios::binary);

  if (!engine_file.good()) {
    ROS_ERROR("no such engine file: %s", model_name_full.c_str());
    return;
  }

  engine_file.seekg(0, engine_file.end);
  const size_t trt_stream_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);

  trt_model_stream_ = std::unique_ptr<char[]>(new char[trt_stream_size]);
  assert(trt_model_stream_);
  engine_file.read(trt_model_stream_.get(), trt_stream_size);
  engine_file.close();

  runtime_ = nvinfer1::createInferRuntime(g_logger_);
  assert(runtime_ != nullptr);
  engine_ =
      runtime_->deserializeCudaEngine(trt_model_stream_.get(), trt_stream_size);
  assert(engine_ != nullptr);
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr);
  if (engine_->getNbBindings() != BUFFER_SIZE) {
    ROS_ERROR("engine->getNbBindings() == %d, but should be %d",
              engine_->getNbBindings(), BUFFER_SIZE);
    return;
  }

  // get sizes of input and output and allocate memory required for input data
  // and for output data
  std::vector<nvinfer1::Dims> input_dims;
  std::vector<nvinfer1::Dims> output_dims;
  for (size_t i = 0; i < engine_->getNbBindings(); ++i) {
    // get binding total size
    size_t binding_size = 1;
    const nvinfer1::Dims &dims = engine_->getBindingDimensions(i);
    // print dimensions of each layer
    std::cout << "layer " << i << ": ";
    for (size_t j = 0; j < dims.nbDims; ++j) {
      binding_size *= dims.d[j];
      std::cout << dims.d[j] << ", ";
    }
    std::cout << std::endl;
    binding_size *= sizeof(float);

    if (binding_size == 0) {
      ROS_ERROR("binding_size == 0");
      return;
    }

    cudaMalloc(&buffers_[i], binding_size);
    if (engine_->bindingIsInput(i)) {
      input_dims.emplace_back(engine_->getBindingDimensions(i));
      ROS_INFO("Input layer, size = %lu", binding_size);
    } else {
      output_dims.emplace_back(engine_->getBindingDimensions(i));
      ROS_INFO("Output layer, size = %lu", binding_size);
    }
  }

  CUDA_CHECK(cudaStreamCreate(&stream_));

  ROS_INFO("Engine preparation finished");
}

void SuperPointFeatureFrontEnd::preprocessImage(cv::Mat &img,
                                                cv::Mat &projection_matrix) {
  int img_rows = img.rows;
  int img_cols = img.cols;

  // Step 1: Crop image to certain aspect ratio
  const float real_aspect_ratio =
      static_cast<float>(img_cols) / static_cast<float>(img_rows);
  const float expected_aspect_ratio =
      static_cast<float>(input_width_) / static_cast<float>(input_height_);

  if (expected_aspect_ratio > real_aspect_ratio) {
    // eg.
    // expected              real
    // [             ]       [   ]
    // [             ]       [   ]
    // [             ]       [   ]
    img_rows = img_cols / expected_aspect_ratio;
    const int img_rows_offset = (img.rows - img_rows) / 2;

    img = img.rowRange(img_rows_offset, img_rows_offset + img_rows);

    // (row => ) y offset in P or K should be decreased
    // ref: https://ksimek.github.io/2013/08/13/intrinsic/
    assert((projection_matrix.type() & CV_MAT_DEPTH_MASK) == CV_32F &&
           (1 + (projection_matrix.type() >> CV_CN_SHIFT)) == 1);
    projection_matrix.at<float>(1, 2) -= static_cast<float>(img_rows_offset);
  } else if (expected_aspect_ratio < real_aspect_ratio) {
    // eg.
    // expected              real
    // [             ]       [                           ]
    // [             ]       [                           ]
    // [             ]       [                           ]
    img_cols = img_rows * expected_aspect_ratio;
    const int img_cols_offset = (img.cols - img_cols) / 2;

    img = img.colRange(img_cols_offset, img_cols_offset + img_cols);

    // (col => ) x offset in P or K should be decreased
    // ref: https://ksimek.github.io/2013/08/13/intrinsic/
    assert((projection_matrix.type() & CV_MAT_DEPTH_MASK) == CV_32F &&
           (1 + (projection_matrix.type() >> CV_CN_SHIFT)) == 1);
    projection_matrix.at<float>(0, 2) -= static_cast<float>(img_cols_offset);
  }

  // Step 2: Resize the image
  cv::resize(img, img, cv::Size(input_width_, input_height_), 0.0, 0.0,
             cv::INTER_LINEAR);
  // superpoint takes in normalized data ranging from 0.0 to 1.0
  // From demo_superpoint.py:
  // > input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
  // > input_image = input_image.astype('float')/255.0
  img.convertTo(img, CV_32FC1, 1.0f / 255.0f);

  const float dst_size_over_src_size =
      static_cast<float>(input_width_) / static_cast<float>(img_cols);
  projection_matrix.rowRange(0, 2) *= dst_size_over_src_size;
}

void SuperPointFeatureFrontEnd::runNeuralNetwork(const cv::Mat &img) {
  int i = 0;
  for (int row = 0; row < input_height_; ++row) {
    for (int col = 0; col < input_width_; ++col) {
      input_data_.get()[i] = img.at<float>(row, col);
      ++i;
    }
  }
  assert(i == input_size_);

  const auto start = std::chrono::system_clock::now();

  CUDA_CHECK(cudaMemcpyAsync(buffers_[0], input_data_.get(),
                             input_size_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));
  context_->enqueue(1, buffers_.data(), stream_, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output_det_data_.get(), buffers_[1],
                             output_det_size_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(output_desc_data_.get(), buffers_[2],
                             output_desc_size_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  cudaStreamSynchronize(stream_);

  const auto end = std::chrono::system_clock::now();
  ROS_INFO(
      "processing 1 image by neural network takes %.4f ms",
      (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() /
          1000.0f);
}

std::vector<cv::KeyPoint> SuperPointFeatureFrontEnd::postprocessDetection() {
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> output_det_tensor(
      output_det_data_.get(), output_det_channel_, output_height_,
      output_width_);
  output_det_tensor.device(*dev_ptr_) = output_det_tensor.exp().eval();
  Eigen::Tensor<float, 3, Eigen::RowMajor> output_det_tensor_channel_sum(
      1, output_height_, output_width_);
  output_det_tensor_channel_sum.device(*dev_ptr_) =
      output_det_tensor.sum(Eigen::array<int, 1>({0}))
          .reshape(Eigen::array<int, 3>({1, output_height_, output_width_}));
  output_det_tensor.device(*dev_ptr_) =
      output_det_tensor /
      (output_det_tensor_channel_sum +
       output_det_tensor_channel_sum.constant(0.00001f))
          .broadcast(Eigen::array<int, 3>({output_det_channel_, 1, 1}));
  Eigen::Tensor<float, 3, Eigen::RowMajor> output_det_tensor_nodust(
      output_det_channel_ - 1, output_height_, output_width_);
  output_det_tensor_nodust.device(*dev_ptr_) = output_det_tensor.slice(
      Eigen::array<int, 3>({0, 0, 0}),
      Eigen::array<int, 3>(
          {output_det_channel_ - 1, output_height_, output_width_}));
  Eigen::Tensor<float, 3, Eigen::RowMajor> output_det_tensor_nodust_transposed(
      output_height_, output_width_, output_det_channel_ - 1);
  output_det_tensor_nodust_transposed.device(*dev_ptr_) =
      output_det_tensor_nodust.shuffle(Eigen::array<int, 3>({1, 2, 0}));
  Eigen::Tensor<float, 4, Eigen::RowMajor>
      output_det_tensor_nodust_transposed_reshaped(
          output_height_, output_width_, output_det_heatmap_factor_,
          output_det_heatmap_factor_);
  output_det_tensor_nodust_transposed_reshaped.device(*dev_ptr_) =
      output_det_tensor_nodust_transposed.reshape(Eigen::array<int, 4>(
          {output_height_, output_width_, output_det_heatmap_factor_,
           output_det_heatmap_factor_}));
  Eigen::Tensor<float, 2, Eigen::RowMajor> heatmap(
      output_height_ * output_det_heatmap_factor_,
      output_width_ * output_det_heatmap_factor_);
  heatmap.device(*dev_ptr_) =
      output_det_tensor_nodust_transposed_reshaped
          .shuffle(Eigen::array<int, 4>({0, 2, 1, 3}))
          .reshape(Eigen::array<int, 2>(
              {output_height_ * output_det_heatmap_factor_,
               output_width_ * output_det_heatmap_factor_}));

  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      heatmap_matrix(heatmap.data(),
                     output_height_ * output_det_heatmap_factor_,
                     output_width_ * output_det_heatmap_factor_);
  const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      heatmap_cmp = (heatmap_matrix.array() > conf_thresh_);

  const Eigen::SparseMatrix<bool> heatmap_cmp_sparse = heatmap_cmp.sparseView();
  std::vector<cv::KeyPoint> keypoints_before_nms;
  for (int k = 0; k < heatmap_cmp_sparse.outerSize(); ++k)
    for (Eigen::SparseMatrix<bool>::InnerIterator it(heatmap_cmp_sparse, k); it;
         ++it) {
      keypoints_before_nms.emplace_back(
          cv::Point2f((float)it.col(), (float)it.row()),
          heatmap(it.row(), it.col()));
    }
  std::sort(keypoints_before_nms.begin(), keypoints_before_nms.end(),
            [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
              return p1.size > p2.size;
            });

  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> nms_matrix =
      Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Zero(
          heatmap_matrix.rows(), heatmap_matrix.cols());

  std::vector<cv::KeyPoint> keypoints_after_nms;
  keypoints_after_nms.reserve(keypoints_before_nms.size());
  for (const auto &keypoint : keypoints_before_nms) {
    const int row = keypoint.pt.y;
    const int col = keypoint.pt.x;
    if (nms_matrix(row, col) == false) {
      keypoints_after_nms.emplace_back(keypoint.pt, 1);
      for (int r = row - dist_thresh_; r < row + dist_thresh_ + 1; ++r) {
        if (r < 0 || r >= heatmap_matrix.rows())
          continue;
        for (int c = col - dist_thresh_; c < col + dist_thresh_ + 1; ++c) {
          if (c < 0 || c >= heatmap_matrix.cols())
            continue;
          nms_matrix(r, c) = true;
        }
      }
    }
  }

  return keypoints_after_nms;
}

std::vector<cv::KeyPoint> SuperPointFeatureFrontEnd::debugOneBatchOutput() {

  const auto start = std::chrono::system_clock::now();

  std::vector<cv::KeyPoint> keypoints_after_nms = postprocessDetection();

  const auto end = std::chrono::system_clock::now();
  ROS_INFO(
      "postprocessing detection of 1 image takes %.4f ms",
      (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() /
          1000.0f);

  ROS_INFO("%lu keypoints_after_nms", keypoints_after_nms.size());

  return keypoints_after_nms;
}

void SuperPointFeatureFrontEnd::addStereoImagePair(
    const cv::Mat &img_l, const cv::Mat &img_r,
    const cv::Mat &projection_matrix_l, const cv::Mat &projection_matrix_r) {
  // Input value type should be float. In the future, it's easy to enable double
  // mode.
  // Reference for type assertion
  // img_l.type() & CV_MAT_DEPTH_MASK is the type code
  // 1 + (img_r.type() >> CV_CN_SHIFT) is the channel number
  // https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
  assert((img_l.type() & CV_MAT_DEPTH_MASK) == CV_32F &&
         (1 + (img_l.type() >> CV_CN_SHIFT)) == 1);
  assert((img_r.type() & CV_MAT_DEPTH_MASK) == CV_32F &&
         (1 + (img_r.type() >> CV_CN_SHIFT)) == 1);

  cv::Mat img_l_fit = img_l.clone();
  cv::Mat img_r_fit = img_r.clone();

  projection_matrix_l_ = projection_matrix_l.clone();
  projection_matrix_r_ = projection_matrix_r.clone();

  preprocessImage(img_l_fit, projection_matrix_l_);
  preprocessImage(img_r_fit, projection_matrix_r_);
  assert(img_l_fit.rows == input_height_ && img_l_fit.cols == input_width_);
  assert(img_r_fit.rows == input_height_ && img_r_fit.cols == input_width_);

  assert(input_data_ != nullptr);
}

void SuperPointFeatureFrontEnd::matchDescriptors(const MatchType match_type) {}