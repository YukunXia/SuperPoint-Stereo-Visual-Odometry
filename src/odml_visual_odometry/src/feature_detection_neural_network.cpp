#include <odml_visual_odometry/feature_detection.hpp>

#include <fstream>
#include <ros/package.h>

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
      std::to_string(input_width_) + ".engine";

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
    assert(projection_matrix.type() & CV_MAT_DEPTH_MASK == CV_32F);
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
    assert(projection_matrix.type() & CV_MAT_DEPTH_MASK == CV_32F);
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

void SuperPointFeatureFrontEnd::runNeuralNetwork(const cv::Mat &img, const bool debug_mode) {
  // Input value type should be float. In the future, it's easy to enable double
  // mode.
  assert((img.type() & CV_MAT_DEPTH_MASK) == CV_32F);
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
  ROS_INFO("processing 1 image by neural network takes %ld ms",
           std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count());

  if (debug_mode) {
    // For NN output verification. 
    std::cout << "output_det[0,0,0] = ";
    for (int w = 0; w < output_width_; ++w) {
      std::cout << output_det_data_.get()[w] << ", ";
    }
    std::cout << std::endl;

    std::cout << "\noutput_desc[0,0,0] = ";
    for (int w = 0; w < output_width_; ++w) {
      std::cout << output_desc_data_.get()[w] << ", ";
    }
    std::cout << std::endl;
  }
}

void SuperPointFeatureFrontEnd::addStereoImagePair(
    const cv::Mat &img_l, const cv::Mat &img_r,
    const cv::Mat &projection_matrix_l, const cv::Mat &projection_matrix_r) {
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