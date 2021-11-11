#include <odml_visual_odometry/feature_detection.hpp>

#include <eigen3/Eigen/Sparse>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
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
}

std::vector<cv::KeyPoint> SuperPointFeatureFrontEnd::DebugOneBatchOutput() {
  // For NN output verification.
  // [0,0,0,:] is the equivalent numpy slicing code for BxCxHxW format tensors
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-format-desc
  // https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html
  // For now, B is eliminated temporarily
  std::cout << "Raw: output_det[0,0,0,:] = ";
  for (int w = 0; w < output_width_; ++w) {
    std::cout << output_det_data_.get()[w] << ", ";
  }
  std::cout << std::endl;

  std::cout << "\nRaw: output_desc[0,0,0,:] = ";
  for (int w = 0; w < output_width_; ++w) {
    std::cout << output_desc_data_.get()[w] << ", ";
  }
  std::cout << std::endl;

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> output_det_tensor(
      output_det_data_.get(), output_det_channel_, output_height_,
      output_width_);
  // Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>
  //     output_det_tensor_row_major(output_det_data_.get(),
  //     output_det_channel_,
  //                                 output_height_, output_width_);
  // const std::array<int, 3> shuffle = {2, 1, 0};
  // dim: output_det_channel_, output_height_, output_width_
  // eg. 65, 120, 396
  // Eigen::Tensor<float, 3, Eigen::RowMajor> output_det_tensor =
  // output_det_tensor_row_major.swap_layout().shuffle(shuffle);
  assert(output_det_tensor.dimension(0) == output_det_channel_);
  assert(output_det_tensor.dimension(1) == output_height_);
  assert(output_det_tensor.dimension(2) == output_width_);

  std::cout << "\nEigen: output_det[0,0,0,:] = ";
  for (int w = 0; w < output_width_; ++w) {
    std::cout << output_det_tensor(0, 0, w) << ", ";
  }
  std::cout << std::endl;
  // const Eigen::Tensor<float, 1> output_det_tensor_0c_0r =
  // output_det_tensor.chip(0,0).chip(0,0); std::cout << output_det_tensor_0c_0r
  // << std::endl;

  std::cout << "\nEigen: output_det[0,0,1,:] = ";
  for (int w = 0; w < output_width_; ++w) {
    std::cout << output_det_tensor(0, 1, w) << ", ";
  }
  std::cout << std::endl;

  std::cout << "\nEigen: output_det[0,:,0,0] = ";
  for (int c = 0; c < output_det_channel_; ++c) {
    std::cout << output_det_tensor(c, 0, 0) << ", ";
  }
  std::cout << std::endl;

  output_det_tensor = output_det_tensor.exp().eval();

  // std::cout << "\nEigen: np.exp(output_det[0,:,0,0]) = ";
  // float sum_0_0 = 0.0f;
  // for (int c = 0; c < output_det_channel_; ++c) {
  //   std::cout << output_det_tensor(c, 0, 0) << ", ";
  //   sum_0_0 += output_det_tensor(c, 0, 0);
  // }
  // std::cout << "\ntotal sum is " << sum_0_0 << std::endl;

  const Eigen::Tensor<float, 3, Eigen::RowMajor> output_det_tensor_channel_sum =
      output_det_tensor.sum(Eigen::array<int, 1>({0}))
          .reshape(Eigen::array<int, 3>({1, output_height_, output_width_}));
  assert(output_det_tensor_channel_sum.dimension(0) == 1);
  assert(output_det_tensor_channel_sum.dimension(1) == output_height_);
  assert(output_det_tensor_channel_sum.dimension(2) == output_width_);
  std::cout << "\ntotal sum of row 0 col 0 is "
            << output_det_tensor_channel_sum(0, 0, 0) << std::endl;
  std::cout << "\ntotal sum of row -1 col -1 is "
            << output_det_tensor_channel_sum(0, output_height_ - 1,
                                             output_width_ - 1)
            << std::endl;

  output_det_tensor /=
      (output_det_tensor_channel_sum +
       output_det_tensor_channel_sum.constant(0.00001f))
          .broadcast(Eigen::array<int, 3>({output_det_channel_, 1, 1}));

  std::cout << "\nEigen: after softmax, output_det[0,:,0,0] = ";
  for (int c = 0; c < output_det_channel_; ++c) {
    std::cout << output_det_tensor(c, 0, 0) << ", ";
  }
  std::cout << std::endl;

  std::cout << "\nEigen: after softmax, output_det[0,:,-1,-1] = ";
  for (int c = 0; c < output_det_channel_; ++c) {
    std::cout << output_det_tensor(c, output_height_ - 1, output_width_ - 1)
              << ", ";
  }
  std::cout << std::endl;

  // aliasing effect won't be avoided even with eval, probably because dims are
  // changed
  Eigen::Tensor<float, 3, Eigen::RowMajor> output_det_tensor_nodust =
      output_det_tensor.slice(
          Eigen::array<int, 3>({0, 0, 0}),
          Eigen::array<int, 3>(
              {output_det_channel_ - 1, output_height_, output_width_}));
  assert(output_det_tensor_nodust.dimension(0) == output_det_channel_ - 1);
  assert(output_det_tensor_nodust.dimension(1) == output_height_);
  assert(output_det_tensor_nodust.dimension(2) == output_width_);

  std::cout << "\nEigen: after removing the last dim, output_det[0,:,0,0] = ";
  for (int c = 0; c < output_det_channel_ - 1; ++c) {
    std::cout << output_det_tensor_nodust(c, 0, 0) << ", ";
  }
  std::cout << std::endl;

  std::cout << "\nEigen: after removing the last dim, output_det[0,:,-1,-1] = ";
  for (int c = 0; c < output_det_channel_ - 1; ++c) {
    std::cout << output_det_tensor_nodust(c, output_height_ - 1,
                                          output_width_ - 1)
              << ", ";
  }
  std::cout << std::endl;

  const Eigen::Tensor<float, 3, Eigen::RowMajor>
      output_det_tensor_nodust_transposed =
          output_det_tensor_nodust.shuffle(Eigen::array<int, 3>({1, 2, 0}));

  assert(output_det_tensor_nodust_transposed.dimension(0) == output_height_);
  assert(output_det_tensor_nodust_transposed.dimension(1) == output_width_);
  assert(output_det_tensor_nodust_transposed.dimension(2) ==
         output_det_channel_ - 1);
  assert(output_det_tensor_nodust_transposed(0, 0, 1) ==
         output_det_tensor_nodust(1, 0, 0));
  assert(output_det_tensor_nodust_transposed(output_height_ - 1,
                                             output_width_ - 1, 1) ==
         output_det_tensor_nodust(1, output_height_ - 1, output_width_ - 1));

  std::cout << "\nEigen: after transposed, output_det[0,0,0,:] = ";
  for (int c = 0; c < output_det_channel_ - 1; ++c) {
    std::cout << output_det_tensor_nodust_transposed(0, 0, c) << ", ";
  }
  std::cout << std::endl;

  std::cout << "\nEigen: after transposed, output_det[0,-1,-1,:] = ";
  for (int c = 0; c < output_det_channel_ - 1; ++c) {
    std::cout << output_det_tensor_nodust_transposed(output_height_ - 1,
                                                     output_width_ - 1, c)
              << ", ";
  }
  std::cout << std::endl;

  const Eigen::Tensor<float, 4, Eigen::RowMajor>
      output_det_tensor_nodust_transposed_reshaped =
          output_det_tensor_nodust_transposed.reshape(Eigen::array<int, 4>(
              {output_height_, output_width_, output_det_heatmap_factor_,
               output_det_heatmap_factor_}));
  assert(output_det_tensor_nodust_transposed_reshaped.dimension(0) ==
         output_height_);
  assert(output_det_tensor_nodust_transposed_reshaped.dimension(1) ==
         output_width_);
  assert(output_det_tensor_nodust_transposed_reshaped.dimension(2) ==
         output_det_heatmap_factor_);
  assert(output_det_tensor_nodust_transposed_reshaped.dimension(3) ==
         output_det_heatmap_factor_);

  std::cout << "\nEigen: after reshaped, output_det[0,0,0,:,:] = ";
  for (int chn_r = 0; chn_r < output_det_heatmap_factor_; ++chn_r) {
    for (int chn_c = 0; chn_c < output_det_heatmap_factor_; ++chn_c) {
      std::cout << output_det_tensor_nodust_transposed_reshaped(0, 0, chn_r,
                                                                chn_c)
                << ", ";
    }
  }
  std::cout << std::endl;

  std::cout << "\nEigen: after reshaped, output_det[0,-1,-1,:,:] = ";
  for (int chn_r = 0; chn_r < output_det_heatmap_factor_; ++chn_r) {
    for (int chn_c = 0; chn_c < output_det_heatmap_factor_; ++chn_c) {
      std::cout << output_det_tensor_nodust_transposed_reshaped(
                       output_height_ - 1, output_width_ - 1, chn_r, chn_c)
                << ", ";
    }
  }
  std::cout << std::endl;

  const Eigen::Tensor<float, 2, Eigen::RowMajor> heatmap =
      output_det_tensor_nodust_transposed_reshaped
          .shuffle(Eigen::array<int, 4>({0, 2, 1, 3}))
          .reshape(Eigen::array<int, 2>(
              {output_height_ * output_det_heatmap_factor_,
               output_width_ * output_det_heatmap_factor_}));

  assert(heatmap.dimension(0) == output_height_ * output_det_heatmap_factor_);
  assert(heatmap.dimension(1) == output_width_ * output_det_heatmap_factor_);
  std::cout << "\nEigen: heatmap [:5,:5] = \n";
  std::cout << heatmap.slice(Eigen::array<int, 2>({0, 0}),
                             Eigen::array<int, 2>({5, 5}))
            << std::endl;
  std::cout << "\nEigen: heatmap [-5:,-5:] = \n";
  std::cout << heatmap.slice(
                   Eigen::array<int, 2>(
                       {output_height_ * output_det_heatmap_factor_ - 5,
                        output_width_ * output_det_heatmap_factor_ - 5}),
                   Eigen::array<int, 2>({5, 5}))
            << std::endl;
  std::cout << "output_height_ * output_det_heatmap_factor_ - 5 = "
            << output_height_ * output_det_heatmap_factor_ - 5
            << ", output_width_ * output_det_heatmap_factor_ - 5 = "
            << output_width_ * output_det_heatmap_factor_ - 5 << std::endl;

  const Eigen::Tensor<float, 2, Eigen::RowMajor> heatmap_padded =
      heatmap.pad(Eigen::array<std::pair<int, int>, 2>(
          {std::pair<int, int>{dist_thresh_, dist_thresh_},
           std::pair<int, int>{dist_thresh_, dist_thresh_}}));
  assert(heatmap_padded.dimension(0) ==
         dist_thresh_ * 2 + output_height_ * output_det_heatmap_factor_);
  assert(heatmap_padded.dimension(1) ==
         dist_thresh_ * 2 + output_width_ * output_det_heatmap_factor_);
  std::cout << "\nEigen: heatmap_padded [:5,:5] = \n";
  std::cout << heatmap_padded.slice(Eigen::array<int, 2>({0, 0}),
                                    Eigen::array<int, 2>({5, 5}))
            << std::endl;
  std::cout << "\nEigen: heatmap_padded [-5:,-5:] = \n";
  std::cout << heatmap_padded.slice(
                   Eigen::array<int, 2>(
                       {dist_thresh_ * 2 +
                            output_height_ * output_det_heatmap_factor_ - 5,
                        dist_thresh_ * 2 +
                            output_width_ * output_det_heatmap_factor_ - 5}),
                   Eigen::array<int, 2>({5, 5}))
            << std::endl;

  // const Eigen::Tensor<float, 3, Eigen::RowMajor> heatmap_maxpooled =
  //     heatmap_padded
  //         .extract_patches(Eigen::array<int, 2>(
  //             {dist_thresh_ * 2 + 1, dist_thresh_ * 2 + 1}))
  //         .reshape(Eigen::array<int, 3>(
  //             {output_height_ * output_det_heatmap_factor_,
  //              output_width_ * output_det_heatmap_factor_,
  //              (dist_thresh_ * 2 + 1) * (dist_thresh_ * 2 + 1)}));
  // std::cout << "heatmap_maxpooled.dimension(0) = "
  //           << heatmap_maxpooled.dimension(0) << std::endl;
  // std::cout << "heatmap_maxpooled.dimension(1) = "
  //           << heatmap_maxpooled.dimension(1) << std::endl;
  // std::cout << "heatmap_maxpooled.dimension(2) = "
  //           << heatmap_maxpooled.dimension(2) << std::endl;

  const Eigen::Tensor<float, 2, Eigen::RowMajor> heatmap_maxpooled =
      heatmap_padded
          .extract_patches(Eigen::array<int, 2>(
              {dist_thresh_ * 2 + 1, dist_thresh_ * 2 + 1}))
          .reshape(Eigen::array<int, 3>(
              {output_height_ * output_det_heatmap_factor_,
               output_width_ * output_det_heatmap_factor_,
               (dist_thresh_ * 2 + 1) * (dist_thresh_ * 2 + 1)}))
          .maximum(Eigen::array<int, 1>({2}));
  assert(heatmap_maxpooled.dimension(0) ==
         output_height_ * output_det_heatmap_factor_);
  assert(heatmap_maxpooled.dimension(1) ==
         output_width_ * output_det_heatmap_factor_);
  std::cout << "\nEigen: heatmap_maxpooled [:5,:5] = \n";
  std::cout << heatmap_maxpooled.slice(Eigen::array<int, 2>({0, 0}),
                                       Eigen::array<int, 2>({5, 5}))
            << std::endl;
  std::cout << "\nEigen: heatmap_maxpooled [-5:,-5:] = \n";
  std::cout << heatmap_maxpooled.slice(
                   Eigen::array<int, 2>(
                       {output_height_ * output_det_heatmap_factor_ - 5,
                        output_width_ * output_det_heatmap_factor_ - 5}),
                   Eigen::array<int, 2>({5, 5}))
            << std::endl;

  Eigen::Tensor<bool, 2, Eigen::RowMajor> heatmap_cmp =
      (heatmap == heatmap_maxpooled);
  std::cout << "\nEigen: heatmap_cmp [:10,:10] = \n";
  std::cout << heatmap_cmp.slice(Eigen::array<int, 2>({0, 0}),
                                 Eigen::array<int, 2>({10, 10}))
            << std::endl;
  std::cout << "\nEigen: heatmap_cmp [-10:,-10:] = \n";
  std::cout << heatmap_cmp.slice(
                   Eigen::array<int, 2>(
                       {output_height_ * output_det_heatmap_factor_ - 10,
                        output_width_ * output_det_heatmap_factor_ - 10}),
                   Eigen::array<int, 2>({10, 10}))
            << std::endl;

  // const Eigen::SparseMatrix<float> sparse = heatmap_cmp.
  Eigen::Map<
      Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      heatmap_cmp_matrix(heatmap_cmp.data(),
                         output_height_ * output_det_heatmap_factor_,
                         output_width_ * output_det_heatmap_factor_);
  std::cout << "\nEigen: heatmap_cmp_matrix [:10,:10] = \n";
  std::cout << heatmap_cmp_matrix.topLeftCorner(10, 10) << std::endl;
  std::cout << "\nEigen: heatmap_cmp_matrix [-10:,-10:] = \n";
  std::cout << heatmap_cmp_matrix.bottomRightCorner(10, 10) << std::endl;

  Eigen::SparseMatrix<bool> heatmap_cmp_sparse =
      heatmap_cmp_matrix.sparseView();
  std::cout << "innerSize = " << heatmap_cmp_sparse.innerSize()
            << ", outerSize = " << heatmap_cmp_sparse.outerSize() << std::endl;

  int i = 0;
  std::vector<cv::KeyPoint> keypoints;
  for (int k = 0; k < heatmap_cmp_sparse.outerSize(); ++k)
    for (Eigen::SparseMatrix<bool>::InnerIterator it(heatmap_cmp_sparse, k); it;
         ++it) {
      const float val = heatmap(it.row(), it.col());
      if (val > conf_thresh_) {
        // std::cout << "row = " << it.row() << ", col = " << it.col()
        //           << ", value = " << val << std::endl;
        keypoints.emplace_back(cv::Point2f((float)it.col(), (float)it.row()),
                               1);
        ++i;
      }
    }
  std::cout << i << " points detected" << std::endl;

  return keypoints;
}

void SuperPointFeatureFrontEnd::addStereoImagePair(
    const cv::Mat &img_l, const cv::Mat &img_r,
    const cv::Mat &projection_matrix_l, const cv::Mat &projection_matrix_r) {
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