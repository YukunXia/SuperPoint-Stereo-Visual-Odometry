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

SuperPointFeatureFrontEnd::~SuperPointFeatureFrontEnd() {
  CUDA_CHECK(cudaStreamDestroy(stream_));

  for (int i = 0; i < BUFFER_SIZE; ++i) {
    CUDA_CHECK(cudaFree(buffers_[i]));
  }

  // TODO: understand why context_->destroy will kill the process with exit code
  // -11 (Maybe upgrade TensorRT to 8.2)
  // I can not reference to the actual definition of nvinfer1::IExecutionContext
  // (Not open sourced) The following line is really just a walkaround. It's
  // likely to cause memory leak. The problem is super strange because runtime_
  // and engine_ have no issue calling the destroy function. Besides, the
  // destructors for TensorRT objects have already considered the validity
  // before trying to call the destroy function. A side note is that the destroy
  // function is said to be deprecated from TensorRT 10.0
  context_.release();
}

void SuperPointFeatureFrontEnd::loadTrtEngine() {
  const std::string model_name_full =
      ros::package::getPath("odml_visual_odometry") + "/models/" +
      machine_name_ + "/" + model_name_prefix_ + "_" +
      std::to_string(model_batch_size_) + "_" + std::to_string(input_height_) +
      "_" + std::to_string(input_width_) + "_" +
      trt_precision_enum2string.at(trt_precision_) + ".engine";

  std::ifstream engine_file(model_name_full, std::ios::binary);

  if (!engine_file.good()) {
    ROS_ERROR("no such engine file: %s", model_name_full.c_str());
    return;
  } else {
    ROS_INFO("engine file `%s` loaded", model_name_full.c_str());
  }

  engine_file.seekg(0, engine_file.end);
  const size_t trt_stream_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);

  std::unique_ptr<char[]> trt_model_stream(new char[trt_stream_size]);
  assert(trt_model_stream);
  engine_file.read(trt_model_stream.get(), trt_stream_size);
  engine_file.close();

  sample::Logger g_logger;
  runtime_ = std::unique_ptr<nvinfer1::IRuntime,
                             std::function<void(nvinfer1::IRuntime *)>>(
      nvinfer1::createInferRuntime(g_logger), [](nvinfer1::IRuntime *ptr) {
        ROS_INFO("runtime_ destructor");
        if (ptr)
          ptr->destroy();
      });
  assert(runtime_ != nullptr);
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine,
                            std::function<void(nvinfer1::ICudaEngine *)>>(
      runtime_->deserializeCudaEngine(trt_model_stream.get(), trt_stream_size),
      [](nvinfer1::ICudaEngine *ptr) {
        if (ptr)
          ptr->destroy();
      });
  assert(engine_ != nullptr);
  context_ =
      std::unique_ptr<nvinfer1::IExecutionContext,
                      std::function<void(nvinfer1::IExecutionContext *)>>(
          engine_->createExecutionContext(),
          [](nvinfer1::IExecutionContext *ptr) {
            if (ptr) {
              ptr->destroy();
            }
          });
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

    CUDA_CHECK(cudaMalloc(&buffers_[i], binding_size));
    if (engine_->bindingIsInput(i)) {
      input_dims.emplace_back(engine_->getBindingDimensions(i));
      ROS_INFO("Input layer, size = %lu", binding_size / sizeof(float));
    } else {
      output_dims.emplace_back(engine_->getBindingDimensions(i));
      ROS_INFO("Output layer, size = %lu", binding_size / sizeof(float));
    }
  }

  CUDA_CHECK(cudaStreamCreate(&stream_));

  ROS_INFO("Engine preparation finished");
}

void SuperPointFeatureFrontEnd::preprocessImage(cv::Mat &img,
                                                cv::Mat &projection_matrix,
                                                const int curr_batch) {
  assert(curr_batch < model_batch_size_);

  preprocessImageImpl(img, projection_matrix);

  // hand over data to input data
  // in this branch (master), only one gray image will be fed into NN. No need
  // to consider transposing HWC to CHW
  cv::Mat img_fit =
      cv::Mat(input_height_, input_width_, CV_32F,
              input_data_.get() + curr_batch * input_height_ * input_width_);

  // store image before transforming from 8U to 32F
  images_dq.push_back(img);
  // superpoint takes in normalized data ranging from 0.0 to 1.0
  // From demo_superpoint.py:
  // > input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
  // > input_image = input_image.astype('float')/255.0
  img.convertTo(img, CV_32FC1, 1.0f / 255.0f);
  img.copyTo(img_fit);
}

void SuperPointFeatureFrontEnd::runNeuralNetwork() {
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

  if (verbose_) {
    const auto end = std::chrono::system_clock::now();
    ROS_INFO("processing 1 image by neural network takes %.4f ms",
             (float)std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                     .count() /
                 1000.0f);
  }
}

void SuperPointFeatureFrontEnd::processOneHeatmap(
    const Eigen::Tensor<float, 3, Eigen::RowMajor> &heatmap,
    const int curr_batch) {
  float *heatmap_data = const_cast<float *>(heatmap.data()) +
                        output_height_ * output_det_heatmap_factor_ *
                            output_width_ * output_det_heatmap_factor_ *
                            curr_batch;
  std::vector<cv::KeyPoint> keypoints_after_nms;

  // xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      heatmap_matrix(heatmap_data, output_height_ * output_det_heatmap_factor_,
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
          heatmap(curr_batch, it.row(), it.col()));
    }
  std::sort(keypoints_before_nms.begin(), keypoints_before_nms.end(),
            [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
              return p1.size > p2.size;
            });

  // pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> nms_matrix =
      Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Zero(
          heatmap_matrix.rows(), heatmap_matrix.cols());

  // initialize output variables
  // keypoints_after_nms.clear();
  keypoints_after_nms.reserve(
      std::min(max_keypoints_, (int)keypoints_before_nms.size()));

  for (const auto &keypoint : keypoints_before_nms) {
    const int row = keypoint.pt.y;
    const int col = keypoint.pt.x;
    if (nms_matrix(row, col) == false) {
      // remove points close to borders
      // bord = self.border_remove
      // toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
      // toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
      // toremove = np.logical_or(toremoveW, toremoveH)
      // pts = pts[:, ~toremove]
      if (row >= border_remove_ &&
          row + border_remove_ < heatmap_matrix.rows() &&
          col >= border_remove_ &&
          col + border_remove_ < heatmap_matrix.cols()) {
        keypoints_after_nms.emplace_back(keypoint.pt, 1);
      }
      // suppress nearby points
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
    if (keypoints_after_nms.size() >= max_keypoints_)
      break;
  }

  // store valuable data into dqs
  keypoints_dq.push_back(keypoints_after_nms);
}

void SuperPointFeatureFrontEnd::postprocessDetectionAndDescription() {
  // transforming raw pointer data into Eigen tensors
  Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> output_det_tensor(
      output_det_data_.get(), model_batch_size_, output_det_channel_,
      output_height_, output_width_);

  // dense = np.exp(semi) # Softmax.
  output_det_tensor.device(*dev_ptr_) = output_det_tensor.exp().eval();

  // dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
  Eigen::Tensor<float, 4, Eigen::RowMajor> output_det_tensor_channel_sum(
      model_batch_size_, 1, output_height_, output_width_);
  output_det_tensor_channel_sum.device(*dev_ptr_) =
      output_det_tensor.sum(Eigen::array<int, 1>({1}))
          .reshape(Eigen::array<int, 4>(
              {model_batch_size_, 1, output_height_, output_width_}));
  output_det_tensor.device(*dev_ptr_) =
      output_det_tensor /
      (output_det_tensor_channel_sum +
       output_det_tensor_channel_sum.constant(0.00001f))
          .broadcast(Eigen::array<int, 4>({1, output_det_channel_, 1, 1}));
  // 1 means 1 copy or no broadcast on this dim

  // # Remove dustbin.
  // nodust = dense[:-1, :, :]
  Eigen::Tensor<float, 4, Eigen::RowMajor> output_det_tensor_nodust(
      model_batch_size_, output_det_channel_ - 1, output_height_,
      output_width_);
  output_det_tensor_nodust.device(*dev_ptr_) = output_det_tensor.slice(
      Eigen::array<int, 4>({0, 0, 0, 0}),
      Eigen::array<int, 4>({model_batch_size_, output_det_channel_ - 1,
                            output_height_, output_width_}));

  // nodust = nodust.transpose(1, 2, 0)
  Eigen::Tensor<float, 4, Eigen::RowMajor> output_det_tensor_nodust_transposed(
      model_batch_size_, output_height_, output_width_,
      output_det_channel_ - 1);
  output_det_tensor_nodust_transposed.device(*dev_ptr_) =
      output_det_tensor_nodust.shuffle(Eigen::array<int, 4>({0, 2, 3, 1}));

  // # Reshape to get full resolution heatmap.
  // Hc = int(H / self.cell)
  // Wc = int(W / self.cell)
  // heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
  Eigen::Tensor<float, 5, Eigen::RowMajor>
      output_det_tensor_nodust_transposed_reshaped(
          model_batch_size_, output_height_, output_width_,
          output_det_heatmap_factor_, output_det_heatmap_factor_);
  output_det_tensor_nodust_transposed_reshaped.device(*dev_ptr_) =
      output_det_tensor_nodust_transposed.reshape(Eigen::array<int, 5>(
          {model_batch_size_, output_height_, output_width_,
           output_det_heatmap_factor_, output_det_heatmap_factor_}));
  // heatmap = np.transpose(heatmap, [0, 2, 1, 3])
  // heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
  Eigen::Tensor<float, 3, Eigen::RowMajor> heatmap(
      model_batch_size_, output_height_ * output_det_heatmap_factor_,
      output_width_ * output_det_heatmap_factor_);
  heatmap.device(*dev_ptr_) =
      output_det_tensor_nodust_transposed_reshaped
          .shuffle(Eigen::array<int, 5>({0, 1, 3, 2, 4}))
          .reshape(Eigen::array<int, 3>(
              {model_batch_size_, output_height_ * output_det_heatmap_factor_,
               output_width_ * output_det_heatmap_factor_}));

  for (int b = 0; b < model_batch_size_; ++b) {
    processOneHeatmap(heatmap, b);
  }

  // transforming raw pointer data into Eigen tensors
  Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> output_desc_tensor(
      output_desc_data_.get(), model_batch_size_, output_desc_channel_,
      output_height_, output_width_);

  // move the channel to the last dim, so that the channelwise data will be
  // continuous
  Eigen::Tensor<float, 4, Eigen::RowMajor> output_desc_tensor_transposed(
      model_batch_size_, output_height_, output_width_, output_desc_channel_);
  output_desc_tensor_transposed.device(*dev_ptr_) =
      output_desc_tensor.shuffle(Eigen::array<int, 4>({0, 2, 3, 1}));

  for (int b = 0; b < model_batch_size_; ++b) {
    const std::vector<cv::KeyPoint> &keypoints_after_nms =
        keypoints_dq.end()[b - model_batch_size_];
    cv::Mat descriptors(keypoints_after_nms.size(), output_desc_channel_,
                        CV_32FC1);
    for (int i = 0; i < keypoints_after_nms.size(); ++i) {
      const cv::KeyPoint &keypoint = keypoints_after_nms[i];
      const int row = keypoint.pt.y;
      const int col = keypoint.pt.x;
      Eigen::VectorXf desc_interpolated =
          bilinearInterpolationDesc(output_desc_tensor_transposed, row, col, b);

      const cv::Mat desc_interpolated_cv_mat(1, output_desc_channel_, CV_32F,
                                             desc_interpolated.data());
      desc_interpolated_cv_mat.copyTo(descriptors.row(i));
    }

    // store valuable data into dqs
    descriptors_dq.push_back(descriptors);
  }
}

Eigen::VectorXf SuperPointFeatureFrontEnd::bilinearInterpolationDesc(
    const Eigen::Tensor<float, 4, Eigen::RowMajor>
        &output_desc_tensor_transposed,
    const int row, const int col, const int curr_batch) {
  // transforming row and col from heatmap coords into desc (downsampled/coarse)
  // coords
  // not exactly divided by 8. Eg. if input row = 360, desc row = 45, then r_359
  // in heatmap -> r_44 in desc
  // This is the same as torch.nn.functional.grid_sample, with align_corner =
  // True. It's by default eq to True in Pytorch 0.4, the version that the
  // pretrained superpoint uses, while later changed to False by default.
  const float row_by8 = static_cast<float>(row) /
                        static_cast<float>(input_height_ - 1) *
                        static_cast<float>(input_height_ / 8 - 1);
  const float col_by8 = static_cast<float>(col) /
                        static_cast<float>(input_width_ - 1) *
                        static_cast<float>(input_width_ / 8 - 1);

  // avoid overflow
  assert(std::ceil(row_by8) <= input_height_ / 8 - 1);
  assert(std::ceil(col_by8) <= input_width_ / 8 - 1);

  const int row_top_l = std::floor(row_by8);
  const int col_top_l = std::floor(col_by8);

  const float row_ratio = 1.0f - (row_by8 - static_cast<float>(row_top_l));
  const float col_ratio = 1.0f - (col_by8 - static_cast<float>(col_top_l));

  assert(row_ratio >= 0.0f && row_ratio < 1.0f);
  assert(col_ratio >= 0.0f && col_ratio < 1.0f);

  const int offset_top_l =
      output_desc_channel_ * (row_top_l * output_width_ + col_top_l) +
      curr_batch * output_desc_size_ / model_batch_size_;
  const int offset_top_r =
      output_desc_channel_ * (row_top_l * output_width_ + col_top_l + 1) +
      curr_batch * output_desc_size_ / model_batch_size_;
  const int offset_bot_l =
      output_desc_channel_ * ((row_top_l + 1) * output_width_ + col_top_l) +
      curr_batch * output_desc_size_ / model_batch_size_;
  const int offset_bot_r =
      output_desc_channel_ * ((row_top_l + 1) * output_width_ + col_top_l + 1) +
      curr_batch * output_desc_size_ / model_batch_size_;

  Eigen::Map<Eigen::VectorXf> desc_top_l(
      const_cast<float *>(output_desc_tensor_transposed.data()) + offset_top_l,
      output_desc_channel_);
  Eigen::Map<Eigen::VectorXf> desc_top_r(
      const_cast<float *>(output_desc_tensor_transposed.data()) + offset_top_r,
      output_desc_channel_);
  Eigen::Map<Eigen::VectorXf> desc_bot_l(
      const_cast<float *>(output_desc_tensor_transposed.data()) + offset_bot_l,
      output_desc_channel_);
  Eigen::Map<Eigen::VectorXf> desc_bot_r(
      const_cast<float *>(output_desc_tensor_transposed.data()) + offset_bot_r,
      output_desc_channel_);

  Eigen::VectorXf desc_interpolated =
      desc_top_l * row_ratio * col_ratio +
      desc_top_r * row_ratio * (1.0f - col_ratio) +
      desc_bot_l * (1.0f - row_ratio) * col_ratio +
      desc_bot_r * (1.0f - row_ratio) * (1.0f - col_ratio);
  desc_interpolated.normalize();

  return desc_interpolated;
}

void SuperPointFeatureFrontEnd::debugOneBatchOutput() {

  const auto start = std::chrono::system_clock::now();

  postprocessDetectionAndDescription();

  const auto end = std::chrono::system_clock::now();
  ROS_INFO(
      "postprocessing detection of 1 image takes %.4f ms",
      (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() /
          1000.0f);

  // ROS_INFO("%lu keypoints_after_nms", keypoints_after_nms.size());
}

void SuperPointFeatureFrontEnd::addStereoImagePair(
    cv::Mat &img_l, cv::Mat &img_r, const cv::Mat &projection_matrix_l,
    const cv::Mat &projection_matrix_r) {
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

  // const auto start = std::chrono::system_clock::now();

  projection_matrix_l_ = projection_matrix_l.clone();
  projection_matrix_r_ = projection_matrix_r.clone();

  if (model_batch_size_ == 1) {
    preprocessImage(img_l, projection_matrix_l_, 0);
    runNeuralNetwork();
    postprocessDetectionAndDescription();

    preprocessImage(img_r, projection_matrix_r_, 0);
    runNeuralNetwork();
    postprocessDetectionAndDescription();

    if (verbose_)
      ROS_INFO("%lu, %lu keypoints for img_l and img_r",
               keypoints_dq.end()[-2].size(), keypoints_dq.end()[-1].size());
  } else if (model_batch_size_ == 2) {
    preprocessImage(img_l, projection_matrix_l_, 0);
    preprocessImage(img_r, projection_matrix_r_, 1);
    runNeuralNetwork();
    postprocessDetectionAndDescription();

    if (verbose_)
      ROS_INFO("%lu, %lu keypoints for img_l and img_r",
               keypoints_dq.end()[-2].size(), keypoints_dq.end()[-1].size());
  } else {
    ROS_ERROR("Wrong batch size (%d)", model_batch_size_);
    return;
  }

  while (images_dq.size() > 4) {
    images_dq.pop_front();
    keypoints_dq.pop_front();
    descriptors_dq.pop_front();
  }

  assert(keypoints_dq.size() <= 4);
  assert(descriptors_dq.size() <= 4);

  // const auto end = std::chrono::system_clock::now();
  // ROS_INFO(
  //     "(pre, mid, post)processing detection of 1 image takes %.4f ms",
  //     (float)std::chrono::duration_cast<std::chrono::microseconds>(end -
  //     start)
  //             .count() /
  //         1000.0f);
}