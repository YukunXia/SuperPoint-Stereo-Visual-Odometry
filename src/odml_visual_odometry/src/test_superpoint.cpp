
#include <odml_visual_odometry/feature_detection.hpp>

// For NN output verification.
int main(int argc, char **argv) {
  SuperPointFeatureFrontEnd feature_front_end;
  // default width = 396 height = 120
  // const cv::Mat image =
  //     cv::Mat::ones(cv::Size(feature_front_end.getInputWidth(),
  //                            feature_front_end.getInputHeight()),
  //                   CV_32FC1);

  cv::Mat image = cv::imread("/home/yukun/ROS_doc/SuperPointPretrainedNetwork/"
                             "image_00/data/0000000010.png",
                             cv::IMREAD_GRAYSCALE);
  cv::resize(image, image,
             cv::Size(feature_front_end.getInputWidth(),
                      feature_front_end.getInputHeight()));
  image.convertTo(image, CV_32FC1, 1.0f / 255.0f);

  // for (int i = 0; i < 100; ++i) {
    feature_front_end.runNeuralNetwork(image);
    std::vector<cv::KeyPoint> keypoints =
        feature_front_end.postprocessDetection();
  // }

  image.convertTo(image, cv::COLOR_GRAY2BGR, 255.0f);
  cv::drawKeypoints(image, keypoints, image);
  // Display the image.
  cv::resize(image, image,
             cv::Size((int)feature_front_end.getInputWidth() * 3.3,
                      (int)feature_front_end.getInputHeight() * 3.3));
  cv::imshow("image", image);

  // Wait for a keystroke.
  cv::waitKey(0);

  // Destroys all the windows created
  cv::destroyAllWindows();

  // After Quantization to FP16
  // output_det[0,0,0] =
  // -30.3594, -22.4219, -22.1719, -21.9531, -21.4531,
  // -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594,
  // -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594,
  // -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594,
  // -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594,
  // -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594, -21.8594,
  // -21.8594, -21.8594, -21.8594, -21.8594, -21.8125, -21.375, -21.3594,
  // -21.1719, -25.4375,

  // output_desc[0,0,0] =
  //  0.0144905, 0.00518032, -0.00114253, -0.0273739,
  // -0.0272855, -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455,
  // -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455,
  // -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455,
  // -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455,
  // -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455,
  // -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0453455,
  // -0.0453455, -0.0453455, -0.0453455, -0.0453455, -0.0430273, -0.0298041,
  // -0.0561964, -0.0587825, -0.0680884,

  // The results should be close to
  // pytorch_result[0][0,0,0]
  // [-30.4029, -22.3350, -22.0438, -21.8212, -21.3145, -21.6777, -21.6777,
  //  -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777,
  //  -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777,
  //  -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777,
  //  -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777,
  //  -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777, -21.6777,
  //  -21.6777, -21.6777, -21.6498, -21.2709, -21.2199, -21.0500, -25.3817]
  // And pytorch_result[1][0,0,0]
  //  [ 0.0161,  0.0051, -0.0048, -0.0342, -0.0326, -0.0503, -0.0503, -0.0503,
  //   -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503,
  //   -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503,
  //   -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503,
  //   -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503, -0.0503,
  //   -0.0503, -0.0503, -0.0503, -0.0503, -0.0484, -0.0351, -0.0599, -0.0618,
  //   -0.0719]
  // From the following python code
  // img = np.ones((120, 396), dtype=np.float32)
  // pytorch_result = net(img_tensor)

  return 0;
}