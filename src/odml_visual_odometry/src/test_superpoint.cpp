
#include <odml_visual_odometry/feature_detection.hpp>
#include <ros/package.h>

// For NN output verification.
int main(int argc, char **argv) {
  SuperPointFeatureFrontEnd feature_front_end;

  for (int i = 0; i < 2; ++i) {

    cv::Mat image = cv::imread(
        ros::package::getPath("odml_visual_odometry") +
            "/sample_images/00000000" +
            ((i < 10) ? "0" + std::to_string(i) : std::to_string(i)) + ".png",
        cv::IMREAD_GRAYSCALE);
    cv::resize(image, image,
               cv::Size(feature_front_end.getInputWidth(),
                        feature_front_end.getInputHeight()));
    feature_front_end.images_dq.push_back(image);
    image.convertTo(image, CV_32FC1, 1.0f / 255.0f);
    feature_front_end.runNeuralNetwork(image);
    feature_front_end.debugOneBatchOutput();

    if (i == 0)
      continue;

    cv::Mat &descriptors0 = feature_front_end.descriptors_dq.end()[-1];
    cv::Mat &descriptors1 = feature_front_end.descriptors_dq.end()[-2];
    std::vector<cv::KeyPoint> &keypoints0 =
        feature_front_end.keypoints_dq.end()[-1];
    std::vector<cv::KeyPoint> &keypoints1 =
        feature_front_end.keypoints_dq.end()[-2];

    std::vector<cv::DMatch> cv_Dmatches;
    cv_Dmatches.clear();
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::BFMatcher::create(cv::NORM_L2, true);
    matcher->match(descriptors0, descriptors1, cv_Dmatches);

    // Create some random colors
    cv::Mat image1_color = feature_front_end.images_dq.back().clone();
    cv::cvtColor(image1_color, image1_color, cv::COLOR_GRAY2BGR);
    cv::RNG rng;
    cv::Scalar color;
    int r, g, b, j;
    for (const auto match : cv_Dmatches) {
      // draw the tracks
      r = rng.uniform(0, 256);
      g = rng.uniform(0, 256);
      b = rng.uniform(0, 256);
      color = cv::Scalar(r, g, b);
      cv::line(image1_color,
               feature_front_end.keypoints_dq.end()[-1][match.queryIdx].pt,
               feature_front_end.keypoints_dq.end()[-2][match.trainIdx].pt,
               color, 2);
      cv::circle(image1_color,
                 feature_front_end.keypoints_dq.end()[-1][match.queryIdx].pt, 3,
                 color, -1);
    }

    cv::resize(image1_color, image1_color,
               cv::Size(feature_front_end.getInputWidth() * 3.3,
                        feature_front_end.getInputHeight() * 3.3));
    cv::imshow("matching_image", image1_color);

    // Wait for a keystroke.
    cv::waitKey(100000);

    // Destroys all the windows created
    // cv::destroyAllWindows();
  }

  return 0;
}