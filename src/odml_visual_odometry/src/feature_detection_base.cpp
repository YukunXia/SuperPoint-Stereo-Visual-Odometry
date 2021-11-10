#include <odml_visual_odometry/feature_detection.hpp>

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////Abstract_classe_func_def////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void FeatureFrontEnd::initMatcher() {
  // Reference: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
  if (matcher_type_ == MatcherType::BF) {
    int norm_type;
    if (descriptor_type_ == DescriptorType::AKAZE ||
        descriptor_type_ == DescriptorType::BRISK ||
        descriptor_type_ == DescriptorType::ORB) {
      norm_type = cv::NORM_HAMMING;
    } else if (descriptor_type_ == DescriptorType::SIFT ||
               descriptor_type_ == DescriptorType::SuperPoint) {
      norm_type = cv::NORM_L2;
    } else {
      ROS_ERROR("[initMatcher] Decscriptor is not implemented");
    }
    matcher_ = cv::BFMatcher::create(norm_type, cross_check_);
  } else if (matcher_type_ == MatcherType::FLANN) {
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  }
}

void FeatureFrontEnd::solveStereoOdometry(tf2::Transform &cam0_curr_T_cam0_prev,
                                          const float stereo_threshold) {
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
  cv::triangulatePoints(projection_matrix_l_, projection_matrix_r_,
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
  const cv::Mat intrinsics_l = projection_matrix_l_.colRange(0, 3);
  cv::Mat inliers;
  const cv::Mat distortion = cv::Mat::zeros(4, 1, CV_32FC1);
  cv::Mat r_vec = r_vec_pred.clone();
  cv::Mat t = t_pred.clone();
  bool pnp_result = cv::solvePnPRansac(
      curr_left_point3d, keypoints_prev_left, intrinsics_l, distortion, r_vec,
      t, false, 500, 2.0, 0.999, inliers, cv::SOLVEPNP_EPNP);

  if (!pnp_result) {
    ROS_ERROR(
        "solvePnPRansac failed! Identity transformation will be applied.");
    ROS_INFO("keypoints_curr_left size = %lu", keypoints_curr_left.size());
    ROS_INFO("inliers rows = %d, cols = %d", inliers.rows, inliers.cols);
    r_vec = r_vec_pred.clone();
    t = t_pred.clone();
  } else {
    r_vec_pred = r_vec.clone();
    t_pred = t.clone();
  }

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

  // tf2::Quaternion q_tf2 = cam0_prev_T_cam0_curr.getRotation();
  // ROS_INFO("From PnP, cam0_prev_T_cam0_curr.rotation():\n axis = %.4f, %.4f,
  // "
  //          "%.4f, and angle = %.4f",
  //          q_tf2.getAxis().getX(), q_tf2.getAxis().getY(),
  //          q_tf2.getAxis().getZ(), q_tf2.getAngle());
  // ROS_INFO(
  //     "From PnP, cam0_prev_T_cam0_curr.translation():\n t = %.4f, %.4f,
  //     %.4f", t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));

  cam0_curr_T_cam0_prev = cam0_prev_T_cam0_curr.inverse();
}

// TODO: Add mask visualization (cautious for empty mask)
cv::Mat FeatureFrontEnd::visualizeMatches(const MatchType match_type) {
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