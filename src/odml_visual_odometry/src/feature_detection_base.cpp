#include <eigen3/Eigen/Geometry>

#include <odml_visual_odometry/ceres_cost_function.hpp>
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
    // https://stackoverflow.com/a/35611756/9556578
    // Quote:  The error is caused by configuring BFMatcher with crossCheck =
    // True. For k > 1, set crossCheck = False (constructor default).
    matcher_ = cv::BFMatcher::create(
        norm_type, cross_check_ & (selector_type_ != SelectorType::KNN));
  } else if (matcher_type_ == MatcherType::FLANN) {
    matcher_ = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(2));
  }
}

void FeatureFrontEnd::clearLagecyData() {
  images_dq.clear();
  keypoints_dq.clear();
  descriptors_dq.clear();
  for (auto &cv_DMatches : cv_DMatches_list) {
    cv_DMatches.clear();
  }
  r_vec_pred = cv::Mat::zeros(3, 1, CV_64FC1);
  t_vec_pred = cv::Mat::zeros(3, 1, CV_64FC1);
  frame_count = 0;
  for (auto &map_of_indices : maps_of_indices) {
    map_of_indices.clear();
  }
  prev_left_points_3d_inited = false;

  inliers_postmatching.clear();
  inliers_pnp.clear();
}

// entering into this function means there's at least 2 time frames
// TODO: add assertion to check
void FeatureFrontEnd::solveStereoOdometry(
    tf2::Transform &cam0_curr_T_cam0_prev) {
  const std::vector<cv::DMatch> &cv_Dmatches_curr_stereo =
      cv_DMatches_list[CURR_LEFT_CURR_RIGHT];
  const std::vector<cv::DMatch> &cv_Dmatches_inter_left =
      cv_DMatches_list[CURR_LEFT_PREV_LEFT];

  // extract common feature points in both matches
  const size_t num_curr_stereo = cv_Dmatches_curr_stereo.size();
  std::vector<cv::Point2f> keypoints_curr_left;
  keypoints_curr_left.reserve(num_curr_stereo);
  std::vector<cv::Point2f> keypoints_curr_right;
  keypoints_curr_right.reserve(num_curr_stereo);
  std::vector<cv::Point2f> keypoints_prev_left;
  keypoints_prev_left.reserve(num_curr_stereo);
  std::vector<cv::Point2f> keypoints_prev_right;
  keypoints_prev_right.reserve(num_curr_stereo);
  inliers_postmatching.clear();
  inliers_postmatching.reserve(num_curr_stereo);

  if (refinement_degree_ >= 3) {
    // `map_from_matched_to_valid_index` will be used in the next time frame
    // -1 means invalid
    map_from_curr_left_matched_to_curr_valid_index.clear();
    map_from_curr_left_matched_to_curr_valid_index.resize(
        keypoints_dq.end()[CURR_LEFT].size(), -1);
    map_from_curr_valid_to_prev_left_matched_index.clear();
    map_from_curr_valid_to_prev_left_matched_index.reserve(
        cv_Dmatches_curr_stereo.size());
  }

  for (const auto &cv_Dmatch : cv_Dmatches_curr_stereo) {
    const int matched_index_curr_left = cv_Dmatch.queryIdx;

    // fulfilled curr stereo matching, but failed lhs interframe matching
    if (maps_of_indices[CURR_LEFT_PREV_LEFT].at(matched_index_curr_left) == -1)
      continue;

    const cv::Point2f &keypoint_curr_left =
        keypoints_dq.end()[CURR_LEFT][matched_index_curr_left].pt;
    const cv::Point2f &keypoint_curr_right =
        keypoints_dq.end()[CURR_RIGHT][cv_Dmatch.trainIdx].pt;

    // if not passing stereo checks (y distance & min disparity) => discard
    if (std::abs(keypoint_curr_left.y - keypoint_curr_right.y) >
            stereo_threshold_ ||
        std::abs(keypoint_curr_left.x - keypoint_curr_right.x) < min_disparity_)
      continue;

    const int matched_index_prev_left =
        maps_of_indices[CURR_LEFT_PREV_LEFT].at(matched_index_curr_left);
    const cv::Point2f &keypoint_prev_left =
        keypoints_dq.end()[PREV_LEFT][matched_index_prev_left].pt;

    // fulfilled curr stereo matching & lhs interframe matching, but failed prev
    // stereo matching
    if (maps_of_indices[PREV_LEFT_PREV_RIGHT].at(matched_index_prev_left) == -1)
      continue;

    // from here, no filtering will be applied
    keypoints_curr_left.push_back(keypoint_curr_left);
    keypoints_curr_right.push_back(keypoint_curr_right);
    // for visualization only. TODO: consider adding a flag here
    inliers_postmatching.push_back(matched_index_curr_left);

    keypoints_prev_left.push_back(keypoint_prev_left);

    const int index_prev_right =
        maps_of_indices[PREV_LEFT_PREV_RIGHT].at(matched_index_prev_left);
    const cv::Point2f keypoint_prev_right =
        keypoints_dq.end()[PREV_RIGHT][index_prev_right].pt;
    keypoints_prev_right.push_back(keypoint_prev_right);

    if (refinement_degree_ >= 3) {
      assert(keypoints_curr_left.size() > 0);
      map_from_curr_left_matched_to_curr_valid_index.at(
          matched_index_curr_left) = keypoints_curr_left.size() - 1;
      map_from_curr_left_matched_to_curr_valid_index.push_back(
          matched_index_curr_left);
      map_from_curr_valid_to_prev_left_matched_index.push_back(
          matched_index_prev_left);
    }
  }

  // triangulation
  // TODO: handle exception when num of keypoints_curr_left is no enough
  cv::Mat curr_left_points_4d;
  cv::triangulatePoints(projection_matrix_l_, projection_matrix_r_,
                        keypoints_curr_left, keypoints_curr_right,
                        curr_left_points_4d);
  // curr_left_points_4d: 64FC1 4xN => 64FC1 Nx4
  curr_left_points_4d = curr_left_points_4d.t();
  // curr_left_points_4d: 64FC1 Nx4 => 64FC4 Nx1
  curr_left_points_4d =
      curr_left_points_4d.reshape(4, curr_left_points_4d.rows);

  cv::Mat curr_left_points_3d;
  // curr_left_points_3d: 32FC3 Nx1
  cv::convertPointsFromHomogeneous(curr_left_points_4d, curr_left_points_3d);

  // PnP
  // TODO: handle exception when num of points is no enough
  const cv::Matx33d intrinsics_l = projection_matrix_l_.colRange(0, 3).clone();

  // // 32SC1 num_pointsx1 -> each at<int>(row,0) is an int index
  const cv::Mat distortion = cv::Mat::zeros(4, 1, CV_64FC1);
  // cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat r_vec = r_vec_pred.clone();
  // cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat t_vec = t_vec_pred.clone();
  inliers_pnp.clear();
  inliers_pnp.reserve(curr_left_points_3d.rows);
  bool pnp_result = cv::solvePnPRansac(
      curr_left_points_3d, keypoints_prev_left, intrinsics_l, distortion, r_vec,
      t_vec, true, 500, 2.0, 0.999, inliers_pnp, cv::USAC_ACCURATE);

  const double acceleration =
      cv::norm(t_vec, t_vec_pred, cv::NORM_L2) / TIME_INTERVAL;
  bool do_optmz = false;
  if (!pnp_result) {
    ROS_ERROR(
        "solvePnPRansac failed! Identity transformation will be applied.");
    r_vec = r_vec_pred.clone();
    t_vec = t_vec_pred.clone();
    ROS_INFO("keypoints_curr_left size = %lu", keypoints_curr_left.size());
    ROS_INFO("inlier size = %lu", inliers_pnp.size());
  } else if (frame_count > IGNORE_FRAME_COUNT &&
             acceleration > MAX_ACCELERATION) {
    ROS_ERROR(
        "solvePnPRansac succeeded but acceleration (%.2f) is abnormally large! "
        "pred t_vec is (%.4f, %.4f, %.4f), but t_vec is (%.4f, %.4f, %.4f) ",
        acceleration, t_vec_pred.at<double>(0, 0), t_vec_pred.at<double>(1, 0),
        t_vec_pred.at<double>(2, 0), t_vec.at<double>(0, 0),
        t_vec.at<double>(1, 0), t_vec.at<double>(2, 0));
    r_vec = r_vec_pred.clone();
    t_vec = t_vec_pred.clone();
  } else {
    // ROS_INFO("solvePnPRansac succeeded! pred t_vec is (%.4f, %.4f, %.4f),
    // now"
    //          "t_vec is (%.4f, %.4f, %.4f)",
    //          t_vec_pred.at<double>(0, 0), t_vec_pred.at<double>(1, 0),
    //          t_vec_pred.at<double>(2, 0), t_vec.at<double>(0, 0),
    //          t_vec.at<double>(1, 0), t_vec.at<double>(2, 0));
    // TODO: update this later after refinement
    r_vec_pred = r_vec.clone();
    t_vec_pred = t_vec.clone();
    do_optmz = true;
  }

  const Eigen::Vector3d axis_angle(
      r_vec.at<double>(0, 0), r_vec.at<double>(1, 0), r_vec.at<double>(2, 0));
  const double angle = axis_angle.norm();
  const Eigen::Vector3d axis = axis_angle.normalized();
  Eigen::Quaterniond q_to_be_optmz(Eigen::AngleAxisd(angle, axis));
  Eigen::Vector3d t_to_be_optmz(t_vec.at<double>(0, 0), t_vec.at<double>(1, 0),
                                t_vec.at<double>(2, 0));

  if (do_optmz && refinement_degree_ > 0) {
    // use inliers to refine PnP
    // TODO: handle exception when num of points is no enough
    // unit: pixel
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    // project 3d point at curr frame to a 2d point at prev left & right frame
    for (const int curr_valid_index : inliers_pnp) {
      // `curr_valid_index` works for keypoints_curr_left, keypoints_curr_right,
      // keypoints_prev_left and keypoint_prev_right in context of the
      // triangulation of the current stereo images

      // project 3d point at curr frame to a 2d point at prev left frame
      ceres::CostFunction *cost_function_l = CostFunctor32::Create(
          curr_left_points_3d.at<cv::Vec3f>(curr_valid_index, 0),
          keypoints_prev_left.at(curr_valid_index), projection_matrix_l_,
          false);
      problem.AddResidualBlock(cost_function_l, loss_function,
                               q_to_be_optmz.coeffs().data(),
                               t_to_be_optmz.data());

      if (refinement_degree_ <= 1)
        continue;

      // project 3d point at curr frame to a 2d point at prev right frame
      ceres::CostFunction *cost_function_r = CostFunctor32::Create(
          curr_left_points_3d.at<cv::Vec3f>(curr_valid_index, 0),
          keypoints_prev_right.at(curr_valid_index), projection_matrix_r_,
          false);
      problem.AddResidualBlock(cost_function_r, loss_function,
                               q_to_be_optmz.coeffs().data(),
                               t_to_be_optmz.data());

      if (refinement_degree_ <= 2)
        continue;

      // Not necessarily optimal, but the current setting is that only after
      // this function is run once, will the following optimization factors been
      // considered
      if (!prev_left_points_3d_inited)
        continue;

      const int matched_index_prev_left =
          map_from_curr_valid_to_prev_left_matched_index.at(curr_valid_index);
      const int valid_index_prev_left =
          map_from_prev_left_matched_to_prev_valid_index.at(
              matched_index_prev_left);
      if (valid_index_prev_left == -1)
        continue;

      // project 3d point at curr frame to a 2d point at prev left frame
      ceres::CostFunction *cost_function_l_inverse = CostFunctor32::Create(
          prev_left_points_3d.at<cv::Vec3f>(valid_index_prev_left, 0),
          keypoints_curr_left.at(curr_valid_index), projection_matrix_l_, true);
      problem.AddResidualBlock(cost_function_l_inverse, loss_function,
                               q_to_be_optmz.coeffs().data(),
                               t_to_be_optmz.data());

      assert(keypoints_curr_right.at(curr_valid_index).x >= 0.0f &&
             keypoints_curr_right.at(curr_valid_index).y >= 0.0f);

      if (refinement_degree_ <= 3)
        continue;

      // project 3d point at curr frame to a 2d point at prev right frame
      ceres::CostFunction *cost_function_r_inverse = CostFunctor32::Create(
          prev_left_points_3d.at<cv::Vec3f>(valid_index_prev_left, 0),
          keypoints_curr_right.at(curr_valid_index), projection_matrix_r_,
          true);
      problem.AddResidualBlock(cost_function_r_inverse, loss_function,
                               q_to_be_optmz.coeffs().data(),
                               t_to_be_optmz.data());
    }

    problem.SetParameterization(q_to_be_optmz.coeffs().data(),
                                new ceres::EigenQuaternionParameterization);

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = 40;
    solver_options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable() ||
        summary.termination_type != ceres::TerminationType::CONVERGENCE) {
      ROS_ERROR("summary.IsSolutionUsable() == false or NOT CONVERGENT");
      ROS_ERROR("Ceres solver report: \n%s", summary.FullReport().c_str());
      q_to_be_optmz = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
      t_to_be_optmz =
          Eigen::Vector3d(t_vec.at<double>(0, 0), t_vec.at<double>(1, 0),
                          t_vec.at<double>(2, 0));
    }
  }

  // output
  tf2::Transform cam0_prev_T_cam0_curr;
  cam0_prev_T_cam0_curr.setRotation(
      tf2::Quaternion(q_to_be_optmz.x(), q_to_be_optmz.y(), q_to_be_optmz.z(),
                      q_to_be_optmz.w()));
  cam0_prev_T_cam0_curr.setOrigin(
      tf2::Vector3(t_to_be_optmz(0), t_to_be_optmz(1), t_to_be_optmz(2)));

  cam0_curr_T_cam0_prev = cam0_prev_T_cam0_curr.inverse();

  // update data for the next time frame
  if (refinement_degree_ >= 3) {
    map_from_prev_left_matched_to_prev_valid_index =
        map_from_curr_left_matched_to_curr_valid_index;
    curr_left_points_3d.copyTo(prev_left_points_3d);
    // prev_left_points_3d = curr_left_points_3d.clone();
    prev_left_points_3d_inited = true;
  }

  ++frame_count;
  // TODO: handle int overflow [not urgent]
  // frame_count = std::min(frame_count, 100);
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

void FeatureFrontEnd::matchDescriptors(const MatchType match_type) {
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
  cv_Dmatches.clear();
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

  if (match_type == MatchType::CURR_LEFT_CURR_RIGHT) {
    // before updating the CURR_LEFT_CURR_RIGHT, assign PREV_LEFT_PREV_RIGHT to
    // be the unupdated CURR_LEFT_CURR_RIGHT, which is the previous time frame's
    // CURR_LEFT_CURR_RIGHT
    maps_of_indices[MatchType::PREV_LEFT_PREV_RIGHT] =
        maps_of_indices[MatchType::CURR_LEFT_CURR_RIGHT];
  }

  std::vector<int> &maps_of_indices_curr_matching =
      maps_of_indices.at(match_type);
  maps_of_indices_curr_matching.clear();
  maps_of_indices_curr_matching.resize(keypoints0.size(), -1);
  for (cv::DMatch &cv_Dmatch : cv_Dmatches) {
    const int matched_index_0 = cv_Dmatch.queryIdx;
    const int matched_index_1 = cv_Dmatch.trainIdx;
    maps_of_indices_curr_matching.at(matched_index_0) = matched_index_1;
  }

  ROS_INFO("%lu matches for %s", cv_Dmatches.size(),
           MatchType_str[match_type].c_str());
}

cv::Mat FeatureFrontEnd::visualizeInliers(const ImagePosition image_position) {
  if (image_position != CURR_LEFT)
    ROS_ERROR("inlier visualization for %s is not implemented yet",
              ImagePosition_str.at(image_position).c_str());

  cv::Mat inlier_visualization_img = images_dq.end()[image_position];
  cv::cvtColor(inlier_visualization_img, inlier_visualization_img,
               cv::COLOR_GRAY2BGR);

  // matched indices
  const std::unordered_set<int> inliers_postmatching_set(
      inliers_postmatching.begin(), inliers_postmatching.end());
  // matched indices
  std::unordered_set<int> inliers_pnp_set;
  for (const int valid_inlier : inliers_pnp) {
    inliers_pnp_set.insert(inliers_postmatching.at(valid_inlier));
  }

  const cv::Scalar color_pnp_set = cv::Scalar(0, 255, 0);
  const cv::Scalar color_postmatching = cv::Scalar(255, 0, 255);
  const cv::Scalar color_others = cv::Scalar(0, 0, 255);
  for (const auto &cv_Dmatch : cv_DMatches_list[CURR_LEFT_CURR_RIGHT]) {
    const int matched_index_curr_left = cv_Dmatch.queryIdx;
    const int matched_index_prev_left =
        maps_of_indices[CURR_LEFT_PREV_LEFT].at(matched_index_curr_left);

    cv::Scalar color;
    int line_width;
    if (inliers_pnp_set.find(matched_index_curr_left) !=
        inliers_pnp_set.end()) {
      color = color_pnp_set;
      line_width = 2;
    } else if (inliers_postmatching_set.find(matched_index_curr_left) !=
               inliers_postmatching_set.end()) {
      color = color_postmatching;
      line_width = 1;
    } else {
      color = color_others;
      line_width = 1;
    }

    if (matched_index_prev_left != -1)
      cv::line(inlier_visualization_img,
               keypoints_dq.end()[CURR_LEFT][matched_index_curr_left].pt,
               keypoints_dq.end()[PREV_LEFT][matched_index_prev_left].pt, color,
               line_width);
    cv::circle(inlier_visualization_img,
               keypoints_dq.end()[CURR_LEFT][matched_index_curr_left].pt, 3,
               color, -1);
  }

  return inlier_visualization_img;
}