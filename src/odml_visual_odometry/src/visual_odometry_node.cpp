#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <odml_data_processing/kitti_data_loaderActionGoal.h>
#include <odml_visual_odometry/feature_detection.hpp>

#include <opencv4/opencv2/opencv.hpp>

nav_msgs::Odometry visual_odom_msg;
ros::Publisher visual_odom_msg_pub;
nav_msgs::Path visual_odom_path;
ros::Publisher visual_odom_path_pub;

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image,
    sensor_msgs::CameraInfo>
    StereoSyncPolicy;
// typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
//                                                         sensor_msgs::CameraInfo>
//     MonoSyncPolicy;

std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>
    sub_image00_ptr = nullptr;
std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>>
    sub_camera00_ptr = nullptr;
std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>
    sub_image01_ptr = nullptr;
std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>>
    sub_camera01_ptr = nullptr;

std::shared_ptr<message_filters::Synchronizer<StereoSyncPolicy>>
    stereo_sync_ptr = nullptr;

std::shared_ptr<tf2_ros::Buffer> tf_buffer_ptr;
std::shared_ptr<tf2_ros::TransformListener> tf_listener;
// cam0 means gray left camera
bool base_stamped_tf_cam0_inited = false;
geometry_msgs::TransformStamped base_stamped_tf_cam0;
tf2::Transform base_T_cam0;

cv_bridge::CvImagePtr cv_ptr_l;
cv_bridge::CvImagePtr cv_ptr_r;
bool first_frame = true;
tf2::Transform world_T_base_curr = tf2::Transform::getIdentity();

std::shared_ptr<FeatureFrontEnd> feature_front_end_ptr;
std::array<image_transport::Publisher, MATCH_TYPE_NUM> pub_matches_img_list;
image_transport::Publisher pub_inliers;

tf2::Transform cam0_curr_T_cam0_prev_last_valid = tf2::Transform::getIdentity();

cv::Mat
cameraInfoToKMatrix(const sensor_msgs::CameraInfo::ConstPtr &camera_info_msg) {
  // Ref http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
  // # Projection/camera matrix # 3x4 row-major matrix
  // #     [fx'  0  cx' Tx]
  // # P = [ 0  fy' cy' Ty]
  // #     [ 0   0   1   0]
  const cv::Mat intrinsics =
      (cv::Mat1d(3, 3) << camera_info_msg->P[0], camera_info_msg->P[1],
       camera_info_msg->P[2], camera_info_msg->P[4], camera_info_msg->P[5],
       camera_info_msg->P[6], camera_info_msg->P[8], camera_info_msg->P[9],
       camera_info_msg->P[10]);
  return intrinsics;
}

cv::Mat
cameraInfoToPMatrix(const sensor_msgs::CameraInfo::ConstPtr &camera_info_msg) {
  // Ref http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
  // # Projection/camera matrix # 3x4 row-major matrix
  // #     [fx'  0  cx' Tx]
  // # P = [ 0  fy' cy' Ty]
  // #     [ 0   0   1   0]
  const cv::Mat projection_matrix =
      (cv::Mat1d(3, 4) << camera_info_msg->P[0], camera_info_msg->P[1],
       camera_info_msg->P[2], camera_info_msg->P[3], camera_info_msg->P[4],
       camera_info_msg->P[5], camera_info_msg->P[6], camera_info_msg->P[7],
       camera_info_msg->P[8], camera_info_msg->P[9], camera_info_msg->P[10],
       camera_info_msg->P[11]);
  return projection_matrix;
}

void publishOdometry(tf2::Transform cam0_curr_T_cam0_prev) {

  if (!base_stamped_tf_cam0_inited) {
    tf_buffer_ptr = std::make_shared<tf2_ros::Buffer>();
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_ptr);
    base_stamped_tf_cam0 = tf_buffer_ptr->lookupTransform(
        "base_link", "camera_gray_left", ros::Time(0), ros::Duration(3.0));
    tf2::fromMsg(base_stamped_tf_cam0.transform, base_T_cam0);
    base_stamped_tf_cam0.header.frame_id = "base";
    base_stamped_tf_cam0.child_frame_id = "cam0";
    tf2_ros::StaticTransformBroadcaster static_broadcaster;
    static_broadcaster.sendTransform(base_stamped_tf_cam0);

    base_stamped_tf_cam0_inited = true;
  }

  // screen out abnormal results
  // TODO: move magic numbers to the config file
  if (cam0_curr_T_cam0_prev.getOrigin().length() > 10) {
    // vel > 100 m/s
    cam0_curr_T_cam0_prev = cam0_curr_T_cam0_prev_last_valid;
  } else {
    cam0_curr_T_cam0_prev_last_valid = cam0_curr_T_cam0_prev;
  }

  const tf2::Transform base_prev_T_base_curr =
      base_T_cam0 * cam0_curr_T_cam0_prev.inverse() * base_T_cam0.inverse();
  world_T_base_curr *= base_prev_T_base_curr;

  visual_odom_msg.header.frame_id = "world";
  visual_odom_msg.child_frame_id = "visual_odom";
  visual_odom_msg.header.stamp = ros::Time::now(); // image_msg->header.stamp;
  visual_odom_msg.pose.pose.orientation.x = world_T_base_curr.getRotation().x();
  visual_odom_msg.pose.pose.orientation.y = world_T_base_curr.getRotation().y();
  visual_odom_msg.pose.pose.orientation.z = world_T_base_curr.getRotation().z();
  visual_odom_msg.pose.pose.orientation.w = world_T_base_curr.getRotation().w();
  visual_odom_msg.pose.pose.position.x = world_T_base_curr.getOrigin().x();
  visual_odom_msg.pose.pose.position.y = world_T_base_curr.getOrigin().y();
  visual_odom_msg.pose.pose.position.z = world_T_base_curr.getOrigin().z();
  visual_odom_msg_pub.publish(visual_odom_msg);

  geometry_msgs::PoseStamped visual_pose;
  visual_pose.header = visual_odom_msg.header;
  visual_pose.pose = visual_odom_msg.pose.pose;
  visual_odom_path.header.stamp = visual_odom_msg.header.stamp;
  visual_odom_path.header.frame_id = "world";
  visual_odom_path.poses.push_back(visual_pose);
  visual_odom_path_pub.publish(visual_odom_path);
}

void stereoCallback(
    const sensor_msgs::Image::ConstPtr &left_image_msg,
    const sensor_msgs::CameraInfo::ConstPtr &left_camera_info_msg,
    const sensor_msgs::Image::ConstPtr &right_image_msg,
    const sensor_msgs::CameraInfo::ConstPtr &right_camera_info_msg) {
  if (stereo_sync_ptr == nullptr) {
    ROS_ERROR("stereo_sync_ptr == nullptr");
    return;
  }

  // MONO8 => CV_8UC1 cv::Mat
  cv_ptr_l =
      cv_bridge::toCvCopy(left_image_msg, sensor_msgs::image_encodings::MONO8);
  cv_ptr_r =
      cv_bridge::toCvCopy(right_image_msg, sensor_msgs::image_encodings::MONO8);
  const cv::Mat img_l = cv_ptr_l->image;
  const cv::Mat img_r = cv_ptr_r->image;

  // TODO: place assertion here
  const cv::Mat img_l_intrinsics = cameraInfoToKMatrix(left_camera_info_msg);
  const cv::Mat img_r_intrinsics = cameraInfoToKMatrix(right_camera_info_msg);
  const cv::Mat projection_matrix_l = cameraInfoToPMatrix(left_camera_info_msg);
  const cv::Mat projection_matrix_r =
      cameraInfoToPMatrix(right_camera_info_msg);

  feature_front_end_ptr->addStereoImagePair(img_l, img_r, projection_matrix_l,
                                            projection_matrix_r);

  tf2::Transform cam0_curr_T_cam0_prev = tf2::Transform::getIdentity();

  if (first_frame) {
    first_frame = false;
    feature_front_end_ptr->matchDescriptors(MatchType::CURR_LEFT_CURR_RIGHT);
    publishOdometry(cam0_curr_T_cam0_prev);
    return;
  }

  const auto start = std::chrono::system_clock::now();

  // no PREV_LEFT_PREV_RIGHT
  for (int i = 0; i < 2; ++i) {
    const MatchType match_type = static_cast<MatchType>(i);
    feature_front_end_ptr->matchDescriptors(match_type);
    // classic_feature_front_end_ptr->solve5PointsRANSAC(match_type,
    // img_l_intrinsics, cam0_curr_T_cam0_prev);

    // For visualization
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    const cv_bridge::CvImage matches_viz_cvbridge =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8,
                           feature_front_end_ptr->visualizeMatches(match_type));
    // TODO: try compressed image
    pub_matches_img_list[match_type].publish(matches_viz_cvbridge.toImageMsg());
  }

  const auto mid = std::chrono::system_clock::now();
  ROS_INFO(
      "matching of 1 image takes %.4f ms",
      (float)std::chrono::duration_cast<std::chrono::microseconds>(mid - start)
              .count() /
          1000.0f);

  feature_front_end_ptr->solveStereoOdometry(cam0_curr_T_cam0_prev);

  const auto end = std::chrono::system_clock::now();
  ROS_INFO(
      "solveStereoOdometry of 1 image takes %.4f ms",
      (float)std::chrono::duration_cast<std::chrono::microseconds>(end - mid)
              .count() /
          1000.0f);

  publishOdometry(cam0_curr_T_cam0_prev);

  // For inliers visualization
  std_msgs::Header header;
  header.stamp = ros::Time::now();
  const cv_bridge::CvImage inliers_viz_cvbridge =
      cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8,
                         feature_front_end_ptr->visualizeInliers(CURR_LEFT));
  // TODO: try compressed image
  pub_inliers.publish(inliers_viz_cvbridge.toImageMsg());
}

// restart the vo and clean legacy data
void dataLodaerGoalCallback(
    const odml_data_processing::kitti_data_loaderActionGoalConstPtr
        &kiti_data_action_goal) {
  stereo_sync_ptr =
      std::make_shared<message_filters::Synchronizer<StereoSyncPolicy>>(
          StereoSyncPolicy(20), *sub_image00_ptr, *sub_camera00_ptr,
          *sub_image01_ptr, *sub_camera01_ptr);
  stereo_sync_ptr->setInterMessageLowerBound(ros::Duration(0.09));
  stereo_sync_ptr->registerCallback(
      boost::bind(&stereoCallback, _1, _2, _3, _4));
  ROS_INFO("visual odometry is ready");

  feature_front_end_ptr->clearLagecyData();

  visual_odom_path.poses.clear();
  base_stamped_tf_cam0_inited = false;
  world_T_base_curr = tf2::Transform::getIdentity();
  cam0_curr_T_cam0_prev_last_valid = tf2::Transform::getIdentity();

  first_frame = true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "visual_odometry_node");

  ros::NodeHandle nh_private = ros::NodeHandle("~");
  bool is_classic;
  nh_private.getParam("is_classic", is_classic);

  if (is_classic) {
    std::string detector_type;
    std::string descriptor_type;
    std::string matcher_type;
    std::string selector_type;
    double stereo_threshold;
    double min_disparity;
    int refinement_degree;
    nh_private.getParam("detector_type", detector_type);
    nh_private.getParam("descriptor_type", descriptor_type);
    nh_private.getParam("matcher_type", matcher_type);
    nh_private.getParam("selector_type", selector_type);
    nh_private.getParam("stereo_threshold", stereo_threshold);
    nh_private.getParam("min_disparity", min_disparity);
    nh_private.getParam("refinement_degree", refinement_degree);
    feature_front_end_ptr = std::make_shared<ClassicFeatureFrontEnd>(
        detector_name_to_type.at(detector_type),
        descriptor_name_to_type.at(descriptor_type),
        matcher_name_to_type.at(matcher_type),
        selector_name_to_type.at(selector_type),
        true, // cross check. only used in KNN mode
        stereo_threshold, min_disparity, refinement_degree);
  } else {
    std::string detector_type;
    std::string descriptor_type;
    nh_private.getParam("detector_type", detector_type);
    nh_private.getParam("descriptor_type", descriptor_type);
    if (detector_type == "SuperPoint" && descriptor_type == "SuperPoint") {
      std::string matcher_type;
      std::string selector_type;
      std::string model_name_prefix;
      std::string machine_name;
      std::string trt_precision;
      int image_height;
      int image_width;
      double conf_thresh;
      int dist_thresh;
      int num_threads;
      int border_remove;
      double stereo_threshold;
      double min_disparity;
      int refinement_degree;
      nh_private.getParam("matcher_type", matcher_type);
      nh_private.getParam("selector_type", selector_type);
      nh_private.getParam("model_name_prefix", model_name_prefix);
      nh_private.getParam("machine_name", machine_name);
      nh_private.getParam("trt_precision", trt_precision);
      nh_private.getParam("image_height", image_height);
      nh_private.getParam("image_width", image_width);
      nh_private.getParam("conf_thresh", conf_thresh);
      nh_private.getParam("dist_thresh", dist_thresh);
      nh_private.getParam("num_threads", num_threads);
      nh_private.getParam("border_remove", border_remove);
      nh_private.getParam("stereo_threshold", stereo_threshold);
      nh_private.getParam("min_disparity", min_disparity);
      nh_private.getParam("refinement_degree", refinement_degree);
      if (image_height % 8 != 0 || image_width % 8 != 0) {
        ROS_ERROR("image_height(%d) or image_width(%d) is indivisble by 8",
                  image_height, image_width);
        return 1;
      }
      feature_front_end_ptr = std::make_shared<SuperPointFeatureFrontEnd>(
          matcher_name_to_type.at(matcher_type),
          selector_name_to_type.at(selector_type),
          true, // cross check. only used in KNN mode
          model_name_prefix, machine_name, trt_precision_string2enum.at(trt_precision),
          image_height, image_width, conf_thresh, dist_thresh, num_threads,
          border_remove, stereo_threshold, min_disparity, refinement_degree);
    } else {
      ROS_ERROR("Detector(%s) or descriptor(%s) not implemented",
                detector_type.c_str(), descriptor_type.c_str());
      return 1;
    }
  }

  ros::NodeHandle nh;

  // TODO: try compressed image
  sub_image00_ptr =
      std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(
          nh, "/kitti/camera_gray_left/image_raw", 10);
  sub_camera00_ptr =
      std::make_shared<message_filters::Subscriber<sensor_msgs::CameraInfo>>(
          nh, "/kitti/camera_gray_left/camera_info", 10);
  sub_image01_ptr =
      std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(
          nh, "/kitti/camera_gray_right/image_raw", 10);
  sub_camera01_ptr =
      std::make_shared<message_filters::Subscriber<sensor_msgs::CameraInfo>>(
          nh, "/kitti/camera_gray_right/camera_info", 10);

  visual_odom_msg_pub = nh.advertise<nav_msgs::Odometry>(
      "/odml_visual_odometry/visual_odom", 100);
  visual_odom_path_pub = nh.advertise<nav_msgs::Path>(
      "/odml_visual_odometry/visual_odom_path", 100);

  ros::Subscriber data_lodaer_goal_sub = nh.subscribe(
      "/kitti_loader_action_server/goal", 1000, dataLodaerGoalCallback);

  image_transport::ImageTransport it(nh);
  for (int i = 0; i < 2; ++i) {
    pub_matches_img_list[i] = it.advertise(
        "/odml_visual_odometry/matches_visualization" + MatchType_str[i], 10);
  }
  pub_inliers = it.advertise("/odml_visual_odometry/inliers_visualization", 10);

  ros::Rate loop_rate(100);
  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}