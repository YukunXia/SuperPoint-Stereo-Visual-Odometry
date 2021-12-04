#include <actionlib/server/simple_action_server.h>
#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Odometry.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <experimental/filesystem>
#include <fstream>
#include <sstream>
#include <string>
namespace fs = std::experimental::filesystem;

#include <odml_data_processing/kitti_data_loaderAction.h>
#include <odml_data_processing/kitti_data_loaderFeedback.h>
#include <odml_data_processing/kitti_data_loaderGoal.h>
#include <odml_data_processing/kitti_data_loaderResult.h>

// parameters for rosbag playing
int pre_waiting_time;
double rosbag_rate;
int start_frame, end_frame;

typedef actionlib::SimpleActionServer<
    odml_data_processing::kitti_data_loaderAction>
    KittiDataServer;
odml_data_processing::kitti_data_loaderFeedback kitti_data_loader_feedback;
odml_data_processing::kitti_data_loaderResult kitti_data_loader_result;

std::string seq, cmd;
std::ostringstream ss;
int sys_ret;

constexpr int NUM_KITTI_BAGS = 14;
std::array<std::string, NUM_KITTI_BAGS> kitti_eval_id_to_file_name = {
    "kitti_2011_10_03_drive_0027_synced.bag",  // 0
    "kitti_2011_10_03_drive_0042_synced.bag",  // 1
    "kitti_2011_10_03_drive_0034_synced.bag",  // 2
    "kitti_2011_09_26_drive_0067_synced.bag",  // 3
    "kitti_2011_09_30_drive_0016_synced.bag",  // 4
    "kitti_2011_09_30_drive_0018_synced.bag",  // 5
    "kitti_2011_09_30_drive_0020_synced.bag",  // 6
    "kitti_2011_09_30_drive_0027_synced.bag",  // 7
    "kitti_2011_09_30_drive_0028_synced.bag",  // 8
    "kitti_2011_09_30_drive_0033_synced.bag",  // 9
    "kitti_2011_09_30_drive_0034_synced.bag",  // 10
    "kitti_2011_09_26_drive_0001_synced.bag",  // 11, extras
    "kitti_2011_09_26_drive_0002_synced.bag",  // 12, extras
    "kitti_2011_09_26_drive_0005_synced.bag"}; // 13, extras

std::array<int, NUM_KITTI_BAGS> kitti_eval_id_to_start_frame = {
    0, 0, 0, 0, 0, 0, 0, 0, 1100, 0, 0, 0, 0, 0};
std::array<int, NUM_KITTI_BAGS> kitti_eval_id_to_end_frame = {
    4540, 1100, 4660, 800,  270,     2760,    1100,
    1100, 5170, 1590, 1200, INT_MAX, INT_MAX, INT_MAX};

std::ofstream visual_odom_result_file;
std::string visual_odom_result_file_name;
int seq_count = 0;
int seq_start = 0;
int seq_end = INT_MAX;
bool world_eigenT_base_start_inited = false;
Eigen::Isometry3d world_eigenT_base_start;
bool base_eigenT_cam0_inited = false;
Eigen::Isometry3d base_eigenT_cam0;

void execute(
    const odml_data_processing::kitti_data_loaderGoalConstPtr &kiti_data_goal,
    KittiDataServer *kitti_data_server) {
  ROS_INFO("[data_processing_node]\nnew goal received, kitti_eval_id = %d, "
           "description = %s\n",
           kiti_data_goal->kitti_eval_id, kiti_data_goal->description.c_str());

  kitti_data_loader_result.loading_finished = false;

  if (kiti_data_goal->kitti_eval_id < 0 ||
      kiti_data_goal->kitti_eval_id >= NUM_KITTI_BAGS)
    return;

  // update seq info for saving
  seq_start = kitti_eval_id_to_start_frame[kiti_data_goal->kitti_eval_id];
  seq_count = 0;
  seq_end = kitti_eval_id_to_end_frame[kiti_data_goal->kitti_eval_id];
  world_eigenT_base_start_inited = false;

  // handle directory
  std::string visual_odom_result_path_name =
      ros::package::getPath("odml_data_processing") + "/kitti_results" +
      (kiti_data_goal->description.empty() ? "/default"
                                           : "/" + kiti_data_goal->description);
  fs::path dir(visual_odom_result_path_name);
  // ROS_INFO("odom result path = %s", visual_odom_result_path_name.c_str());
  if (!(fs::exists(dir))) {
    if (fs::create_directory(dir)) {
      ROS_INFO("%s is created", visual_odom_result_path_name.c_str());
    }
  }

  // initialize result filename
  visual_odom_result_file_name =
      std::to_string(kiti_data_goal->kitti_eval_id) + "_pred.txt";
  if (visual_odom_result_file_name.size() == 10)
    visual_odom_result_file_name = "0" + visual_odom_result_file_name;

  visual_odom_result_file.open(visual_odom_result_path_name + "/" +
                               visual_odom_result_file_name);
  if (!visual_odom_result_file.is_open()) {
    ROS_ERROR("[data_processing_node]\nvisual_odom_result_file `%s` is not "
              "opened\n",
              visual_odom_result_file_name.c_str());
    return;
  } else {
    ROS_INFO("[data_processing_node]\nvisual_odom_result_file `%s` is opened\n",
             visual_odom_result_file_name.c_str());
  }

  // reset transformation states
  world_eigenT_base_start_inited = false;
  base_eigenT_cam0_inited = false;

  // play the bag
  cmd = "rosbag play " + ros::package::getPath("odml_data_processing") +
        "/bags/" + kitti_eval_id_to_file_name[kiti_data_goal->kitti_eval_id] +
        " -d " + std::to_string(pre_waiting_time) + " -r " +
        std::to_string(rosbag_rate) + " --quiet";

  ROS_INFO("The command is %s\n", cmd.c_str());
  sys_ret = system(cmd.c_str());
  // wait for visual odometry to finish
  sleep(2);

  visual_odom_result_file.close();

  // http://wiki.ros.org/actionlib_tutorials/Tutorials/SimpleActionServer%28ExecuteCallbackMethod%29
  kitti_data_loader_result.loading_finished = true;
  kitti_data_server->setSucceeded(kitti_data_loader_result);
}

void visualOdomCallback(const nav_msgs::Odometry::ConstPtr visual_odom_msg) {
  if (seq_count < seq_start) {
    ++seq_count;
    return;
  }

  // from here: seq_count >= seq_start
  const geometry_msgs::Pose &world_geoT_base_curr = visual_odom_msg->pose.pose;

  Eigen::Isometry3d world_eigenT_base_curr;
  tf::poseMsgToEigen(world_geoT_base_curr, world_eigenT_base_curr);

  if (!world_eigenT_base_start_inited) {
    world_eigenT_base_start_inited = true;
    world_eigenT_base_start = world_eigenT_base_curr;
  }

  Eigen::Isometry3d base_start_eigenT_base_curr =
      world_eigenT_base_start.inverse() * world_eigenT_base_curr;

  if (!base_eigenT_cam0_inited) {
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener(tf_buffer);
    // cam0 means gray left camera
    geometry_msgs::TransformStamped base_stamped_tf_cam0;
    base_stamped_tf_cam0 = tf_buffer.lookupTransform(
        "base_link", "camera_gray_left", ros::Time(0), ros::Duration(3.0));
    tf::transformMsgToEigen(base_stamped_tf_cam0.transform, base_eigenT_cam0);
    base_eigenT_cam0_inited = true;
  }

  const Eigen::Isometry3d cam0_start_eigenT_cam0_curr =
      base_eigenT_cam0.inverse() * base_start_eigenT_base_curr *
      base_eigenT_cam0;
  const Eigen::Matrix4d cam0_start_eigenTmat_cam0_curr =
      cam0_start_eigenT_cam0_curr.matrix();

  // pose: world_T_base_curr
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      visual_odom_result_file << cam0_start_eigenTmat_cam0_curr(r, c) << " ";
    }
  }
  visual_odom_result_file << "\n";
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "data_processing_node");

  ros::NodeHandle nh_private = ros::NodeHandle("~");
  nh_private.getParam("pre_waiting_time", pre_waiting_time);
  pre_waiting_time = std::max(2, pre_waiting_time);
  nh_private.getParam("rosbag_rate", rosbag_rate);

  ros::NodeHandle nh;

  ros::Subscriber visual_odom_msg_sub = nh.subscribe(
      "/odml_visual_odometry/visual_odom", 100, visualOdomCallback);

  KittiDataServer kitti_data_server(
      nh, "kitti_loader_action_server",
      boost::bind(&execute, _1, &kitti_data_server), false);
  kitti_data_server.start();

  ROS_INFO("[data_processing_node]\n ready\n");

  ros::Rate loop_rate(100);
  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}