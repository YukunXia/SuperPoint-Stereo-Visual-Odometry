#include <actionlib/server/simple_action_server.h>
#include <ros/package.h>
#include <ros/ros.h>

#include <sstream>
#include <string>

#include <odml_data_processing/kitti_data_loaderAction.h>
#include <odml_data_processing/kitti_data_loaderFeedback.h>
#include <odml_data_processing/kitti_data_loaderGoal.h>
#include <odml_data_processing/kitti_data_loaderResult.h>

// parameters for rosbag playing
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

void execute(
    const odml_data_processing::kitti_data_loaderGoalConstPtr &kiti_data_goal,
    KittiDataServer *kitti_data_server) {
  kitti_data_loader_result.loading_finished = false;

  // // section1, generate new message filter and tf_buffer
  // message_filters::Synchronizer<MySyncPolicy> sync(
  //     MySyncPolicy(20), *sub_image00_ptr, *sub_camera00_ptr,
  //     *sub_velodyne_ptr);
  // sync.setInterMessageLowerBound(ros::Duration(0.09));
  // sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  // // section 2, initialize all pointers
  // init(goal);

  // section 3, play the bag
  start_frame = kiti_data_goal->start_frame;
  end_frame = kiti_data_goal->end_frame;

  ss.clear();
  ss.str("");
  ss << std::setw(4) << std::setfill('0') << kiti_data_goal->seq;
  seq = std::string(ss.str());
  cmd = "rosbag play " + ros::package::getPath("odml_data_processing") +
        "/bags/kitti_" + kiti_data_goal->date + "_drive_" + seq +
        "_synced.bag -d 2 -r " + std::to_string(rosbag_rate);

  ROS_INFO("The command is %s", cmd.c_str());
  sys_ret = system(cmd.c_str());

  kitti_data_loader_result.loading_finished = true;
  kitti_data_server->setSucceeded(kitti_data_loader_result);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "data_processing_node");

  ros::NodeHandle nh_private = ros::NodeHandle("~");
  nh_private.getParam("rosbag_rate", rosbag_rate);

  ros::NodeHandle nh;

  KittiDataServer kitti_data_server(
      nh, "kitti_loader_action_server",
      boost::bind(&execute, _1, &kitti_data_server), false);
  kitti_data_server.start();

  ros::Rate loop_rate(100);
  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}