#include <ros/ros.h>

#include <odml_data_processing/kitti_data_loaderActionGoal.h>
#include <odml_data_processing/kitti_data_loaderActionResult.h>

#include <string>
#include <vector>

bool loading_finished = true;
ros::Publisher pub_goal;
int config_id = -1;

std::vector<std::string> detector_list;
std::vector<std::string> descriptor_list;
std::vector<int> width_list;
std::vector<int> height_list;
std::vector<std::string> description_list;

std::vector<std::string> detectors = {"SIFT"};
std::vector<std::string> descriptors = {"SIFT"};
std::vector<std::pair<int, int>> resolutions = {
    {120, 392}};
std::vector<int> seq_ids = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10};

void dataLodaerResultCallback(
    const odml_data_processing::kitti_data_loaderActionResultConstPtr
        &kiti_data_action_result) {
  assert(kiti_data_action_result->result.loading_finished == true);
  loading_finished = true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "data_processing_eval_node");

  ros::NodeHandle nh_private = ros::NodeHandle("~");
  std::string device;
  double rosbag_rate;
  bool verbose;
  nh_private.getParam("device", device);
  nh_private.getParam("rosbag_rate", rosbag_rate);
  nh_private.getParam("verbose", verbose);
  ros::NodeHandle nh;

  nh.setParam("/is_classic", true);
  nh.setParam("/matcher_type", "BF");
  nh.setParam("/selector_type", "KNN");
  nh.setParam("/conf_thresh", 0.015);
  nh.setParam("/dist_thresh", 4);
  nh.setParam("/num_threads", 8);
  nh.setParam("/border_remove", 4);
  nh.setParam("/stereo_threshold", 2.0);
  nh.setParam("/min_disparity", 0.25);
  nh.setParam("/refinement_degree", 4);
  nh.setParam("/verbose", verbose);
  nh.setParam("/device", device);
  nh.setParam("/machine_name", device);
  nh.setParam("/rosbag_rate", rosbag_rate);

  // nh.setParam("/model_id", 0);
  for (const auto &detector : detectors) {
    for (const auto &descriptor : descriptors) {
      for (const auto &resolution : resolutions) {
        detector_list.push_back(detector);
        descriptor_list.push_back(descriptor);
        const int height = resolution.first;
        const int width = resolution.second;
        height_list.push_back(height);
        width_list.push_back(width);
        description_list.push_back(detector + "_" + descriptor + "_" +
                                   std::to_string(height) + "_" +
                                   std::to_string(width));
        ROS_INFO("[data_processing_eval_node]\n%d-th configuration = %s\n",
                 (int)description_list.size() - 1,
                 description_list.back().c_str());
      }
    }
  }
  ROS_INFO("[data_processing_eval_node]\n In total, %lu configurations\n",
           description_list.size());

  // http://wiki.ros.org/actionlib_tutorials/Tutorials/SimpleActionServer%28ExecuteCallbackMethod%29
  ros::Subscriber data_lodaer_result_sub = nh.subscribe(
      "/kitti_loader_action_server/result", 1000, dataLodaerResultCallback);
  pub_goal = nh.advertise<odml_data_processing::kitti_data_loaderActionGoal>(
      "/kitti_loader_action_server/goal", 10);

  for (int i = 0; i < 3; ++i) {
    ROS_INFO("[data_processing_eval_node]\ncountdown: %d sec\n", 3 - i);
    sleep(1);
  }

  ros::Rate loop_rate(100);
  int seq_id = seq_ids.size();
  while (ros::ok() && config_id < (int)description_list.size()) {
    if (loading_finished == true) {
      if (config_id >= 0)
        ROS_INFO("[data_processing_eval_node]\ndata loading for seq %d is "
                 "finished\n",
                 seq_id);

      // start a new model to eval
      if (seq_id == seq_ids.size()) {
        if (config_id == description_list.size() - 1) {
          ROS_INFO(
              "[data_processing_eval_node]\ndata loading for all sequences "
              "and models is finished. Quitting "
              "data_processing_eval_node...\n");
          break;
        }
        ++config_id;
        seq_id = 0;

        ROS_INFO("[data_processing_eval_node]\nnew round of seqs. loading "
                 "%d-th configuration (%s)\n",
                 config_id, description_list.at(config_id).c_str());
        ROS_INFO("[data_processing_eval_node]\ndevice: %s, config_id: %d\n",
                 device.c_str(), config_id);
        
        nh.setParam("/model_id", config_id);
        nh.setParam("/detector_type", detector_list.at(config_id));
        nh.setParam("/descriptor_type", descriptor_list.at(config_id));
        nh.setParam("/image_height", height_list.at(config_id));
        nh.setParam("/image_width", width_list.at(config_id));
        ROS_INFO("[data_processing_eval_node]\nnew parameters are set\n");
      }

      odml_data_processing::kitti_data_loaderActionGoal goal;
      goal.goal.kitti_eval_id = seq_ids.at(seq_id);
      goal.goal.description = description_list.at(config_id);
      ROS_INFO("[data_processing_eval_node]\nsending new goal now: "
               "kitti_eval_id = %d = seq_ids.at(%d), description = %s\n",
               goal.goal.kitti_eval_id, seq_id, goal.goal.description.c_str());
      pub_goal.publish(goal);
      ++seq_id;
      loading_finished = false;
    }
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}