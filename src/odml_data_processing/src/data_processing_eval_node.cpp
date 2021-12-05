#include <ros/ros.h>

#include <odml_data_processing/kitti_data_loaderActionGoal.h>
#include <odml_data_processing/kitti_data_loaderActionResult.h>

#include <string>
#include <vector>

bool loading_finished = true;
ros::Publisher pub_goal;
int model_id = -1;

std::vector<std::string> engine_file_list;
std::vector<std::string> model_prefix_list;
std::vector<int> batch_choice_list;
std::vector<int> width_list;
std::vector<int> height_list;
std::vector<std::string> precision_list;

std::vector<std::string> model_prefices = {"superpoint_pretrained",
                                          "sp_sparse",
                                           "sp_mbv1",
                                           "sp_mbv2",
                                           "sp_squeeze",
                                           "sp_resnet18"};
std::vector<int> batch_choices = {1, 2};
std::vector<std::pair<int, int>> resolutions = {
    {360, 1176}, {240, 784}, {120, 392}};
std::vector<std::string> precisions = {"32", "16"};
std::vector<int> seq_ids = {4};

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

  nh.setParam("/is_classic", false);
  nh.setParam("/detector_type", "SuperPoint");
  nh.setParam("/descriptor_type", "SuperPoint");
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
  for (const auto &model_prefix : model_prefices) {
    for (const auto &batch_choice : batch_choices) {
      for (const auto &resolution : resolutions) {
        const int height = resolution.first;
        const int width = resolution.second;
        for (const auto &precision : precisions) {
          const std::string engine_file =
              model_prefix + "_" + std::to_string(batch_choice) + "_" +
              std::to_string(width) + "_" + std::to_string(height) + "_FP" +
              precision;
          engine_file_list.push_back(engine_file);
          ROS_INFO("[data_processing_eval_node]\n%d-th engine_file = %s\n",
                   (int)engine_file_list.size() - 1, engine_file.c_str());

          model_prefix_list.push_back(model_prefix);
          batch_choice_list.push_back(batch_choice);
          height_list.push_back(height);
          width_list.push_back(width);
          precision_list.push_back("FP" + precision);
        }
      }
    }
  }
  ROS_INFO("[data_processing_eval_node]\n In total, %lu engine files\n",
           engine_file_list.size());

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
  while (ros::ok() && model_id < (int)model_prefix_list.size()) {
    if (loading_finished == true) {
      if (model_id >= 0)
        ROS_INFO(
            "[data_processing_eval_node]\ndata loading for seq %d finished\n",
            seq_id);

      // start a new model to eval
      if (seq_id == seq_ids.size()) {
        if (model_id == model_prefix_list.size() - 1) {
          ROS_INFO(
              "[data_processing_eval_node]\ndata loading for all sequences "
              "and models is finished. Quitting "
              "data_processing_eval_node...\n");
          break;
        }
        ++model_id;
        seq_id = 0;

        ROS_INFO("[data_processing_eval_node]\nnew round of seqs. loading "
                 "%d-th engine (%s)\n",
                 model_id, engine_file_list.at(model_id).c_str());
        ROS_INFO("[data_processing_eval_node]\ndevice: %s, model_id: %d\n",
                 device.c_str(), model_id);

        nh.setParam("/model_id", model_id);
        nh.setParam("/image_height", height_list.at(model_id));
        nh.setParam("/image_width", width_list.at(model_id));
        nh.setParam("/model_name_prefix", model_prefix_list.at(model_id));
        nh.setParam("/model_batch_size", batch_choice_list.at(model_id));
        nh.setParam("/trt_precision", precision_list.at(model_id));
        ROS_INFO("[data_processing_eval_node]\nnew parameters are set\n");
      }

      odml_data_processing::kitti_data_loaderActionGoal goal;
      goal.goal.kitti_eval_id = seq_ids.at(seq_id);
      goal.goal.description = engine_file_list.at(model_id);
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