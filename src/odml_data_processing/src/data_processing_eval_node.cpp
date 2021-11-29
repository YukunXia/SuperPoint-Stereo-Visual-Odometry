#include <ros/ros.h>

#include <odml_data_processing/kitti_data_loaderActionGoal.h>
#include <odml_data_processing/kitti_data_loaderActionResult.h>

#include <string>
#include <vector>

bool loading_finished = true;
ros::Publisher pub_goal;
int model_id = -1;
std::string device;

std::vector<std::string> engine_file_list;
std::vector<std::string> model_prefix_list;
std::vector<int> batch_choice_list;
std::vector<int> width_list;
std::vector<int> height_list;
std::vector<std::string> precision_list;

std::vector<std::string> model_prefices = {"superpoint_pretrained", "sp_sparse",
                                           "sp_mbv1", "sp_mbv2", "sp_squeeze"};
std::vector<int> batch_choices = {1, 2};
std::vector<std::pair<int, int>> resolutions = {
    {360, 1176}, {240, 784}, {120, 392}};
std::vector<std::string> precisions = {"32", "16"};
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
  nh_private.getParam("device", device);
  ros::NodeHandle nh;

  nh.setParam("/is_classic", false);
  nh.setParam("/detector_type", "SuperPoint");
  nh.setParam("/descriptor_type", "SuperPoint");
  nh.setParam("/matcher_type", "BF");
  nh.setParam("/selector_type", "KNN");
  nh.setParam("/machine_name", "workstation");
  nh.setParam("/conf_thresh", 0.015);
  nh.setParam("/dist_thresh", 4);
  nh.setParam("/num_threads", 8);
  nh.setParam("/border_remove", 4);
  nh.setParam("/stereo_threshold", 2.0);
  nh.setParam("/min_disparity", 0.25);
  nh.setParam("/refinement_degree", 4);
  nh.setParam("/verbose", false);

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
          ROS_INFO("%d-th engine_file = %s", (int)engine_file_list.size() - 1,
                   engine_file.c_str());

          model_prefix_list.push_back(model_prefix);
          batch_choice_list.push_back(batch_choice);
          height_list.push_back(height);
          width_list.push_back(width);
          precision_list.push_back("FP" + precision);
        }
      }
    }
  }
  ROS_INFO("\n In total, %lu engine files", engine_file_list.size());

  // http://wiki.ros.org/actionlib_tutorials/Tutorials/SimpleActionServer%28ExecuteCallbackMethod%29
  ros::Subscriber data_lodaer_result_sub = nh.subscribe(
      "/kitti_loader_action_server/result", 1000, dataLodaerResultCallback);
  pub_goal = nh.advertise<odml_data_processing::kitti_data_loaderActionGoal>(
      "/kitti_loader_action_server/goal", 10);

  for (int i = 0; i < 3; ++i) {
    ROS_INFO("countdown: %d sec", 3 - i);
    sleep(1);
  }

  ros::Rate loop_rate(100);
  int seq_id = seq_ids.size();
  while (ros::ok() && model_id < (int)model_prefix_list.size()) {
    if (loading_finished == true) {
      if (model_id >= 0)
        ROS_INFO("data loading for seq %d finished", seq_id);

      // start a new model to eval
      if (seq_id == seq_ids.size()) {
        ++model_id;
        seq_id = 0;
        nh.setParam("/device", device);
        nh.setParam("/model_id", model_id);
        nh.setParam("/image_height", height_list.at(model_id));
        nh.setParam("/image_width", width_list.at(model_id));
        nh.setParam("/model_name_prefix", model_prefix_list.at(model_id));
        nh.setParam("/model_batch_size", batch_choice_list.at(model_id));
        nh.setParam("/trt_precision", precision_list.at(model_id));

        ROS_INFO("loading %d-th engine (%s)", model_id,
                 engine_file_list.at(model_id).c_str());
      }

      odml_data_processing::kitti_data_loaderActionGoal goal;
      goal.goal.kitti_eval_id = seq_ids.at(seq_id);
      goal.goal.description = engine_file_list.at(model_id);
      pub_goal.publish(goal);
      ++seq_id;
      loading_finished = false;
    }
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}