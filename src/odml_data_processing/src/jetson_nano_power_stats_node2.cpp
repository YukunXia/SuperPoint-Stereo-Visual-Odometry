#include <odml_data_processing/kitti_data_loaderActionGoal.h>
#include <odml_data_processing/kitti_data_loaderActionResult.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include <fstream>
#include <string>

float total_energy = 0.0f;
float curr_total_energy = 0.0f;
int num_of_inference = 0;
bool is_data_loading = false;
bool is_visual_odometry_on = false;
std::string energy_log_file_name;
std::ofstream energy_log_file;
ros::Time prev_time;
ros::Duration total_delta_time;

void kittiGoalCallback(
    const odml_data_processing::kitti_data_loaderActionGoalConstPtr
        &kiti_data_action_goal,
    ros::NodeHandle *nh) {
  const int seq_id = kiti_data_action_goal->goal.kitti_eval_id;
  ROS_INFO("[jetson_nano_power_stats_node2]:\nnew kittiGoalCallback msg "
           "received, seq_id = %d\n",
           seq_id);
  energy_log_file_name =
      "_seq_" + std::to_string(kiti_data_action_goal->goal.kitti_eval_id) +
      ".log";

  bool is_classic;
  nh->getParam("/is_classic", is_classic);
  if (!is_classic) {
    std::string detector_type;
    std::string descriptor_type;
    nh->getParam("/detector_type", detector_type);
    nh->getParam("/descriptor_type", descriptor_type);
    if (detector_type == "SuperPoint" && descriptor_type == "SuperPoint") {
      std::string model_name_prefix;
      int model_batch_size;
      std::string trt_precision;
      int image_height;
      int image_width;
      nh->getParam("/model_name_prefix", model_name_prefix);
      nh->getParam("/model_batch_size", model_batch_size);
      nh->getParam("/trt_precision", trt_precision);
      nh->getParam("/image_height", image_height);
      nh->getParam("/image_width", image_width);
      energy_log_file_name = ros::package::getPath("odml_data_processing") +
                             "/kitti_energy_logs/" + model_name_prefix + "_" +
                             std::to_string(model_batch_size) + "_" +
                             std::to_string(image_height) + "_" +
                             std::to_string(image_width) + "_" + trt_precision +
                             energy_log_file_name;
    }
  } else {
    energy_log_file_name =
        ros::package::getPath("odml_data_processing") + "/kitti_energy_logs/" +
        kiti_data_action_goal->goal.description + energy_log_file_name;
  }

  ROS_INFO("[jetson_nano_power_stats_node2]:\nenergy_log_file_name = %s\n",
           energy_log_file_name.c_str());
  energy_log_file.open(energy_log_file_name);
  assert(energy_log_file.is_open());

  is_data_loading = true;
  curr_total_energy = 0.0f;
  total_energy = 0.0f;
  num_of_inference = 0;
  is_visual_odometry_on = false;
  total_delta_time = ros::Duration(0.0);
}

void kittiResultCallback(
    const odml_data_processing::kitti_data_loaderActionResultConstPtr
        &kiti_data_action_result) {
  assert(kiti_data_action_result->result.loading_finished == true);
  energy_log_file << num_of_inference << "," << total_energy;
  energy_log_file.close();
}

void updateTimeAndEnergy() {
  std::ifstream total_power_file;
  total_power_file.open(
      "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input");
  assert(total_power_file.is_open());
  std::string total_power_str;
  total_power_file >> total_power_str;
  const float total_power = std::stof(total_power_str) / 1000.0f;
  // ROS_INFO("[jetson_nano_power_stats_node2]\n total_power = %.4f\n",
  //          total_power);

  const ros::Time curr_time = ros::Time::now();
  const float delta_time = (curr_time - prev_time).toSec();
  curr_total_energy += total_power * delta_time;

  // ROS_INFO(
  //     "[jetson_nano_power_stats_node2]:\ndelta_time = %.4f s, power = %.4f
  //     w\n", delta_time, total_power);
  total_delta_time += curr_time - prev_time;
  prev_time = curr_time;
}

void energyUpdatingSignalCallback(
    const std_msgs::Bool::ConstPtr energy_updating_signal) {
  if (!is_data_loading) {
    ROS_WARN("[jetson_nano_power_stats_node2]:\nenergy updating signal (%d) "
             "received when not loading data\n",
             energy_updating_signal->data);
  }
  assert(is_visual_odometry_on != energy_updating_signal->data);

  is_visual_odometry_on = energy_updating_signal->data;
  if (is_visual_odometry_on) {
    prev_time = ros::Time::now();
    total_delta_time = ros::Duration(0.0);
    curr_total_energy = 0.0f;
    ++num_of_inference;
    ROS_INFO("[jetson_nano_power_stats_node2]:\nenergy_updating = True\n");
  } else {
    updateTimeAndEnergy();
    total_energy += curr_total_energy;
    ROS_INFO("[jetson_nano_power_stats_node2]:\nenergy_updating = False\n");
    ROS_INFO(
        "[jetson_nano_power_stats_node2]:\ntotal energy = %.4f, "
        "num_of_inference = %d, energy per inference = %.4f\n, "
        "total_delta_time = %.4f s, current inference average power = %.4f",
        total_energy, num_of_inference, total_energy / (float)num_of_inference,
        total_delta_time.toSec(), curr_total_energy / total_delta_time.toSec());
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "jetson_nano_power_stats_node2");
  ros::NodeHandle nh;

  ros::Subscriber data_lodaer_goal_sub =
      nh.subscribe<odml_data_processing::kitti_data_loaderActionGoal>(
          "/kitti_loader_action_server/goal", 100,
          boost::bind(&kittiGoalCallback, _1, &nh));
  ros::Subscriber data_lodaer_result_sub =
      nh.subscribe<odml_data_processing::kitti_data_loaderActionResult>(
          "/kitti_loader_action_server/result", 100, kittiResultCallback);
  ros::Subscriber energy_updating_signal_sub = nh.subscribe<std_msgs::Bool>(
      "/energy_updating_signal", 100, energyUpdatingSignalCallback);

  ROS_INFO("[jetson_nano_power_stats_node2]\n ready\n");
  ros::Rate loop_rate(500);
  while (ros::ok()) {
    if (is_visual_odometry_on)
      updateTimeAndEnergy();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}