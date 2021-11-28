#include <ros/ros.h>

#include <odml_data_processing/kitti_data_loaderActionGoal.h>
#include <odml_data_processing/kitti_data_loaderActionResult.h>

bool loading_finished = false;
ros::Publisher pub_goal;

void dataLodaerResultCallback(
    const odml_data_processing::kitti_data_loaderActionResultConstPtr
        &kiti_data_action_result) {
  assert(kiti_data_action_result->result.loading_finished == true);
  loading_finished = true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "data_processing_eval_node");

  

  ros::NodeHandle nh;

  // http://wiki.ros.org/actionlib_tutorials/Tutorials/SimpleActionServer%28ExecuteCallbackMethod%29
  ros::Subscriber data_lodaer_result_sub = nh.subscribe(
      "/kitti_loader_action_server/result", 1000, dataLodaerResultCallback);
  pub_goal = nh.advertise<odml_data_processing::kitti_data_loaderActionGoal>(
      "/kitti_loader_action_server/goal", 10);

  for (int i=0; i<10; ++i) {
    ROS_INFO("countdown: %d sec", 10-i);
    sleep(1);
  }

  ros::Rate loop_rate(100);
  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}