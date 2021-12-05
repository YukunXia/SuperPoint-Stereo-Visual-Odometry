#! /usr/bin/env python

import rospy
import rospkg
import std_msgs.msg
from jtop import jtop
import odml_data_processing.msg

total_energy = 0.0
num_of_inference = 0
is_data_loading = False
is_visual_odometry_on = False

jetson = jtop()

seq_id = -1
energy_log_file_name = ""


def kittiGoalCallback(goal_msg):
    global is_data_loading, prev_time, total_energy, num_of_inference, seq_id, energy_log_file_name, log_file, is_visual_odometry_on

    seq_id = goal_msg.goal.kitti_eval_id
    rospy.loginfo(
        "[jetson_nano_power_stats_node]:\nnew kittiGoalCallback msg received, seq_id = {}\n".format(seq_id))

    is_classic = rospy.get_param("/is_classic")
    assert(is_classic == False)  # not implemented yet
    if not is_classic:
        detector_type = rospy.get_param("/detector_type")
        descriptor_type = rospy.get_param("/descriptor_type")
        assert(detector_type == "SuperPoint" and descriptor_type == "SuperPoint")
        model_name_prefix = rospy.get_param("/model_name_prefix")
        model_batch_size = rospy.get_param("/model_batch_size")
        image_height = rospy.get_param("/image_height")
        image_width = rospy.get_param("/image_width")
        trt_precision = rospy.get_param("/trt_precision")
        energy_log_file_name = model_name_prefix + \
            "_" + str(model_batch_size) + "_" + str(image_height) + \
            "_" + str(image_width) + "_" + trt_precision + \
            "_seq_" + str(seq_id) + ".log"
        rospack = rospkg.RosPack()
        energy_log_file_name = rospack.get_path(
            "odml_data_processing") + "/kitti_energy_logs/" + energy_log_file_name
        rospy.loginfo("[jetson_nano_power_stats_node]:\nenergy_log_file_name = {}\n".format(
            energy_log_file_name))
        log_file = open(energy_log_file_name, "w")

    is_data_loading = True
    total_energy = 0.0
    num_of_inference = 0
    is_visual_odometry_on = False


def kittiLoaderResultCallback(result_msg):
    global log_file, is_data_loading

    assert(result_msg.result.loading_finished == True)

    log_file.write("{},{}".format(num_of_inference, total_energy))
    log_file.close()

    is_data_loading = False


def energyUpdatingSignalCallback(energy_updating_signal):
    global is_visual_odometry_on, prev_time, total_energy, num_of_inference
    if not is_data_loading:
        rospy.logwarn("[jetson_nano_power_stats_node]:\nenergy updating signal ({}) received when not loading data\n".format(
            energy_updating_signal.data))
    assert(is_visual_odometry_on != energy_updating_signal.data)

    is_visual_odometry_on = energy_updating_signal.data
    if (is_visual_odometry_on):
        prev_time = rospy.Time.now()
        num_of_inference += 1
        rospy.loginfo(
            "[jetson_nano_power_stats_node]:\nenergy_updating = True\n")
    else:
        updateTimeAndEnergy()
        rospy.loginfo(
            "[jetson_nano_power_stats_node]:\nenergy_updating = False\n")
        rospy.loginfo("[jetson_nano_power_stats_node]:\ntotal energy = {}, num_of_inference = {}, energy per inference = {}\n".format(
            total_energy, num_of_inference, total_energy / num_of_inference))


def updateTimeAndEnergy():
    global prev_time, total_energy, jetson, num_of_inference

    curr_time = rospy.Time.now()

    delta_time = (curr_time - prev_time).to_sec()
    while not (jetson.ok()):
        jetson.start()
    power = jetson.stats["power cur"] / 1000.0
    total_energy += power * delta_time

    rospy.loginfo("[jetson_nano_power_stats_node]:\ndelta_time = {} s, power = {}w\n".format(
        delta_time, power))
    prev_time = curr_time


if __name__ == '__main__':
    rospy.init_node('jetson_nano_power_stats_node')
    rospy.Subscriber("/kitti_loader_action_server/goal",
                     odml_data_processing.msg.kitti_data_loaderActionGoal, kittiGoalCallback, queue_size=100)
    rospy.Subscriber("/energy_updating_signal",
                     std_msgs.msg.Bool, energyUpdatingSignalCallback, queue_size=100)
    rospy.Subscriber("/kitti_loader_action_server/result",
                     odml_data_processing.msg.kitti_data_loaderActionResult, kittiLoaderResultCallback, queue_size=100)

    jetson.start()
    if not jetson.ok():
        rospy.logerr("[jetson_nano_power_stats_node]:\nfailed to start jtop\n")
    loop_rate = rospy.Rate(500)
    rospy.loginfo(
        "[jetson_nano_power_stats_node]:\nstarts, jetson.ok() = {}\n".format(jetson.ok()))
    while not rospy.is_shutdown():
        if is_visual_odometry_on:
            updateTimeAndEnergy()
        loop_rate.sleep()
