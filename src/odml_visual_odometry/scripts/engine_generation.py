#!/usr/bin/env python

# import rospy
import rospkg
import subprocess
import glob
import time
from os.path import exists

rospack = rospkg.RosPack()
odml_visual_odometry_path = rospack.get_path(
    'odml_visual_odometry') + "/models/"
model_prefices = ["superpoint_pretrained",
                  "sp_sparse", "sp_mbv1", "sp_mbv2", "sp_squeeze", "sp_resnet18"]
<<<<<<< HEAD
device = "jetson"
=======
device = "laptop"
>>>>>>> origin/long_term_eval
workspaces = {"workstation": 4096, "laptop": 4096, "jetson": 3072}
onnx_files = set([path.split("/")[-1]
                 for path in glob.glob(odml_visual_odometry_path + "*.onnx")])
print("onnx files are: ", onnx_files)
resolutions = [(360, 1176), (240, 784), (120, 392)]
precisions = ["32", "16"]

test_mode = False

for model_prefix in model_prefices:
    for b in range(1, 3):
        onnx_file_name = model_prefix+"_b" + str(b) + ".onnx"
        if onnx_file_name in onnx_files:
            print("found onnx file: ", onnx_file_name)
            for resolution in resolutions:
                for precision in precisions:
                    width = resolution[0]
                    height = resolution[1]
                    # TODO: check if the folder exists
                    engine_file = "{}/{}_{}_{}_{}_FP{}.engine".format(
                        device, model_prefix, b, width, height, precision)
                    if exists(odml_visual_odometry_path + engine_file):
                        print("engine_file ({}) already exists".format(engine_file))
                        continue
                    else:
                        print("engine_file ({}) will be created".format(engine_file))

                    if test_mode:
                        continue

                    if precision == "16":
                        trt_proc = subprocess.Popen(
                            ['trtexec', '--onnx={}'.format(odml_visual_odometry_path+onnx_file_name), "--explicitBatch", "--minShapes=input:{}x1x{}x{}".format(b, width, height), "--optShapes=input:{}x1x{}x{}".format(b, width, height), "--maxShapes=input:{}x1x{}x{}".format(b, width, height), "--workspace={}".format(workspaces[device]), "--saveEngine={}".format(odml_visual_odometry_path+engine_file), "--fp{}".format(precision)])
                    elif precision == "32":
                        trt_proc = subprocess.Popen(
                            ['trtexec', '--onnx={}'.format(odml_visual_odometry_path+onnx_file_name), "--explicitBatch", "--minShapes=input:{}x1x{}x{}".format(b, width, height), "--optShapes=input:{}x1x{}x{}".format(b, width, height), "--maxShapes=input:{}x1x{}x{}".format(b, width, height), "--workspace={}".format(workspaces[device]), "--saveEngine={}".format(odml_visual_odometry_path+engine_file)])
                    else:
                        print("no such precision")
                        continue
                    trt_proc.wait()
                    time.sleep(60.0)
        else:
            print("didn't find onnx file: ", onnx_file_name)
