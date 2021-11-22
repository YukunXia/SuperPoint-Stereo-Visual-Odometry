Example of deployment:

b2 means batchsize = 2
-> can be optimized to only storing 1 onnx with dynamic batch size, width and height.

`trtexec --onnx=superpoint_pretrained_b2.onnx --explicitBatch --minShapes=input:2x1x360x1176 --optShapes=input:2x1x360x1176 --maxShapes=input:2x1x360x1176 --workspace=4096 --saveEngine=superpoint_pretrained_2_360_1176_FP32.engine --verbose`