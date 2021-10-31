# Troubleshooting

## SIFT not declared
> error: ‘cv::SIFT’ has not been declared
>  125 |     detector_ = cv::SIFT::create();

Solution:

`sudo subl /opt/ros/noetic/share/cv_bridge/cmake/cv_bridgeConfig.cmake`

At line 94-96, replace opencv path to the true one with >=4.5.1 version. eg. 

```
if(NOT "include;/usr/local/include/opencv4 " STREQUAL " ")
  set(cv_bridge_INCLUDE_DIRS "")
  set(_include_dirs "include;/usr/local/include/opencv4")
```