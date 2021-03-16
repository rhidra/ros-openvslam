Unofficial fork of OpenVSLAM for usage with ROS.

Modify the `run_slam.cc` and `run_localization.cc` files to publish point cloud and localization data to ROS topics. To install the package, follow the official OpenVSLAM documentation.

Topics published:

- `/openvslam/point_cloud` (`sensor_msgs/PointCloud2`): tracking sparse point cloud
- `/openvslam/camera_pose` (`geometry_msgs/PoseStamped`): camera pose

Topics subscribed:

- `/camera/image_raw` (`sensor_msgs/Image`): input RGB(-D) video stream