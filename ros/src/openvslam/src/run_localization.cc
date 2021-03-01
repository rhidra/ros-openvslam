#ifdef USE_PANGOLIN_VIEWER
#include <pangolin_viewer/viewer.h>
#elif USE_SOCKET_PUBLISHER
#include <socket_publisher/publisher.h>
#endif

#include <openvslam/system.h>
#include <openvslam/config.h>

#include <iostream>
#include <chrono>
#include <numeric>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>
#include "openvslam/data/landmark.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/map_database.h"
#include "openvslam/publish/map_publisher.h"

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

void pose_odometry_pub(auto cam_pose_, auto pc2_msg_, auto pose_pub_, auto odometry_pub_, auto point_cloud_pub_){
    Eigen::Matrix3d rotation_matrix = cam_pose_.block(0, 0, 3, 3);
    Eigen::Vector3d translation_vector = cam_pose_.block(0, 3, 3, 1);

    tf2::Matrix3x3 tf_rotation_matrix(rotation_matrix(0, 0), rotation_matrix(0, 1), rotation_matrix(0, 2),
                                      rotation_matrix(1, 0), rotation_matrix(1, 1), rotation_matrix(1, 2),
                                      rotation_matrix(2, 0), rotation_matrix(2, 1), rotation_matrix(2, 2));
    
    tf2::Vector3 tf_translation_vector(translation_vector(0), translation_vector(1), translation_vector(2));

    tf_rotation_matrix = tf_rotation_matrix.inverse();
    tf_translation_vector = -(tf_rotation_matrix * tf_translation_vector);

    tf2::Transform transform_tf(tf_rotation_matrix, tf_translation_vector);

    tf2::Matrix3x3 rot_open_to_ros (0, 0, 1,
                                  -1, 0, 0,
                                   0,-1, 0);

    tf2::Transform transformA(rot_open_to_ros, tf2::Vector3(0.0, 0.0, 0.0));
    tf2::Transform transformB(rot_open_to_ros.inverse(), tf2::Vector3(0.0, 0.0, 0.0));

    transform_tf = transformA * transform_tf * transformB;

    ros::Time now = ros::Time::now();

    nav_msgs::Odometry odom_msg_;
    odom_msg_.header.stamp = now;
    odom_msg_.header.frame_id = "map";
    odom_msg_.child_frame_id = "base_link_frame";
    odom_msg_.pose.pose.orientation.x = transform_tf.getRotation().getX();
    odom_msg_.pose.pose.orientation.y = transform_tf.getRotation().getY();
    odom_msg_.pose.pose.orientation.z = transform_tf.getRotation().getZ();
    odom_msg_.pose.pose.orientation.w = transform_tf.getRotation().getW();
    odom_msg_.pose.pose.position.x = transform_tf.getOrigin().getX();
    odom_msg_.pose.pose.position.y = transform_tf.getOrigin().getY();
    odom_msg_.pose.pose.position.z = transform_tf.getOrigin().getZ();
    odometry_pub_.publish(odom_msg_);

    // Create pose message and update it with current camera pose
    geometry_msgs::PoseStamped camera_pose_msg_;
    camera_pose_msg_.header.stamp = now;
    camera_pose_msg_.header.frame_id = "map";
    camera_pose_msg_.pose.position.x = transform_tf.getOrigin().getX();
    camera_pose_msg_.pose.position.y = transform_tf.getOrigin().getY();
    camera_pose_msg_.pose.position.z = transform_tf.getOrigin().getZ();
    camera_pose_msg_.pose.orientation.x = transform_tf.getRotation().getX();
    camera_pose_msg_.pose.orientation.y = transform_tf.getRotation().getY();
    camera_pose_msg_.pose.orientation.z = transform_tf.getRotation().getZ();
    camera_pose_msg_.pose.orientation.w = transform_tf.getRotation().getW();
    pose_pub_.publish(camera_pose_msg_);

    // Point cloud publish
    point_cloud_pub_.publish(pc2_msg_);
}

void mono_localization(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path,
                       const std::string& mask_img_path, const std::string& map_db_path, const bool mapping) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // load the prebuilt map
    SLAM.load_map_database(map_db_path);
    // startup the SLAM process (it does not need initialization of a map)
    SLAM.startup(false);
    // select to activate the mapping module or not
    if (mapping) {
        SLAM.enable_mapping_module();
    }
    else {
        SLAM.disable_mapping_module();
    }

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    std::vector<double> track_times;
    const auto tp_0 = std::chrono::steady_clock::now();

    // initialize this node
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    ros::Publisher camera_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/openvslam/camera_pose", 1);
    ros::Publisher odometry_pub_publisher = nh.advertise<nav_msgs::Odometry>("/openvslam/odometry", 1);
    ros::Publisher point_cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/openvslam/point_cloud", 1);

    // run the SLAM as subscriber
    image_transport::Subscriber sub = it.subscribe("camera/image_raw", 1, [&](const sensor_msgs::ImageConstPtr& msg) {
        const auto tp_1 = std::chrono::steady_clock::now();
        const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0).count();

        // input the current frame and estimate the camera pose
        auto cam_pose = SLAM.feed_monocular_frame(cv_bridge::toCvShare(msg, "bgr8")->image, timestamp, mask);

        const auto tp_2 = std::chrono::steady_clock::now();

        const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
        track_times.push_back(track_time);

        std::vector<openvslam::data::landmark*> all_landmarks;
        std::set<openvslam::data::landmark*> local_landmarks;
        SLAM.get_map_publisher()->get_landmarks(all_landmarks, local_landmarks);
        pcl::PointCloud<pcl::PointXYZ> cloud_;

        for (const auto& lm: all_landmarks) {
          auto pos = lm->get_pos_in_world();
          auto pt = pcl::PointXYZ();
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud_.points.push_back(pt);
        }
        auto pc2_msg_ = sensor_msgs::PointCloud2();
        pcl::toROSMsg(cloud_, pc2_msg_);
        pc2_msg_.header.frame_id = "map";
        // pc2_msg_->header.stamp = ros::now();

        pose_odometry_pub(cam_pose, pc2_msg_, camera_pose_publisher, odometry_pub_publisher, point_cloud_publisher);
    });

    // run the viewer in another thread
#ifdef USE_PANGOLIN_VIEWER
    std::thread thread([&]() {
        viewer.run();
        if (SLAM.terminate_is_requested()) {
            // wait until the loop BA is finished
            while (SLAM.loop_BA_is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
            ros::shutdown();
        }
    });
#elif USE_SOCKET_PUBLISHER
    std::thread thread([&]() {
        publisher.run();
        if (SLAM.terminate_is_requested()) {
            // wait until the loop BA is finished
            while (SLAM.loop_BA_is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
            ros::shutdown();
        }
    });
#endif

    ros::spin();

    // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
    viewer.request_terminate();
    thread.join();
#elif USE_SOCKET_PUBLISHER
    publisher.request_terminate();
    thread.join();
#endif

    // shutdown the SLAM process
    SLAM.shutdown();

    if (track_times.size()) {
        std::sort(track_times.begin(), track_times.end());
        const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
        std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
        std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
    }
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif
    ros::init(argc, argv, "run_localization");

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto setting_file_path = op.add<popl::Value<std::string>>("c", "config", "setting file path");
    auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db", "path to a prebuilt map database");
    auto mapping = op.add<popl::Switch>("", "mapping", "perform mapping as well as localization");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !setting_file_path->is_set() || !map_db_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(setting_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // run localization
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
        mono_localization(cfg, vocab_file_path->value(), mask_img_path->value(), map_db_path->value(), mapping->is_set());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
