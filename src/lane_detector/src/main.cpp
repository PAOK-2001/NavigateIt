// Lane Detector
#include <bits/stdc++.h>
#include "LaneDetector.cpp"
// ROS libraries
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
// OpenCV libraries
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

int nodeRate = 25;
Mat frame, msg;
std_msgs::Float32 output;
std_msgs::Int32 intersectionLikely;
std_msgs::Int32 lineCount;

void receive_img(const sensor_msgs::ImageConstPtr &img) {
    msg = cv_bridge::toCvShare(img,"bgr8")->image;
    frame = msg.clone();
}

int main(int argc, char** argv){
    // ROS initialization
    int intersectionCount = 0;
    ros::init(argc, argv, "lane_detector");
    ros::NodeHandle handler;
    ros::Rate rate(nodeRate);
    ros::Publisher pixelsToCenter = handler.advertise<std_msgs::Float32>("/line_error", 10);
    ros::Publisher intersectionLikelyPub = handler.advertise<std_msgs::Int32>("/intersection_likely", 10);
    ros::Publisher lineCountPub = handler.advertise<std_msgs::Int32>("/line_count", 10);
    image_transport::ImageTransport imageHandler(handler);

    image_transport::Subscriber camSub = imageHandler.subscribe("/video_source/dash_cam", 10, receive_img);
    while(frame.empty()){ros::spinOnce();}
    LaneDetector lanes(frame);
    while(handler.ok()&& ros::ok()){	
        if(frame.empty()){
            cout<<"NULL frame ";   
        }
        else{
            lanes.load_frame(frame);
            lanes.find_lanes();
            intersectionCount = lanes.intersectionLikely ? min(intersectionCount+1,75) : max(intersectionCount-1,0);
            intersectionLikely.data = intersectionCount;
            lineCount.data = lanes.lineCount;
            lanes.predict_center();
            //lanes.display(frame);
            output.data = lanes.getWidth()/2-lanes.getCenterx();
            intersectionLikelyPub.publish(intersectionLikely);
            lineCountPub.publish(lineCount);
            pixelsToCenter.publish(output);
            if(waitKey(1)== 27){
                break;
            }
        }
        ros::spinOnce();
        rate.sleep();	
    }
    return 0;
}
