// Lane Detector
#include <bits/stdc++.h>
#include "LaneDetector.cpp"
// ROS libraries
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
// OpenCV libraries
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

sensor_msgs::ImagePtr curFrame;

void receive_img(sensor_msgs::ImagePtr &img) {
    curFrame = img;
}

int main(int argc, char** argv){
    // ROS initialization
    ros::init(argc, argv, "lane_detector");
    ros::NodeHandle handler;
    image_transport::ImageTransport imageHandler(handler);

    image_transport::Subscriber camSub = imageHandler.subscribe("/dash_cam", 10, receive_img);
    Mat frame = cv_bridge::toCvCopy(curFrame)->image;
    LaneDetector lanes(frame);
    while(handler.ok()){
        // Check if selected source is sending information
        if(frame.empty()){
            cout<<"NULL frame ";
            
        }else{
            lanes.load_frame(frame);
            lanes.find_lanes();
            lanes.find_center();
            lanes.predict_center();
            lanes.display(frame);
            // Wait 5 miliseconds
            // Read key board input, setting esc as break key
            if(waitKey(5)== 27){
                break;
            }
        }
    }
    return 0;
}
