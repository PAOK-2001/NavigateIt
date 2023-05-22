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

int main(int argc, char** argv){
    // ROS initialization
    ros::init(argc, argv, "lane_detector");
    ros::NodeHandle handler;
    image_transport::ImageTransport imageHandler(handler);
    image_transport::Publisher videoPub = imageHandler.advertise("video", 5); 
    Mat frame;
    sensor_msgs::ImagePtr msg;
    string  cam_port =  "nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true";
    VideoCapture dashCam(cam_port);
    // Check if the dashCam is readable
    if(!dashCam.isOpened()){
        cout<<"Error reading dashCam feed\n";
        return -1;
    }
    dashCam.read(frame);
    LaneDetector lanes(frame);
    while(handler.ok()){
        dashCam.read(frame);
        if(frame.empty()){
            cout<<"NULL frame ";
            
        }else{
            lanes.load_frame(frame);
            lanes.find_lanes();
            lanes.find_center();
            lanes.predict_center();
            lanes.display(frame);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            videoPub.publish(msg);
            if(waitKey(5)== 27){
                break;
            }
        }
    }
    return 0;
}
