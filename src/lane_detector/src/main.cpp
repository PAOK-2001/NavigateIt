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
    LaneDetector lanes;
    Mat frame;
    sensor_msgs::ImagePtr msg;
    // Create VideoCapture object
    int test;
    cout<<"Choose number of test footage to use\n";
    cin>>test;
    string path = "Test_Footage/Test"+to_string(test)+".mp4";
    cout << path;
    VideoCapture dashCam(path);
    // Check if the dashCam is readable
    if(!dashCam.isOpened()){
        cout<<"Error reading dashCam feed\n";
        return -1;
    }
    while(handler.ok()){
        dashCam.read(frame);
        // Check if selected source is sending information
        if(frame.empty()){
            cout<<"NULL frame ";
            
        }else{
            // Load image into lane detector
            lanes.load_frame(frame);
            // Find lanes on given frame
            lanes.find_lanes();
            // Find center of previously calculated lanes
            lanes.find_center();
            lanes.predict_center();
            // Overlap lanes on the video
            lanes.display(frame);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            videoPub.publish(msg);
            // Wait 5 miliseconds
            // Read key board input, setting esc as break key
            if(waitKey(5)== 27){
                break;
            }
        }
    }
    return 0;
}