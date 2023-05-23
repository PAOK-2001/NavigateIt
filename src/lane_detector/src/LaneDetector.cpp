#ifndef LANEDETECTOR_CPP
#define LANEDETECTOR_CPP

#include <bits/stdc++.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
using namespace std;
using namespace cv;

class LaneDetector{
    private:
        KalmanFilter center_estimator;
        // Images states used in lane detection
        Mat edgeImg, lineImg;
        // Lane lines in the format of a 4 integer vector such as  (x1, y1, x2, y2), which are the endpoint coordinates. 
        Vec4i centerLine;
        // Coordinate that represents the center of given lane
        Point2f center, predictedCenter;
        //Frame dimensions
        int width, height;
        // Auxiliary functions
        static pair<float,float> linear_fit(Vec4i lineCoordinates); 
        static float average_coefficient(vector<float> registeredCoefficients);
        static Vec4i make_coordinate(float slope, float intercept, int imgHeight, int imgWidth);

    public:
        LaneDetector();
        LaneDetector(Mat initFrame);
        void init_kf();
        void load_frame(Mat cameraFrame);
        void find_lanes();
        void find_center();
        void predict_center();
        void display(Mat cameraFrame);
};

// Default constructor of lane detector
LaneDetector::LaneDetector(){
    init_kf();
}

LaneDetector::LaneDetector(Mat initFrame){
    width = initFrame.cols;
    height = initFrame.rows;
    init_kf();
}

void LaneDetector::init_kf(){
    center_estimator = KalmanFilter(2,2);
    setIdentity(center_estimator.measurementMatrix);
    setIdentity(center_estimator.processNoiseCov, Scalar::all(1e-5));
    setIdentity(center_estimator.measurementNoiseCov, Scalar::all(1e-3));
    setIdentity(center_estimator.errorCovPost, Scalar::all(1));
}

void LaneDetector::predict_center(){
    cv::Mat measurement(2, 1, CV_32F);
    measurement.at<float>(0) = max(center.x - width/2, 0.0f);
    measurement.at<float>(1) = max(center.y - height/2, 0.0f);
    center_estimator.correct(measurement);
    Mat predictedState = center_estimator.predict();
    predictedCenter = Point2f(predictedState.at<float>(0) + width/2, predictedState.at<float>(1) + height/2);
}
// Find slope (rise/run) and intercept (b=y-slope(x)) given end coordinates
// @param Vec4i containing line end points (x1,y1,x2,y2)
// @return Pair containing <slope,intercept>
pair<float,float> LaneDetector::linear_fit(Vec4i lineCoordinates){
    float slope     = (lineCoordinates[3] -lineCoordinates[1])/(float)(lineCoordinates[2]-lineCoordinates[0]);
    float intercept =  lineCoordinates[1] - slope*lineCoordinates[0];
    return  make_pair(slope,intercept);
}

// Find the average from a vector of cohedicientes
// @param registerCoefficients Vector of floats
// @return avrg float representing average of vector. Retuns 0 if empty
float LaneDetector::average_coefficient(vector<float> registeredCoefficients){
    float size, sum;
    size = registeredCoefficients.size();
    if(size == 0){
        return 0;
    }
    for(auto & num: registeredCoefficients){
        sum+=num;
    }
    float avrg = sum/(float)size;
    return avrg;
}

// computes coordinate based on slope and intercept of line equation
// @param slope, intercept float representing linear equation values
// @return coordinates Vec4i in format [x1,y1,x2,y2]
Vec4i LaneDetector::make_coordinate(float slope, float intercept, int imgHeight, int imgWidth){
    int x1, y1, x2, y2;
    // Set default hight to bottom of image, assuming the hood of car isn't in frame
    y1 = imgHeight;
    // Second height is arbitrary value, depending on how far we want to visualize lanes.
    // #Note: there is a limit into how far the lane curvature holds.
    y2 = round(imgHeight*(1.0/1.2));
    // Calculate x coordinates solving in line eq. y = mx+b---> x = (y-b)/m making sure they are in the image
    x1 = min(max(0.0f, round((y1-intercept)/slope)), (float)imgWidth);
    x2 = min(max(0.0f, round((y2-intercept)/slope)), (float)imgWidth);
    // Build line coordinated structure
    Vec4i returnCoordinates = {x1,y1,x2,y2};
    return returnCoordinates;
}

// loads a frame and apply preprocessiong tecnques as well as masking to crop region of interest where lanes are
// @param cameraFrame OpenCV image matrix
void LaneDetector::load_frame(Mat cameraFrame){
    Mat grayScale, thresholded;
    cvtColor(cameraFrame,grayScale,COLOR_BGR2GRAY);
    threshold(grayScale, thresholded, 75, 255, THRESH_BINARY_INV);
    Canny(thresholded,edgeImg,40,150);
}

void LaneDetector::find_lanes(){
    Mat mask = edgeImg.clone();
    mask     = Scalar(0,0,0);
    lineImg  = mask;

    cvtColor(lineImg,lineImg,COLOR_GRAY2BGR);
    Point p1 = Point(99,719);
    Point p2 = Point(99,450);
    Point p3 = Point(249,359);
    Point p4 = Point(1029,359);
    Point p5 = Point(1179,450);
    Point p6 = Point(1179,719);
    vector<Point> ROI ={p1,p2,p3,p4,p5,p6};
    fillPoly(mask,ROI,(255,255,255));
    bitwise_and(mask,edgeImg,edgeImg);
    vector<Vec4i> lines;
    // Uses HoughTransform to fine most lines in canny image.
    HoughLinesP(edgeImg,lines,2,CV_PI/180,190,15,5);
    // Find regression for lines
    vector<float> slopes, intercepts;
    for(auto &lineP: lines){
        pair<float, float> fit =linear_fit(lineP);
        slopes.push_back(fit.first);
        intercepts.push_back(fit.second);
    }
    // For each side, find average lane
    float averageSlope     = average_coefficient(slopes);
    float averageIntercept = average_coefficient(intercepts);
    // Make coordinates for final lanes
    centerLine = make_coordinate(averageSlope, averageIntercept, height, width);
    // Draw line on lineImg
    line(lineImg,Point(centerLine[0],centerLine[1]),Point(centerLine[2],centerLine[3]),Scalar(255,255,0),15);
}

//Find center given the two lanes.
void LaneDetector::find_center(){
    float x = centerLine[2];
    float y = centerLine[3];
    center = Point(x,y);
    circle(lineImg,center,15,Scalar(0,0,255),-1);
}

// Displays detected lanes unto a given image
// @param frame to draw lanes unto.
void LaneDetector::display(Mat cameraFrame){

    circle(lineImg,predictedCenter,15,Scalar(0,255,0),-1);
    // Blend the lineImg of detected frame with camera feed for live visualization
    addWeighted(cameraFrame,1,lineImg,0.4,0,cameraFrame);
    namedWindow("Lane Detector");
    imshow("Lane Detector", cameraFrame);
    
}

#endif
