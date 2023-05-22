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
        Vec4i leftLine, rightLine;
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
    setIdentity(center_estimator.measurementNoiseCov, Scalar::all(1e-1));
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
    y2 = round(imgHeight*(1.0/1.43));
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
    // Image preprocessing code to load camera feed
    // Saturate image
    cameraFrame = cameraFrame*1.20;
    // Convert to grayscale
    Mat grayScale;
    cvtColor(cameraFrame,grayScale,COLOR_BGR2GRAY);
    // Canny edge detection
    Canny(grayScale,edgeImg,40,150);
}

void LaneDetector::find_lanes(){
    // Masking to exclude ROI
    Mat mask = edgeImg.clone();
    mask     = Scalar(0,0,0);
    // Create canvas for line image, converting to BGR to allow for color lanes. 
    lineImg  = mask;
    cvtColor(lineImg,lineImg,COLOR_GRAY2BGR);
    // Make mask using OpenCV poligon and aproximate coordinates based on lane width and camera FOV
    // Create points for polygon
    Point p1 = Point(10,mask.rows);
    Point p2 = Point(1120,650);
    Point p3  = Point(1740,mask.rows);
    vector<Point> ROI ={p1,p2,p3};
    fillPoly(mask,ROI,(255,255,255));
    // Exclude Region of Interest by combining mask using bitwise_and operator
    bitwise_and(mask,edgeImg,edgeImg);
    // Find all lines in frame using HoughLinesP
    vector<Vec4i> lines;
    // Uses HoughTransform to fine most lines in canny image.
    HoughLinesP(edgeImg,lines,2,CV_PI/180,200,40,5);
    // Find regression for lines
    vector<float> rightSide_slopes, rightSide_intercepts, leftSide_slopes,leftSide_intercepts;
    for(auto &lineP: lines){
        pair<float, float> fit =linear_fit(lineP);
        // Exclude outliers.
        if(fit.first == 0){
            continue;
        }
        // Seperate left lane (negative slope)
        else if(fit.first<0){
            leftSide_slopes.push_back(fit.first);
            leftSide_intercepts.push_back(fit.second);
        }
        // Separate right lane (positive slope)
        else{
            rightSide_slopes.push_back(fit.first);
            rightSide_intercepts.push_back(fit.second);
        }
    }
    // For each side, find average lane
    float rightSlope     = average_coefficient(rightSide_slopes);
    float rightIntercept = average_coefficient(rightSide_intercepts);
    float leftSlope      = average_coefficient(leftSide_slopes);
    float leftIntercept  = average_coefficient(leftSide_intercepts);
    // Make coordinates for final lanes
    leftLine  = make_coordinate(leftSlope, leftIntercept, height, width);
    rightLine = make_coordinate(rightSlope, rightIntercept, height, width);
    // Draw line on lineImg
    line(lineImg,Point(leftLine[0],leftLine[1]),Point(leftLine[2],leftLine[3]),Scalar(255,255,0),15);
    line(lineImg,Point(rightLine[0],rightLine[1]),Point(rightLine[2],rightLine[3]),Scalar(0,255,0),15);
}

//Find center given the two lanes.
void LaneDetector::find_center(){
    float x = leftLine[2]+((rightLine[2]-leftLine[2])/2.0);
    float y = (rightLine[3]);
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