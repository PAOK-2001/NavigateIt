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
        Mat structuring_element = getStructuringElement(MORPH_ELLIPSE,Size(3,3));
        // Lane lines in the format of a 4 integer vector such as  (x1, y1, x2, y2), which are the endpoint coordinates. 
        Vec4i centerLine;
        // Coordinate that represents the center of given lane
        Point2f center, predictedCenter;
        // For detecting intersections
        float curAvgSlope;
        float avgAvgSlope = 0;
        float varianceAvgSlope = 0;
        int frameCount = 0;
        //Frame dimensions
        int width, height;
        // Auxiliary functions
        static float average(vector<float> input_vector);

    public:
        bool intersectionLikely = false;
        LaneDetector();
        LaneDetector(Mat initFrame);
        float getCenterx();
        int getWidth();
        int getHeight();
        void init_kf();
        void load_frame(Mat cameraFrame);
        void find_lanes();
        void predict_center();
        void display(Mat cameraFrame);
};

LaneDetector::LaneDetector(){
    init_kf();
}

LaneDetector::LaneDetector(Mat initFrame){
    width = initFrame.cols;
    height = initFrame.rows;
    init_kf();
}

float LaneDetector::getCenterx(){
    return predictedCenter.x;
}

int LaneDetector::getWidth(){
    return width;
}

int LaneDetector::getHeight(){
    return height;
}

void LaneDetector::init_kf(){
    center_estimator = KalmanFilter(2,2);
    setIdentity(center_estimator.measurementMatrix);
    setIdentity(center_estimator.processNoiseCov, Scalar::all(1e-5));
    setIdentity(center_estimator.measurementNoiseCov, Scalar::all(1e-5));
    setIdentity(center_estimator.errorCovPost, Scalar::all(1));
}

void LaneDetector::predict_center(){
    cv::Mat measurement(2, 1, CV_32F);
    measurement.at<float>(0) = center.x - width/2;
    measurement.at<float>(1) = center.y - height/2;
    center_estimator.correct(measurement);
    Mat predictedState = center_estimator.predict();
    predictedCenter = Point2f(predictedState.at<float>(0) + width/2, predictedState.at<float>(1) + height/2);
}

float LaneDetector::average(vector<float> input_vector){
    float size, sum;
    size = input_vector.size();
    if(size == 0){
        return 0;
    }
    for(auto & num: input_vector){
        sum+=num;
    }
    float avrg = sum/(float)size;
    return avrg;
}

void LaneDetector::load_frame(Mat cameraFrame){
    Mat grayScale, thresholded;
    cvtColor(cameraFrame,grayScale,COLOR_BGR2GRAY);
    threshold(grayScale, thresholded, 75, 255, THRESH_BINARY_INV);
    morphologyEx(thresholded,thresholded,MORPH_OPEN,structuring_element);
    // imshow("Thresholded", thresholded);
    // waitKey(1);
    Canny(thresholded,edgeImg,40,150);
}

void LaneDetector::find_lanes(){
    Mat mask = edgeImg.clone();
    mask     = Scalar(0,0,0);
    lineImg  = mask;
    cvtColor(lineImg,lineImg,COLOR_GRAY2BGR);
    vector<Point> ROI = {
        Point(330,719),
        Point(330,669),
        Point(909,669),
        Point(909,719)
    };
    fillPoly(mask,ROI,(255,255,255));
    // imshow("Mask", mask);
    // waitKey(1);
    // imshow("Edge", edgeImg);
    // waitKey(1);
    bitwise_and(mask,edgeImg,edgeImg);
    vector<Vec4i> lines;
    HoughLinesP(edgeImg,lines,2,CV_PI/180,4,10,5);
    vector<float> x;
    curAvgSlope = 0;
    for(int i=0; i<lines.size(); i++) {
        auto &lineP = lines[i];
        float curSlope = (lineP[3] - lineP[1]) / (lineP[2] - lineP[0]); 
        curAvgSlope = (curAvgSlope + curSlope)/ (i+1);
        x.push_back(lineP[0]);
        x.push_back(lineP[2]);
        line(lineImg,Point(lineP[0],lineP[1]),Point(lineP[2],lineP[3]),Scalar(120,120,0),10);
    }
    frameCount++;
    avgAvgSlope = (avgAvgSlope + curAvgSlope) / frameCount;
    float delta = avgAvgSlope - curAvgSlope;
    varianceAvgSlope = (varianceAvgSlope + pow(delta,2)) / frameCount;

    if (abs(delta) > sqrt(varianceAvgSlope)*3) intersectionLikely = true; // check for deviation greater than 3 sigmas (0.3% chance of appearing in normal distribution)
    else intersectionLikely = false;

    if(x.size() == 0){
        center = Point2f(round(lineImg.cols/2),round(lineImg.rows*(1.0/1.2)));
    }else{
        float average_x = average(x);
        center = Point2f(round(average_x),round(lineImg.rows*(1.0/1.2)));
    }
    circle(lineImg,center,15,Scalar(0,0,255),-1);
}

void LaneDetector::display(Mat cameraFrame){
    circle(lineImg,predictedCenter,15,Scalar(0,255,0),-1);
    line(lineImg,Point(640,0),Point(640,719),Scalar(0,120,120),10);
    addWeighted(cameraFrame,1,lineImg,0.4,0,cameraFrame);
    //overlay_writer.write(cameraFrame);
    // namedWindow("Lane Detector");
    // imshow("Lane Detector", cameraFrame);
}

#endif
