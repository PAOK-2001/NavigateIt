# import the opencv library
import cv2 as cv
import numpy as np
import rospy
from std_msgs.msg import String

class ColorDetector:
    
    def __init__(self):
        #self.color_pub = rospy.Publisher("/traffic_light", String, queue_size=1)
        self.min_pixel_count = 2500
        self.l_lower_red = np.array([0,190,100])
        self.l_upper_red = np.array([10,255,255])
        self.h_lower_red = np.array([170,150,100])
        self.h_upper_red = np.array([180,255,255])
        self.yellow_thresh_down = np.array([22, 90, 50])
        self.yellow_thresh_up = np.array([45, 255, 255])
        self.green_thresh_down = np.array([57, 190, 100])
        self.green_thresh_up = np.array([113, 255, 255])

    def detect_color(self, frame):
        names = ["green","yellow","red"]
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask_green = cv.inRange(frame_hsv, self.green_thresh_down, self.green_thresh_up)
        mask_yellow = cv.inRange(frame_hsv, self.yellow_thresh_down, self.yellow_thresh_up)
        mask_red = cv.inRange(frame_hsv, self.l_lower_red, self.l_upper_red) + cv.inRange(frame_hsv, self.h_lower_red, self.h_upper_red)
        cv.imshow("Red Mask", mask_red)
        cv.imshow("Yellow Mask", mask_yellow)
        cv.imshow("Green Mask", mask_green)
        pixels = [np.sum(mask_green == 255),np.sum(mask_yellow == 255),np.sum(mask_red == 255)]    
        max_count = max(pixels)
        if max_count >= self.min_pixel_count:
            return names[pixels.index(max_count)]
        else:
            return "None"

if __name__ == "__main__":
    detector = ColorDetector()
    #rospy.init_node("traffic_monitor")
    camera = cv.VideoCapture(0)  
    while True: 
        ret, frame = camera.read()
        cv.imshow('frame', frame)
        color = detector.detect_color(frame)
        print(color)
        #`detector.color_pub.publish(color)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()
    