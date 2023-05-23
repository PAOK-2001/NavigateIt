import cv2
import jetson.utils
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#ros_bridge = CvBridge()

camera = jetson.utils.videoSource("csi://0", ['--input-width=1280','--input-height=720'])
    #video_publisher = rospy.Publisher("/dash_cam",Image,queue_size=10)
    #rospy.init_node("video_source")
    #rate = rospy.rate(10)

while True:
        img = camera.Capture()
        imgbgr = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format="bgr8")
        jetson.utils.cudaConvertColor(img, imgbgr)
        cv_img = jetson.utils.cudaToNumpy(imgbgr)
        cv2.imshow("Output", cv_img)
        cv2.waitKey(1)

