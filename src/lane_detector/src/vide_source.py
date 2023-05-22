import cv2
import jetson.utils


camera = jetson.utils.videoSource("csi://0", ['--input-width=1280','--input-height=720','--frameRate=59.999999','--flipMethod=rotate-181'])

while True:
        img = camera.Capture()
	imgbgr = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format="bgr8")
	jetson.utils.cudaConvertColor(img, imgbgr)
	cv_img = jetson.utils.cudaToNumpy(imgbgr)
        cv2.imshow("Output", cv_img)
        cv2.waitKey(1)
