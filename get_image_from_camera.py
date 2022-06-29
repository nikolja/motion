import cv2
import sys

video_capture = cv2.VideoCapture(0)

#video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
#video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

ret, image = video_capture.read()
cv2.imwrite(sys.argv[1], image)

video_capture.release()