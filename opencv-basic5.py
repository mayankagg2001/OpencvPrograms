# Program to change image in bgr format to different formats and to filter out some specefic color in image

import cv2
import random
import numpy as np

cap = cv2.VideoCapture(0)
print(cv2.cvtColor(np.uint8([[[255, 204, 204]]]),cv2.COLOR_RGB2HSV))
print(cv2.cvtColor(np.uint8([[[102, 0, 0]]]),cv2.COLOR_RGB2HSV))
while True:
    ret,frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #lightblue = cv2.cvtColor(np.uint8([[[255,238,0]]]),cv2.COLOR_BGR2HSV)
    #darkblue = cv2.cvtColor(np.uint8([[[255,0,0]]]),cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([130,255,255])
    

    # lower_red = np.array([])
    # upper_red = np.array([])

    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    result  = cv2.bitwise_and(frame,frame,mask = mask)


    #cv2.imshow("frame",frame)
    #cv2.imshow("frame2",hsv)
    #cv2.imshow("frame3",mask)
    cv2.imshow("frame4",result)
    
    if(cv2.waitKey(1)==ord('q')): break

cap.release()
cv2.destroyAllWindows()
