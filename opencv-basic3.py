# To read video using opencv by reading images at different

import numpy as np
import random
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    image = np.zeros(frame.shape , np.uint8)
    smaller_frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    image[:smaller_frame.shape[0],:smaller_frame.shape[1]] = smaller_frame
    image[smaller_frame.shape[0]:,smaller_frame.shape[1]:] = cv2.rotate(smaller_frame,cv2.cv2.ROTATE_180)
    image[smaller_frame.shape[0]:,:smaller_frame.shape[1]] = smaller_frame
    image[:smaller_frame.shape[0],smaller_frame.shape[1]:] = cv2.rotate(smaller_frame,cv2.cv2.ROTATE_180)
    cv2.imshow('frame',image)

    if cv2.waitKey(1) == ord('q'):                  #waits 1 ms and if q is 

        break                                     #pressed in that time interval it breaks else it continues the loop every 1 ms
cap.release()
cv2.destroyAllWindows()