import cv2
import numpy as np
import random

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    width = int(cap.get(3))
    height = int (cap.get(4))

    img = cv2.line(frame,(0,0),(width,height),(255,255,255),5)
    img = cv2.circle(frame,(20,20),10,(0,0,255),-1)
    img = cv2.rectangle(frame,(10,20),(40,50),(255,123,210),-1)
    font = cv2.FONT_HERSHEY_PLAIN
    img = cv2.putText(img,'Mayank',(30,height-10),font,4,(0,0,0),5,cv2.LINE_AA)
    cv2.imshow('frame',frame)

    if(cv2.waitKey(1)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()