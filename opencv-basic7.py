# Object detection in open cv (Detected soccer ball in the image)
import cv2
import random
import numpy as np

img = cv2.imread("soccer_practice.jpg",0)       
img = cv2.resize(img,(0,0),fx=0.8,fy=0.8)       #template size must be equal to image size 
template = cv2.imread("shoe.png",0)
template = cv2.resize(template,(0,0),fx=0.8,fy=0.8)

h,w = template.shape

methods = [cv2.TM_CCOEFF,cv2.TM_CCOEFF_NORMED,cv2.TM_CCORR,cv2.TM_CCORR_NORMED,cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()
    result = cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    bottom_right = (location[0]+w,location[1]+h)
    cv2.rectangle(img2,location,bottom_right,255,5)
    cv2.imshow('Match',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

