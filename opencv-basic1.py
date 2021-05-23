
# Basic opencv :-
# 1) Read an image using opencv
# 2) Resize the image using opencv
# 3) Rotate the image using opencv

import numpy as np
import cv2

img = cv2.imread("download.jpg",-1)      #cv2.imread("download.jpg",1)  cv2.imread("download.jpg",0)
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
img = cv2.rotate(img,cv2.cv2.ROTATE_180)
cv2.imshow('Image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()