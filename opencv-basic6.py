# Edge detection using opencv

import cv2
import numpy as np
import random

img = cv2.imread("chessboard.png")
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(img2,100,0.01,1)
for corner in corners:
    x,y = np.uint0(np.ravel(corner))
    cv2.circle(img,(x,y),4,(255,0,0),-1)

for i in range(len(corners)):
    for j in range(i+1,len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(lambda x: int (x),np.random.randint(0,255,3)))
        cv2.line(img,corner1,corner2,color,1)

cv2.imshow("frame",img)
# cv2.imshow("frame2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()