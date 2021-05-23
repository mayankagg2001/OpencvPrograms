
# Program to play with pixels

import cv2
import random
img = cv2.imread("download.jpg",-1)
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
print(img.shape)


# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         img[i][j] = [255,0,0]
#         #img[i][j] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]               ""Giving random colors""

# for i in range(img.shape[0]//2):
#     for j in range(img.shape[1]):                                                                    ""Copying Different parts of images to another part""
#         img[i][j] = img[i+img.shape[0]//2][j]

# copy = img[:img.shape[0]//2,:]
# img[img.shape[0]//2:,:] = copy


cv2.imshow("frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()