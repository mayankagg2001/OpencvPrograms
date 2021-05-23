# Apply moustache and sunglasses filter to a person face  

import cv2
import numpy as np

img = cv2.imread('Jamie_Before.jpg')
img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

glasses = cv2.imread('glasses.png',-1)
mustache = cv2.imread('mustache.png',-1)

def overlay_img(color_face,face,img,overlay_img):
    (x,y,w,h) = face
    alpha = overlay_img
    inverse_alpha = 1-overlay_img
    y1 = y
    y2 = y+img.shape[0]
    x1 = x
    x2 = x+img.shape[1]

    xo1 = 0
    xo2 = img.shape[1]

    yo1 = 0
    yo2 = img.shape[0]

    # print(alpha.shape)
    # print(img[yo1:yo2,xo1:xo2,1].shape)
    # print(color_face[y1:y2,x1:x2,1].shape)

    for i in range(3):
        color_face[y1:y2,x1:x2,i] = alpha*img[yo1:yo2,xo1:xo2,i] + inverse_alpha*color_face[y1:y2,x1:x2,i]
    return color_face

faces = face_cascade.detectMultiScale(gray,1.2,6)
for (x,y,w,h) in faces:
    
    color_face = img[y:y+h,x:x+w]
    gray_face = gray[y:y+h,x:x+w]
    
    eyes = eye_cascade.detectMultiScale(gray_face,1.1,6)
    nose = nose_cascade.detectMultiScale(gray_face,1.1,8)
    for (x1,y1,w1,h1) in eyes:
        # cv2.rectangle(color_face,(x1,y1),(x1+w1,y1+h1),(0,255,0),4)
        glasses2 = cv2.resize(glasses.copy(),(w1,h1))
        color_face = overlay_img(color_face,(x1,y1,w1,h1),glasses2[:,:,0:3],glasses2[:,:,3]/255.0)
        break
    for (x1,y1,w1,h1) in nose:
        # cv2.rectangle(color_face,(x1,y1),(x1+w1,y1+h1),(0,255,0),4)
        mustache2 = cv2.resize(mustache.copy(),(w1,h1))
        # print(mustache.shape)
        # mustache2 = mustache
        color_face = overlay_img(color_face,(x1+1,y1+int(h1/2),w1,h1),mustache2[:,:,0:3],mustache2[:,:,3]/255.0)
        break
    

    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
    

cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
