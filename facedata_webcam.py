import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

face_data = []

dataset = './data/'

location = input("Enter location to save file ")

skip = 0

count = 0

while True:
    ret,frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if not ret:
        continue
    

    faces = face_cascade.detectMultiScale(grey,1.3,5)
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    offset = 10

    
    if(len(faces) !=0):
        (x,y,w,h) = faces[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,126,120),4)
        if skip%10 == 0:
            data = grey[x-offset:x+w+offset,y-offset:y+h+offset]
            data = cv2.resize(data,(100,100))
            face_data.append(data)
            print(len(face_data))
        skip+=1
    cv2.imshow("frame",frame)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_data = np.asarray(face_data) 

face_data = face_data.reshape((face_data.shape[0], -1))


np.save(dataset+location+'.npy',face_data)