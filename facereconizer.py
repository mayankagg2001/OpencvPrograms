import cv2
import numpy as np
import os
dataset_path = './data/'

def dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


def knn(data,x,n_neighors = 5):
    X_data = data[:,:-1]
    Y_data = data[:,-1]
    vals = []
    m = X_data.shape[0]
    for i in range(m):
        d = dist(X_data[i],x)
        vals.append((d,Y_data[i]))
    vals = sorted(vals)
    vals = np.array(vals)
    x = np.unique(vals,return_counts=True)
    index = x[1].argmax()
    return x[0][index]

skip = 0
face_data = []
labels = []

classid = 0

for name in os.listdir(dataset_path):
    if name.endswith('.npy'):

        data_item = np.load(dataset_path+name)
        face_data.append(data_item)

        target = classid*np.ones((data_item.shape[0],1))
        classid+=1
        labels.append(target)

face_data = np.concatenate(face_data,axis = 0)
labels = np.concatenate(labels,axis=0)

data = np.concatenate((face_data,labels),axis=1)

print(data.shape)
print(face_data.shape)
print(labels.shape)

# Testing

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        continue
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey,1.3,5)
    height = int(cap.get(4))
    if(len(faces) == 0):
         continue

    
    for (x,y,w,h) in faces:
        
        
        font = cv2.FONT_HERSHEY_PLAIN
        face = grey[x-10:x+w+10,y-10:y+h+10]
        face = cv2.resize(face,(100,100))
        face = face.flatten()
        output  = knn(data,face,5)
        frame = cv2.putText(frame,str(output),(10,height-10),font,4,(0,0,0),5,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame',frame)
    if(cv2.waitKey(10) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()


