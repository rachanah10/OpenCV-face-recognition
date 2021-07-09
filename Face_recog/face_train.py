import os
import cv2 as cv
import numpy as np

#get list of all labels
p=[]
for i in os.listdir(r'C:\Users\Acer\OneDrive\Desktop\python_demo\photos\train'):
    p.append(i)

print(p)
DIR=r'C:\Users\Acer\OneDrive\Desktop\python_demo\photos\train'

features= []
labels= []
haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for person in p:
        path = os.path.join(DIR, person)
        label = p.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array= cv.imread(img_path)
            if img_array is None:
                continue

            gray= cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) #use haar cascade

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]  #to crop out the face
                features.append(faces_roi)      #adds to the list of face images
                labels.append(label)            #adds to the names

create_train()
print("training done---------------------------------------------")
#creates a face list using haar cascades(above code)

#convert list to np array
features = np.array(features, dtype='object')
labels = np.array(labels)
#training the model using LBP module
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
