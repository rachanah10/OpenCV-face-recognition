import cv2 as cv
import numpy as np

img=cv.VideoCapture(0)
haar_cascade= cv.CascadeClassifier('haar_face.xml')

while True:
    isTrue, frame= img.read()
    # cv.imshow('vid', frame)
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

    # print(f'Number of faces found = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
         cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=4)
         cv.putText(frame, "Face", (x,y-10), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), thickness=2)

    cv.imshow('Detected Faces', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break


img.release()
cv.destroyAllWindows()
