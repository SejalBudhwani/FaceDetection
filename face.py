import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_cascade.xml')
people = ['benaffleck', 'jen', 'mindy']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('index.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

face_detect  =haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for x,y,w,h in face_detect:
    face_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    print(people[label])
    print(confidence)
    
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2 )
    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
    
cv.imshow('detected', img)
cv.waitKey(0)