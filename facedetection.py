import cv2 as cv
import numpy as np

img = cv.imread('lost.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blank = np.zeros(img.shape[:2], dtype = 'uint8')
haar_cascade = cv.CascadeClassifier('haar_cascade.xml')
eye = cv.CascadeClassifier('eye.xml')
face_det = haar_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors= 3)
i = 0
for (x,y,w,h) in face_det:
    i = i +1
    rectangle = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
    roi_color = img[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]
    eye_detect = eye.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=12)
    for (ex,ey,ew,eh) in eye_detect:
        cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 1)
        mask = cv.bitwise_and(img, img, mask = rectangle)
    r = cv.resize(roi_color, (500,500), interpolation= cv.INTER_LINEAR)
    cv.imwrite('%s.JPEG' % (i), r)
    cv.waitKey(0)
    
    print(i)
cv.imshow('img',img)
cv.waitKey(0)