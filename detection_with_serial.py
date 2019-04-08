import numpy as np
import cv2 as cv
import serial
ser = serial.Serial('/dev/ttyUSB1')#replace this with COM1 from arduino IDE

face_cas = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cas = cv.CascadeClassifier('haarcascade_eye.xml')

cap = cv.VideoCapture(0)

while True:
 try: 
  ret , img = cap.read()
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  faces = face_cas.detectMultiScale(gray, 1.5, 5)
  for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(230,90,61),2)
    ser.write(b'face')
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyes = eye_cas.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
      cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(61,61,230),2)
      ser.write(b'eye')
  cv.imshow('img',img)
  k= cv.waitKey(30) & 0xff
  if k == 27:
   break
 except KeyboardInterrupt:
  break
cap.release()
ser.close()
cv.destroyAllWindows()
