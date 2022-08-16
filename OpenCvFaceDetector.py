import cv2
import numpy as np
import os
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

name = [] # напишите свои id узанные в OpenCvUserData

cameraGet = cv2.VideoCapture(0)
cameraGet.set(3, 640)
cameraGet.set(4, 480)
minW = 0.1 * cameraGet(3)
minH = 0.1 * cameraGet(4)

while True:
    ret,img = cameraGet.red()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 19)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id,confidence = recognizer.predict(gary[y:y+h,x:x +w])
        if (confidence <100):
            id = names[id]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "{0}%".format(round(100 - confidence))
        cv2.putText(img,str(id),(x + 5, y - 5),font,1,(255,255,255),2)
        cv2.putText(img,str(confidence),(x + 5, y - 5),font,1,(255,255,0),1)
    cv2.imshow('camera',img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
print("\n [INFO] Программ завршина")
cameraGet.release()
cv2.destoyAllWindows()