import cv2
cameraGet = cv2.VideoCapture(0)
cameraGet.set(3, 640)
cameraGet.set(4, 480)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_id = input('\n введите id базы фото ==>   ')
print('\n [INFO] съемка фото, смотрите в камеру и ждите ')
count = 0
while True:
    ret,img = cameraGet.read()

    gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count+=1
        cv2.imwrite("dataset/User." + str(face_id) + '.'+ str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count>=30:
        break


print("\n [INFO] Процесс завершен")
cameraGet.release()
cv2.destroyAllWindows()