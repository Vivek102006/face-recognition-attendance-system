import cv2
import numpy as np
import os

name=input("Enter your name: ")
path = f"dataset/{name}"

if not os.path.exists(path):
    os.makedirs(path)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(r'C:\Users\Vivek Kumar\Documents\face_recognization\haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

count = 0

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    

    for (x, y, w, h) in faces:
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = gray[y:y + h, x:x + w]
        cv2.imwrite(f"{path}/{name}_{count}.jpg", face_img)

    cv2.imshow('Face Recognization', frame)

    if cv2.waitKey(30) & 0xFF == ord('q') or count >= 100:
        break
cap.release()
cv2.destroyAllWindows()