import cv2
import os
import numpy as np
from datetime import datetime

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load images
path = "dataset"
images = []
classNames = []

for file in os.listdir(path):
    img = cv2.imread(f"{path}/{file}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        images.append(face)
        classNames.append(os.path.splitext(file)[0])

# Create face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train recognizer
faces = []
labels = []

for i, img in enumerate(images):
    faces.append(img)
    labels.append(i)

recognizer.train(faces, np.array(labels))

# Face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Attendance function
def markAttendance(name):
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("Name,Time\n")

    with open("attendance.csv", "r+") as f:
        data = f.readlines()
        nameList = [line.split(",")[0] for line in data]

        if name not in nameList:
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            f.write(f"{name},{time}\n")

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 90:
            name = classNames[label].upper()
            markAttendance(name)

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Attendance System", img)

    if cv2.waitKey(1) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()