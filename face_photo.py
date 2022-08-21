from fileinput import filename
import cv2
import os

imagePath = os.path.abspath("images/team.jpg")
cascPath = os.path.abspath("haarcascade_frontalface_default.xml")

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

filename = 'myconvertedimage.jpg'

cv2.imwrite(filename, image)
