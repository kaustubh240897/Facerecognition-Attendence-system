import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyttsx3


engine = pyttsx3.init()

path = 'ImagesAttendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for li in myList:
    curImg = cv2.imread(f'{path}/{li}')
    images.append(curImg)
    classNames.append(os.path.splitext(li)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtvalueString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtvalueString}')




encodedListKnownFaces = findEncodings(images)
print('encoding completed !!')

# Taking images from camera

capture = cv2.VideoCapture(0)

while(True):
    success, img1 = capture.read()
    # reducing size of images because we are capturing live images
    imgS = cv2.resize(img1,(0,0), None, 0.5,0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodedListKnownFaces, encodeFace)
        faceDis = face_recognition.face_distance(encodedListKnownFaces, encodeFace)
        print(faceDis)
        # we will match to face which have min face distance 
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
            cv2.rectangle(img1,(x1,y1), (x2,y2),(0,255,0),2)
            cv2.rectangle(img1,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img1,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)
            engine.say(f'Thank you {name}')
            engine.setProperty('rate',120)
            engine.setProperty('volume',0.9)
            engine.runAndWait()



    cv2.imshow('WebCam', img1)
    cv2.waitKey(1)





     




