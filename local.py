import cv2
import numpy as np
import face_recognition

imgsharukh = face_recognition.load_image_file('Imageslocal/kd.jpg')
# convert images into rgb
imgsharukh = cv2.cvtColor(imgsharukh,cv2.COLOR_BGR2RGB)

imgsharukhtest = face_recognition.load_image_file('Imageslocal/kd3.jpg')
# convert images into rgb
imgsharukhtest = cv2.cvtColor(imgsharukhtest,cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgsharukh)[0]
encodeSharukh = face_recognition.face_encodings(imgsharukh)[0]
cv2.rectangle(imgsharukh,(faceLoc[3], faceLoc[0], faceLoc[1], faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgsharukhtest)[0]
encodeTest = face_recognition.face_encodings(imgsharukhtest)[0]
cv2.rectangle(imgsharukhtest,(faceLocTest[3],faceLocTest[0], faceLocTest[1], faceLocTest[2]), (255,0,255),2)

results = face_recognition.compare_faces([encodeSharukh], encodeTest)
facedis = face_recognition.face_distance([encodeSharukh], encodeTest)

print(results, facedis)
cv2.putText(imgsharukhtest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Sharukh Khan', imgsharukh)
cv2.imshow('Sharukh Test', imgsharukhtest)

cv2.waitKey(0)

