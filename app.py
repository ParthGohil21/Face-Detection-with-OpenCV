import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os


# Specify path for training images and get the class name of images
path=r'C:\Users\parth\OneDrive\Desktop\ML Projects\face recognition\Training Images'
Images=[]
ClassNames=[]
MyList=os.listdir(path)
print('Image Names :',MyList)

for img in MyList:
    CurrentImage=cv2.imread(f'{path}/{img}') #reading image
    Images.append(CurrentImage) 
    ClassNames.append(os.path.splitext(img)[0]) #Putting image name into Class Names list
    print('Class Names : ', ClassNames)
    


def FindEncodings(images):
    """This function will find face encoding of the given images"""

    EncodeList=[]

    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # print(img[0])
        encode=face_recognition.face_encodings(img)[0]
        EncodeList.append(encode)
    return EncodeList

Encode_List=FindEncodings(Images)
print('Encoding Finished') 


cap = cv2.VideoCapture(0)   
while True:
    ret, frame=cap.read()

    img=cv2.resize(frame, (0,0), None, 0.25, 0.25) 
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


    FacesFrame=face_recognition.face_locations(img) 
    EncodeFrame=face_recognition.face_encodings(img,FacesFrame)


    for EncodeFace, FaceLocation in zip(EncodeFrame, FacesFrame):
        Matches=face_recognition.compare_faces(Encode_List,EncodeFace)
        FaceDistance=face_recognition.face_distance(Encode_List,EncodeFace)


        MatchIndex=np.argmin(FaceDistance)


        if Matches[MatchIndex]:
            name=ClassNames[MatchIndex].upper()

            print(name)

            y1, x2, y2, x1 = FaceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)