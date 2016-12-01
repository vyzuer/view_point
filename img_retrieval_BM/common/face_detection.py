import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def detect(img, dump_path="./", dump_result = False, img_show = False, mark_faces = False, img_src = None):

    face_cascade = cv2.CascadeClassifier('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-python/common/FaceCascade/haarcascade_frontalface_default.xml')
    
    # img = cv2.imread(img_src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if mark_faces == True:
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
    
    if img_show == True:
        plt.imshow(img)
        plt.show()
        plt.close()

    if dump_result == True:
        dir_name, img_name = os.path.split(img_src)
        file_name = dump_path + img_name
        cv2.imwrite(file_name, img)

    return faces

