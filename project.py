#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pyfirmata import Arduino
import time


# In[2]:


class function():
    def camera(self):
        cap = cv2.VideoCapture(0)                       # 0번 카메라 연결
        if cap.isOpened() :
            while True:
                ret, frame = cap.read()                 # 카메라 프레임 읽기
                if ret:
                    cv2.imshow('camera',frame)          # 프레임 화면에 표시
                    if cv2.waitKey(1) != -1:            # 아무 키나 누르면
                        cv2.imwrite('/image\\photo.jpg', frame) # 프레임을 'photo.jpg'에 저장
                        break
                else:
                    print('no frame!')
                    break
        else:
            print('no camera!')
            
        cap.release()
        cv2.destroyAllWindows()

    def mask(self):
        global count_f
        global count_m
        facenet = cv2.dnn.readNet('/models\\deploy.prototxt', '/models\\res10_300x300_ssd_iter_140000.caffemodel')
        model = load_model('/models\\mask_detector.model')
        img = cv2.imread('/image\\photo.jpg')
        h, w = img.shape[:2]
    
        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        facenet.setInput(blob)
        dets = facenet.forward()
    
        faces = []

        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)
    
            face = img[y1:y2, x1:x2]
            faces.append(face)

        for i, face in enumerate(faces):
            face_input = cv2.resize(face, dsize=(224, 224))
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)
    
            mask, nomask = model.predict(face_input).squeeze()

        
            if mask < 0.95:
                count_f+=1
                return 0
            else:
                count_m+=1
                return 1


# In[3]:


if (os.path.isdir('/image') == False):
    os.mkdir('/image')
if (os.path.isdir('/image\\face') == False):
    os.mkdir('/image\\face')
if (os.path.isdir('/image\\mask') == False):
    os.mkdir('/image\\mask')


count_f = 0
count_m = 0

serial = Arduino("COM5")
time.sleep(2)

Relay=serial.get_pin('d:9:o')

Relay.write(1)

f = function()
while True:
    f.camera()
    readPC = f.mask()
        
    print('1번')
    print(readPC)
  
    if readPC == 1:
        Relay.write(0) # 문열림 NO를 연결 전기 흐름
        print('2번')
        print(readPC)
    else:
        Relay.write(1) # 문닫힘 NO 평상시 상태 전기 흐름X
        print('3번')
        print(readPC)
    
    serial.pass_time(10)  # 10초간 상태 유지
    print('RMX')
    print(readPC)
    Relay.write(1) # 문닫힘


# In[ ]:




