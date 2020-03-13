# Follow Frederik's github guide, but use sudo when running command: sudo pip3 install -r requirements.txt (must be run inside virtual environment


import numpy as np
import argparse
import time
import cv2

# Simple test for NeoPixels on Raspberry Pi
import time
import board
import neopixel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os

#GLOBAL VARIABLES
distance = 50



# input arg parsing
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fullscreen',
                    help='Display window in full screen', action='store_true')
parser.add_argument(
    '-d', '--debug', help='Display debug info', action='store_true')
args = parser.parse_args()

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)


def draw_border(img, pt1, pt2, color=(255, 0, 0), thickness=8, r=15, d=10):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# start the webcam feed
cap0 = cv2.VideoCapture(0)
#cap0.set(5,20)
cap0.set(3, 256)
cap0.set(4, 144)

cap1 = cv2.VideoCapture(2)
#cap1.set(5,20)
cap1.set(3, 256)
cap1.set(4, 144)



while True:
    # time for fps
    start_time = time.time()

    ret_val_0, img_0 = cap0.read()
    ret_val_1, img_1 = cap1.read()
    
    
    img_0 = cv2.flip(img_0,1)
    img_1 = cv2.flip(img_1,1)
    
    #img_0 = cv2.resize(img_0, (img_1.shape[1], img_1.shape[0]))
    # only necessary if cams are not of identical size
    
    frame = np.concatenate((img_0, img_1), axis=1)
    
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(distance,distance)) 
    
    

    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        draw_border(frame, (x, y-50), (x+w, y+h+10))
    
    
    # debug info
    if args.debug:

        cv2.imshow('my webcam', frame)
        #cv2.imshow('video', cv2.resize(frame, (800, 480), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    fps = str(int(1.0 / (time.time() - start_time)))
    print("fps = " + str(fps))
        
    

cv2.destroyAllWindows()

