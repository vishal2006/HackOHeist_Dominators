# importing required libraries
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util

#----------------------------------------------------------------------------------------------------------------

### Initialized variables

cap = cv2.VideoCapture(0)

### Arguement parsing

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir',help='Enter the path of the DL model',required=True)
parser.add_argument('--graph',help='Enter model name',default='detect.tflite')
parser.add_argument('--labels',help='Enter the name of label',default='labelmap.txt')
parser.add_argument('--threshold',help='Minimum threshold parameter',default=0.5)
parser.add_argument('--resolution',help='Resolution of webcam',default='1280x720')
parser.add_argument('--edgetpu',help='',action='store_true')



#----------------------------------------------------------------------------------------------------------------
# Functions

# creating class for capturing images from video


#------------------------------------------------------------------------------------------------------------------

while(cap.isOpened()):

  ret, frame = cap.read()
  if ret == True:


    cv2.imshow('Frame',frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

cap.release()
cv2.destroyAllWindows()



