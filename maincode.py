# importing required libraries
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread

#----------------------------------------------------------------------------------------------------------------

## Initialized variables

cap = cv2.VideoCapture(0)

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



