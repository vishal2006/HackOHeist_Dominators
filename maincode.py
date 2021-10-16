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

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

### Tensorflow

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)



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



