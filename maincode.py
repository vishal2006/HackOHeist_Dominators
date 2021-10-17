# importing required libraries
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
import tkinter
import glob
from tkinter import *
from PIL import ImageTk, Image
import ctypes
from tkinter import messagebox
import pygame
pygame.mixer.init()
pygame.mixer.music.load("M416.wav")
import RPi.GPIO as GPIO
GPIO.setwarnings(False)

#----------------------------------------------------------------------------------------------------------------

### Initialized variables
cap = cv2.VideoCapture(0)


a = 6
b = 13
c = 19
d = 26

GPIO.setmode(GPIO.BCM)

GPIO.setup(a,GPIO.OUT)
GPIO.setup(b,GPIO.OUT)
GPIO.setup(c,GPIO.OUT)
GPIO.setup(d,GPIO.OUT)

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

input_mean = 127.5
input_std = 127.5

frame_rate_calc = 1
freq = cv2.getTickFrequency()


#----------------------------------------------------------------------------------------------------------------
# Functions

def login():
    if entry1.get()=="" or entry2.get()=="":
        messagebox.showerror("Error","All fields are required",parent=root)
    
    elif entry1.get()!="Dominators" or entry2.get()!="123456":
        messagebox.showerror("Error","Invalid Username/Password",parent=root)
    
    else:
        root.destroy()
        detect()




def detect():

    while(cap.isOpened()):

        ret, frame1 = cap.read()
        if ret == True:

            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)
            
            if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            body_cascade = cv2.CascadeClassifier('cascade.xml')

            boxes = interpreter.get_tensor(output_details[0]['index'])[0] 
            classes = interpreter.get_tensor(output_details[1]['index'])[0] 
            scores = interpreter.get_tensor(output_details[2]['index'])[0]

            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                        
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                        if (int(classes[0]) == 0.0):
                            pygame.mixer.music.play()
                            object_name = labels[int(classes[i])] 
                            label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
                            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                            label_ymin = max(ymin, labelSize[1] + 10) 
                            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
                            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 

                            xc = int(xmin+((xmax-xmin)/2))
                            yc = int(ymin+((ymax-ymin)/2))
                            cv2.circle(frame,(xc,yc), 10, (0,0,180), -1)

                            fxc, fyc = (frame.shape[1])/2,(frame.shape[0])/2
                            disx, disy = (xc - fxc),(yc-fyc)
                            
                            print("disx: ",disx,"disy: ",disy)
                            print("xc: ",xc,"yc: ",yc)
                            
                            if disx  >= 1:
                                GPIO.output(a,GPIO.LOW)
                                GPIO.output(b,GPIO.HIGH)
                                GPIO.output(c,GPIO.LOW)
                                GPIO.output(d,GPIO.LOW)
                                time.sleep(2)
                                
                            elif disx  <0:
                                GPIO.output(a,GPIO.HIGH)
                                GPIO.output(b,GPIO.LOW)
                                GPIO.output(c,GPIO.LOW)
                                GPIO.output(d,GPIO.LOW)
                                time.sleep(2)
                            
                            if disy >= 1:
                                GPIO.output(a,GPIO.LOW)
                                GPIO.output(b,GPIO.LOW)
                                GPIO.output(c,GPIO.HIGH)
                                GPIO.output(d,GPIO.LOW)
                                time.sleep(2)
                                
                            elif disy <0:
                                GPIO.output(a,GPIO.HIGH)
                                GPIO.output(b,GPIO.HIGH)
                                GPIO.output(c,GPIO.LOW)
                                GPIO.output(d,GPIO.LOW)
                                time.sleep(2)
                                
                            if disx <=5 and disy <=5:
                                GPIO.output(a,GPIO.LOW)
                                GPIO.output(b,GPIO.LOW)
                                GPIO.output(c,GPIO.LOW)
                                GPIO.output(d,GPIO.LOW)
                        
                            

            
            frame2 = frame.copy()
            frame_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (1280,720))
            input_data = np.expand_dims(frame_resized, axis=0)
            bgui = cv2.imread("bgui.jpg")
            image1 = cv2.resize(bgui, (1600, 900))
            framecamera = cv2.resize(frame2, (878, 456))
            
            image1[130:130+framecamera.shape[0],120:120+framecamera.shape[1]] = framecamera
            
            cv2.putText(image1,"State:",(780,760),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image1,"Normal",(780,820),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image1,"Soldier",(200,780),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image1,"Objects Detected:",(1200,130),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image1,"GPS Coordiantes:",(1160,770),cv2.FONT_HERSHEY_TRIPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image1,"19.0759899 72.8773928",(1160,800),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            body = body_cascade.detectMultiScale(gray)

            if body:
                cv2.putText(image1,"Rest",(200,820),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),1,cv2.LINE_AA)

            for (x,y,w,h) in body:
                cv2.rectangle(image1,(x,y),(x+w,y+h),(255,0,0),2)

            for j in range(len(scores)):
                if ((scores[j] > min_conf_threshold) and (scores[j] <= 1.0)):
                    cv2.putText(image1,labels[int(classes[j])],(1280,(190+(j*40))),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2,cv2.LINE_AA)
                    
            image1 = cv2.resize(image1, (1200, 700))

            
            cv2.imshow('Survelliance System', image1)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break
#------------------------------------------------------------------------------------------------------------------

### Main


root = Tk()
root.title("Surveillance System")
root.geometry("1200x675")
root.minsize(1200,675)
root.maxsize(1200,675)

bg = ImageTk.PhotoImage(file="bg.png")


my_canvas = Canvas(root, width=1200, height=675)
my_canvas.pack(fill="both",expand=True)

my_canvas.create_image(0,0, image=bg,anchor="nw")

text1 = my_canvas.create_text(700,100,fill="RoyalBlue1",font=("Segoe UI", 45, "bold"),text="Login Page")
text2 = my_canvas.create_text(650,280,fill="white",font=("Segoe UI", 15, "bold"),text="Username")
test3 = my_canvas.create_text(650,380,fill="white",font=("Segoe UI", 15, "bold"),text="Password")

 
entry1 = Entry(root,font=("Calibri 18 bold"),justify="center",bg="white",fg="black",borderwidth=2,width=25)
entry2 = Entry(root,font=("Calibri 18 bold"),justify="center",bg="white",fg="black",borderwidth=2,width=25,show="*")

my_button = Button(root,text="LOGIN",font=("Segoe UI", 18," bold"),command=login,width=10,fg="white",bg="RoyalBlue1",relief=GROOVE,activebackground="lightblue")

window1 = my_canvas.create_window(650,325,window=entry1)
window2 = my_canvas.create_window(650,400,window=entry2)

my_button.place(x=690,y=500)
entry1.place(x=600,y=300)
entry2.place(x=600,y=400)

root.mainloop()

cv2.destroyAllWindows()
cap.release()
GPIO.output(a,GPIO.LOW)
GPIO.output(b,GPIO.HIGH)
GPIO.output(c,GPIO.HIGH)
GPIO.output(d,GPIO.LOW)
GPIO.cleanup()



