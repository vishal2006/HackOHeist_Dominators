import tkinter
import glob
import cv2
from tkinter import *
from PIL import ImageTk, Image
import ctypes
from tkinter import messagebox
import numpy as np

def login():
    if entry1.get()=="" or entry2.get()=="":
        messagebox.showerror("Error","All fields are required",parent=root)
    
    elif entry1.get()!="Dominators" or entry2.get()!="123456":
        messagebox.showerror("Error","Invalid Username/Password",parent=root)
    
    else:
        call()
def call():
  root.destroy()
  detect()


def detect():
    cap=cv2.VideoCapture(0)
    while True:
        
        ret,frame1= cap.read()
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (1280,720))
        input_data = np.expand_dims(frame_resized, axis=0)
        bgui = cv2.imread("src\\bgui.jpg")
        image1 = cv2.resize(bgui, (1600, 900))
        framecamera = cv2.resize(frame, (878, 456))
        
        image1[130:130+framecamera.shape[0],120:120+framecamera.shape[1]] = framecamera
        
        cv2.putText(image1,"Mode:",(780,760),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2,cv2.LINE_AA)
        #cv2.putText(image1,"Normal",(780,820),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image1,"Direction:",(200,780),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2,cv2.LINE_AA)
        #cv2.putText(image1,mod,(200,820),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image1,"Objects Detected:",(1200,130),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image1,"GPS Coordinates:",(1160,770),cv2.FONT_HERSHEY_TRIPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image1,"21° 2' 27.528'' N 75° 3' 26.748'' E",(1160,800),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        image1 = cv2.resize(image1, (1200, 700))
        cv2.imshow('Survelliance System', image1)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
       
    cv2.destroyAllWindows()

        

root = Tk()
root.title("Surveillance System")
root.geometry("1200x675")
root.minsize(1200,675)
root.maxsize(1200,675)

bg = ImageTk.PhotoImage(file="src\\bg.png")


my_canvas = Canvas(root, width=1200, height=675)
my_canvas.pack(fill="both",expand=True)

my_canvas.create_image(0,0, image=bg,anchor="nw")

my_button = Button(root,text="LOGIN",font=("Segoe UI", 18," bold"),command=login,width=10,fg="white",bg="RoyalBlue1",relief=GROOVE,activebackground="lightblue")

text1 = my_canvas.create_text(700,100,fill="RoyalBlue1",font=("Segoe UI", 45, "bold"),text="Login Page")
text2 = my_canvas.create_text(650,280,fill="white",font=("Segoe UI", 15, "bold"),text="Username")
test3 = my_canvas.create_text(650,380,fill="white",font=("Segoe UI", 15, "bold"),text="Password")


entry1 = Entry(root,font=("Calibri 18 bold"),justify="center",bg="white",fg="black",borderwidth=2,width=25)
entry2 = Entry(root,font=("Calibri 18 bold"),justify="center",bg="white",fg="black",borderwidth=2,width=25,show="*")


window1 = my_canvas.create_window(650,325,window=entry1)
window2 = my_canvas.create_window(650,400,window=entry2)

my_button.place(x=690,y=500)
entry1.place(x=600,y=300)
entry2.place(x=600,y=400)

root.mainloop()