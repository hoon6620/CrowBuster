import tensorflow as tf
import numpy as np
import cv2
from PIL import Image,ImageTk
import tkinter as tk
from playmusic import PlayWavFie
import threading
import time
import glob
import os
import datetime
class Execute:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter("./weights/weights380.tflite")
        self.detected_data = []
        self.before_data = None
        self.preview_frame = None
        self.detect_class = np.empty(0)
        self.checker = None 
        self.exec_f = None
        self.use_camera = 1
        self.detect_name = None
        self.audio_time = time.time()-10
        self.logflag=False
        self.path=glob.glob(os.path.abspath('./sound/*.mp3'))
        
        self.thing=["nothing","crow","nothing"]

    def setCameraPreview(self,preview):
        self.preview = preview

    def set_exec_f(self,exec_f):
        self.exec_f = exec_f #
        
    def exec_model(self):
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        inputs = self.interpreter.tensor(input_details[0]['index'])
        output1 = self.interpreter.tensor(output_details[0]['index'])
        

        #print(input_details[0]['shape'])
        #print(output1)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_w = cap.get(3)
        cap_h = cap.get(4)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.logPath = '../CBLogs/' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + '/'
        print(self.logPath)
        os.makedirs(self.logPath)
        out = cv2.VideoWriter(self.logPath + "log_video.avi",
                              fourcc, 10.0, (640, 480))
        prev_time = 0
        FPS = 10
        
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            
            current_time = time.time() - prev_time            
            if(current_time < 1./FPS):
                continue
            
            if not ret:
                break
            #print("a")
            

            model_n,model_w,model_h,model_c = input_details[0]['shape']
            in_frame = cv2.resize(frame, (model_w, model_h))
            in_frame = in_frame.reshape((model_n, model_h, model_w, model_c))
            inputs().setfield(in_frame,dtype=np.float32)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            #print(output_data)
            
            if np.argmax(output_data) == 1:
                if output_data[1]/np.sum(output_data) > 0.45:
                    class_id=1
                else:
                    class_id=2
            else:
                class_id=np.argmax(output_data)
            if self.detect_class.__len__() < 10:
               
                self.detect_class = np.array([class_id for _ in range(10)])
            else:
                self.detect_class = np.append(self.detect_class[1:],class_id)
                #print(self.detect_class)
                self.exec_f.model.select_sound
            
            if self.detect_class.__len__() == 10:
                median_class_id = np.median(self.detect_class)
                self.detect_name=self.thing[int(median_class_id)]
                frame = cv2.putText(frame,self.detect_name,(0,int(cap_h//10)),cv2.FONT_HERSHEY_PLAIN,cap_h//130,(0,0,255),3)
                if self.logflag and time.time()-self.audio_time>5:
                    self.exec_f.txtbox.configure(state ='normal')
                    if self.detect_name==self.thing[1]:
                        self.logdata="Repulsion Failure\t"+self.logdata
                        self.exec_f.txtbox.insert(1.0,"Repulsion Failure\t")
                    else:
                        self.logdata="Repulsion Success\t"+self.logdata
                        self.exec_f.txtbox.insert(1.0,"Repulsion Success\t")
                        #print("")#escape crow log
                    self.logflag=False
                    self.exec_f.log=self.logdata+self.exec_f.log
                    self.exec_f.txtbox.configure(state ='disabled')
                    threading.Thread(target=self.writefile).start()
                    
                if self.detect_name==self.thing[1] and time.time()-self.audio_time>10:

                    self.logflag=True
                    if not self.exec_f.model.select_sound:
                        sound = np.random.choice(self.path)
                        #print(sound)
                    else:
                        sound = self.path[(int(np.random.choice(self.exec_f.model.select_sound)))]
                        #print(self.exec_f.model.select_sound)
                        #print(sound)
                    
                    #print(time.time()-self.audio_time)
                    PlayWavFie(sound) 
                    logstr = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+" "+sound.split("sound/")[1]
                    self.audio_time=time.time()
                    self.logdata = logstr + "\n"
                    self.exec_f.txtbox.configure(state ='normal')
                    self.exec_f.txtbox.insert(1.0,self.logdata)
                    self.exec_f.txtbox.configure(state ='disabled')
                    print(logstr)#sound log
                    cv2.imwrite(self.logPath+str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '.jpg', frame)
                #print(median_class_id)
                    
            out.write(frame)
            frame = cv2.resize(frame,(600,400))
            self.preview_frame = frame
        
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            #print(self.use_camera == 1)
            if self.use_camera == 1:
                if self.checker != img:
                    self.exec_f.canvas.create_image((0,0),image=img,anchor=tk.NW,tag="img")
                    self.checker = img
            if self.stop:
                self.exec_f.canvas.delete("all")
                break
        cap.release()
    def writefile(self):
        file = open(self.logPath+'logger.log','w')
        file.write(self.exec_f.log)
        file.close()
execute = Execute()
