import tkinter as tk
from tkinter import ttk
from tkinter import font
from tkinter import scrolledtext as st
from PIL import Image,ImageTk
from execute_tflite import execute
import sqlite3
import os
import configparser



class ExecuteFrame(tk.Frame):
    def __init__(self,start_f,master=None,**kwargs):

        #file = open('./logger.log','r')
        #self.log=file.read()
        #file.close()
        self.log = ""

        tk.Frame.__init__(self,master,**kwargs,width=700,height=700)
        self.canvas = tk.Canvas(self,width=600,height=400)
        self.canvas.place(anchor=tk.CENTER,x=350,y=200)

        self.txtbox = st.ScrolledText(self,width=70,height=10,font=font.Font(size=16))
        self.txtbox.place(anchor=tk.CENTER,x=350,y=520)
        
        self.txtbox.insert(tk.INSERT,self.log)
        self.txtbox.configure(state ='disabled')

        self.button = tk.Button(self,text="Stop",font=font.Font(size=26),command=self.Stop)
        self.button.place(anchor=tk.CENTER,x=350,y=660)
        
        self.parent = start_f
        self.model = execute
        self.model.set_exec_f(self)
        
        self.checker = None

    
        #self.model.modelpath = int(config.get(section1,"model"))
        
        
        #self.model.use_camera=1
    def exec_timer(self):
        if self.parent.time == "1":
            self.afterID=self.after(self.parent.time_chenger(self.parent.ed_time),self.Stop)
            #print("exec_stop")
            #print(self.afterID)
      
    def Back(self):
        self.pack_forget()
        self.parent.pack()
        self.parent.exec_timer()
    
    def Start(self):
        self.exec_timer()
        self.pack()
        config_dict=self.parent.Read_Config()
        self.model.select_sound=(config_dict["sound"].split(","))
        self.model.select_sound.remove("None")
        self.model.stop = False
        self.model.exec_model()

    def Stop(self):
        if self.parent.time == "1":
            self.after_cancel(self.afterID)
        self.model.stop = True
        self.pack_forget()
        self.parent.pack()
        self.parent.exec_timer()
    
    def update(self):
        
        if self.checker != self.model.img:
            self.canvas.create_image((0,0),image=self.model.img,anchor=tk.NW,tag="img")
            self.checker = self.model.img
        else:
            pass
        
        self.after(1,self.update)

    

    def set_config(self):
        if not os.path.isdir("./config"):
            os.mkdir("./config")
        if not os.path.exists("./config/config.ini"):
            self.create_config()
        
        config = configparser.SafeConfigParser()
        config.read("./config/config.ini")

        section1 = "exec"
        if not config.has_section(section1):
            self.create_config()
            config.read("./config/config.ini")
        if not  config.has_option(section1,"sound"):
            self.create_config()
            config.read("./config/config.ini")
        self.sound_path = config.get(section1,"sound")
