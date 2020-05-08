import threading
import time
from tkinter import *
from tkinter import filedialog
import numpy as np
import pandas as pd
from project_data import data_processing
from eeg_ann_class_V1 import eeg_ann
import logging

# git_Ebrahim!Eyoel!Julio_eeg_gui


# git_Eyoel_eeg_gui__ ًWindow Initialize_0
me = Tk()
me.geometry("500x560") # Size
me.title("EEG Classifier")
me.resizable(0, 0) # fix size

me.config()
logger.info('Gui intilized')
# Bold title in the Top
melabel = Label(me, text="EEG Classifier Project \n EEG Worriers", font=("Times", 30, 'bold'))
melabel.pack(side=TOP)

# Global variable
global file_type
global net_eeg
file_type=".npy"

# git_Ebrahim_eeg_gui__clickbut_2
# Callback function of buttons pressing (two load buttons)
def clickbut(typee):
    global net_eeg
    if typee==".mat":
        logger.info('Load dataset button pressed, chose .mat, and now loading file and processing data')
        try:
            file=filedialog.askopenfilename(defaultextension='.mat')
        except:
            logger.error("File openning error")

        text_load.set("Loading  .... ")
        txt_load.config(fg="green")

        data_processing.features_eeg(data_processing.dataset_read(file),chan_sel=int(txt_box_ch.get()))
        logger.info('Load dataset, Features extracted')
        ann1 = eeg_ann()
        train_x = pd.read_csv("file_csv_1.csv", header=None)
        train_x = np.array(train_x)
        train_y = pd.read_csv("file_csv_1_y.csv", header=None)
        train_y = np.array(train_y)
        # Create neural network

        net_eeg = ann1.model(train_x.T, train_y.T, n_h=float(txt_box_ann_hid_lay.get()),
                             num_iters=float(txt_box_ann_iter_num.get()), alpha=float(txt_box_ann_alpha.get()),
                             print_cost=True)
        np.save('parameters.npy', net_eeg)
        text_load.set("Done !! ")
        txt_load.config(fg="red")
        logger.info('Load dataset finished, ANN trained and its saved in paramters.np')

    elif typee==".npy":
        logger.info('Load ANN button pressed')
        try:
            file = filedialog.askopenfilename(defaultextension='.npy')
        except:
            logger.error("File openning error")
        text_load.set("Loading  .... ")
        txt_load.config(fg="green")
        ann1 = eeg_ann()
        # Create neural network
        para = np.load(file, allow_pickle='TRUE').item()
        net_eeg=ann1.model(params=para)

        text_load.set("Done !! ")
        txt_load.config(fg="red")
        logger.info('ANN parameters loaded')

# git_Ebrahim_eeg_gui__eval_but_3
# Fuction of press Evaluate button
def eval_but():
    global eeg_ann
    ann1 = eeg_ann()
    res=ann1.predict(parameters=net_eeg, X=[[float(txt_box_min.get())],[float(txt_box_max.get())],[float(txt_box_mean.get())],[float(txt_box_sd.get())],[float(txt_box_var.get())]])
    res=res*10
    if res<1.5:
        result_var_text.set("Left Hand")
    elif res>1.5 and res<4.5:
        result_var_text.set("Right Hand")
    elif res>4.5 and res<9:
        result_var_text.set("Feet")
    else:
        result_var_text.set("Tongue")
    logger.info('Movement Detect pressed, and the result: '+result_var_text.get())


# git_Ebrahim!Eyoel!Julio_eeg_gui__gui_layers_objects_1
# Buttons Layer
frame_bts = Frame(me)
frame_bts.pack(fill=X)
btn_load_ds = Button(frame_bts, padx=2, pady=2, bd=4, bg='white', command=lambda: clickbut(".mat"), text="Load dataset for training",
              font=("Courier New", 13))
btn_load_ds.pack(side=LEFT,padx=5,pady=5)
btn_load_ann = Button(frame_bts, padx=2, pady=2, bd=4, bg='white', command=lambda: clickbut(".npy"), text="Load neural network",
              font=("Courier New", 13))
btn_load_ann.pack(side=RIGHT,padx=5,pady=5)

# Parameters of ANN layer
frame_entry_chan = Frame(me)
frame_entry_chan.pack(fill=X,pady=5)
frame_entry_chan.config(bg="LightSteelBlue1")


txt_ch=Label(frame_entry_chan, text="Channel Number", font=("Times", 11))
txt_ch.pack(side=LEFT,padx=10,pady=0)
txt_ch.config(bg="LightSteelBlue1")

txt_box_ch = Entry(frame_entry_chan)
txt_box_ch.config(width=3)
txt_box_ch.pack(side=LEFT, padx=3, pady=0)
txt_box_ch.insert(0,"0")

frame_entry_ann = Frame(frame_entry_chan)
frame_entry_ann.pack(side=RIGHT,fill=X,ipadx=8)
frame_entry_ann.config(bg="LightSkyBlue4")

text_ann_hid_lay=Label(frame_entry_ann, text="H_Layer", font=("Times", 11))
text_ann_hid_lay.pack(side=LEFT,padx=10,pady=0)
text_ann_hid_lay.config(bg="LightSkyBlue4")

txt_box_ann_hid_lay = Entry(frame_entry_ann)
txt_box_ann_hid_lay.config(width=3)
txt_box_ann_hid_lay.pack(side=LEFT, padx=0, pady=0)
txt_box_ann_hid_lay.insert(0,"8")

text_ann_iter_num=Label(frame_entry_ann, text="Iter_num", font=("Times", 11))
text_ann_iter_num.pack(side=LEFT,padx=10,pady=0)
text_ann_iter_num.config(bg="LightSkyBlue4")

txt_box_ann_iter_num = Entry(frame_entry_ann)
txt_box_ann_iter_num.config(width=7)
txt_box_ann_iter_num.pack(side=LEFT, padx=0, pady=0)
txt_box_ann_iter_num.insert(0,"200000")

text_ann_alpha=Label(frame_entry_ann, text="Alpha", font=("Times", 11))
text_ann_alpha.pack(side=LEFT,padx=10,pady=0)
text_ann_alpha.config(bg="LightSkyBlue4")

txt_box_ann_alpha = Entry(frame_entry_ann)
txt_box_ann_alpha.config(width=7)
txt_box_ann_alpha.pack(side=LEFT, padx=0, pady=0)
txt_box_ann_alpha.insert(0,".999999")

# File upload messege layer
frame_file_lb = Frame(me)
frame_file_lb.pack(fill=X,pady=5)
text_load = StringVar(me)
text_load.set("No file .. ")
txt_load=Label(frame_file_lb, textvariable=text_load, font=("Times", 11))
txt_load.pack(side=LEFT,padx=10,pady=0)


####### Features and Image Layer #########
frame_group = Frame(me)
frame_group.pack(fill=X)
Par_group = LabelFrame(frame_group, text = "Parameters", padx=10, pady=10)
Par_group.pack(side=LEFT,padx=15,pady=5)


# Min
frame1 = Frame(Par_group)
frame1.pack(fill=X)
txt_box_min = Entry(frame1)
txt_box_min.pack(side=RIGHT, padx=5, pady=5)
txt_box_min.insert(0,"0.4007120253164557")

txt_box_min.bind("<Button-1>", lambda e: txt_box_min.delete(0, END))

lab_min=Label(frame1,text="Min Val")
lab_min.pack(side=LEFT, anchor=N)

# Max
frame2 = Frame(Par_group)
frame2.pack(fill=X)
txt_box_max = Entry(frame2)
txt_box_max.pack(side=RIGHT, padx=5, pady=5)
txt_box_max.insert(0,"0.5992879746835443")
txt_box_max.bind("<Button-1>", lambda e: txt_box_max.delete(0, END))

lab_max=Label(frame2,text="Max Val")
lab_max.pack(side=LEFT, anchor=N)

# Mean
frame3 = Frame(Par_group)
frame3.pack(fill=X)
txt_box_mean = Entry(frame3)
txt_box_mean.pack(side=RIGHT, padx=5, pady=5)
txt_box_mean.insert(0,"0.372924967491331")
txt_box_mean.bind("<Button-1>", lambda e: txt_box_mean.delete(0, END))

lab_mean=Label(frame3,text="Mean Val")
lab_mean.pack(side=LEFT, anchor=N)

# SD
frame4 = Frame(Par_group)
frame4.pack(fill=X)
txt_box_sd = Entry(frame4)
txt_box_sd.pack(side=RIGHT, padx=5, pady=5)
txt_box_sd.insert(0,"0.12406002861735548")
txt_box_sd.bind("<Button-1>", lambda e: txt_box_sd.delete(0, END))

lab_sd=Label(frame4,text="SD Val")
lab_sd.pack(side=LEFT, anchor=N)

# Var
frame5 = Frame(Par_group)
frame5.pack(fill=X)
txt_box_var = Entry(frame5)
txt_box_var.pack(side=RIGHT, padx=5, pady=5)
txt_box_var.insert(0,"0.15390890700539062")
txt_box_var.bind("<Button-1>", lambda e: txt_box_var.delete(0, END))

lab_var=Label(frame5,text="Var Val")
lab_var.pack(side=LEFT, anchor=N)

# Check Button
frame6 = Frame(Par_group)
frame6.pack(fill=X)
btn_check = Button(frame6, padx=2, pady=2, bd=4, bg='white', command=lambda: eval_but(), text="Detect movement")
btn_check.pack(fill=X)

# image
img = PhotoImage(file = "./img.png")
img = img.zoom(25)
img = img.subsample(40)
background_label = Label(frame_group, image=img)
background_label.config(width=280,height=220)
background_label.pack()

# Result Layer
frame_result = Frame(me)
frame_result.pack(fill=X,side=BOTTOM)
result_var_text=StringVar(me)
result_var = Label(frame_result, textvariable=result_var_text, font=("Times", 25, 'bold'),fg="red")
result_var.pack(side=BOTTOM)
result_fix = Label(frame_result, text=("The movement is"), font=("Times", 20))
result_fix.pack(side=BOTTOM)
