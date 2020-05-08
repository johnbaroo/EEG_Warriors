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

