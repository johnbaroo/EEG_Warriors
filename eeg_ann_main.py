import csv
import numpy as np
import pandas as pd
from eeg_ann_class_V1 import eeg_ann

# git_Ebrahim!Eyoel_eeg_ann_main

# git_Ebrahim!Eyoel_eeg_ann_main__read_db_0
ann1=eeg_ann()
train_x = pd.read_csv("file_csv_1.csv",header=None)
train_x = np.array(train_x)
train_y = pd.read_csv("file_csv_1_y.csv",header=None)
train_y = np.array(train_y)
