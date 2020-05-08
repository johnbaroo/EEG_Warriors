import self as self
from numpy import array
import numpy as np
from scipy.io import loadmat
from statistics import mean
import csv

# git_Ebrahim_data_processing
class data_processing:
# git_Ebrahim_data_processing__dataset_read_0
    def dataset_read(self, n='A01T.mat'): # read .mat files and convert it into 3d array
        eeg_dataset = loadmat(n)
        IM_left_hand=[0]*6
        IM_right_hand=[0]*6
        IM_feet=[0]*6
        IM_tongue=[0]*6
        print(len(IM_tongue))
        for i_run in range(6):
            idx_s=round(3*96735/6.45)-1 # IM begin
            idx_e=round(6*96735/6.45)-1  # IM end

            idx_d=round((idx_e-idx_s)/4) -1

            IM_left_hand[i_run]=eeg_dataset['data'][0,i_run+3]['X'][0][0][idx_s:idx_s+idx_d,0:22]
            IM_right_hand[i_run]=eeg_dataset['data'][0,i_run+3]['X'][0][0][idx_s+idx_d:idx_s+2*idx_d,0:22]
            IM_feet[i_run]=eeg_dataset['data'][0,i_run+3]['X'][0][0][idx_s+2*idx_d:idx_s+3*idx_d,0:22]
            IM_tongue[i_run]=eeg_dataset['data'][0,i_run+3]['X'][0][0][idx_s+3*idx_d:idx_s+4*idx_d,0:22]
        return IM_left_hand,IM_right_hand,IM_feet,IM_tongue # return 4 3d arrays for each movement

