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

# git_Julio_data_processing__features_eeg_1
    def features_eeg(self,d_r_v=dataset_read(self, n='A01T.mat'),chan_sel=0):
        IM_left_hand,IM_right_hand,IM_feet,IM_tongue=d_r_v
        feat_eeg_left_hand =[[[0 for col in range(5)] for col in range(22)] for row in range(6)]
        feat_eeg_right_hand =[[[0 for col in range(5)] for col in range(22)] for row in range(6)]
        feat_eeg_feet =[[[0 for col in range(5)] for col in range(22)] for row in range(6)]
        feat_eeg_tongue =[[[0 for col in range(5)] for col in range(22)] for row in range(6)]
        file_csv_11 = open('file_csv_1.csv', 'w', newline='')
        file_csv_1 = csv.writer(file_csv_11, delimiter=',')

        file_csv_11_y = open('file_csv_1_y.csv', 'w', newline='')
        file_csv_1_y = csv.writer(file_csv_11_y, delimiter=',')
# git_Julio_data_processing__features_eeg_2
        for run_idx in range(6):
            for channel_eeg in range(0,22):
                # Left Hand
                min_v=array(IM_left_hand[run_idx][:,channel_eeg]).min()
                max_v=array(IM_left_hand[run_idx][:,channel_eeg]).max()

                min_r=abs(min_v)/(abs(max_v)+abs(min_v))
                max_r=abs(max_v)/(abs(max_v)+abs(min_v))

                av_r=np.mean(array(IM_left_hand[run_idx][:,channel_eeg]))
                std_r=np.std(IM_left_hand[run_idx][:,channel_eeg])
                var_r=np.var(IM_left_hand[run_idx][:,channel_eeg])
                feat_eeg_left_hand[run_idx][channel_eeg]=[min_r,max_r,av_r,std_r,var_r]
                if chan_sel==channel_eeg:
                    file_csv_1.writerow([min_r,max_r,av_r,std_r/100,var_r/1000])
                    file_csv_1_y.writerow([0.0])


                # Right Hand
                min_v=array(IM_right_hand[run_idx][:,channel_eeg]).min()
                max_v=array(IM_right_hand[run_idx][:,channel_eeg]).max()

                min_r=abs(min_v)/(abs(max_v)+abs(min_v))
                max_r=abs(max_v)/(abs(max_v)+abs(min_v))

                av_r=np.mean(array(IM_right_hand[run_idx][:,channel_eeg]))
                std_r=np.std(IM_right_hand[run_idx][:,channel_eeg])
                var_r=np.var(IM_right_hand[run_idx][:,channel_eeg])
                feat_eeg_right_hand[run_idx][channel_eeg]=[min_r,max_r,av_r,std_r,var_r]
                if chan_sel==channel_eeg:
                    file_csv_1.writerow([min_r,max_r,av_r,std_r/100,var_r/1000])
                    file_csv_1_y.writerow([.3])


                # Feet
                min_v=array(IM_feet[run_idx][:,channel_eeg]).min()
                max_v=array(IM_feet[run_idx][:,channel_eeg]).max()

                min_r=abs(min_v)/(abs(max_v)+abs(min_v))
                max_r=abs(max_v)/(abs(max_v)+abs(min_v))

                av_r=np.mean(array(IM_feet[run_idx][:,channel_eeg]))
                std_r=np.std(IM_feet[run_idx][:,channel_eeg])
                var_r=np.var(IM_feet[run_idx][:,channel_eeg])
                feat_eeg_feet[run_idx][channel_eeg]=[min_r,max_r,av_r,std_r,var_r]
                if chan_sel==channel_eeg:
                    file_csv_1.writerow([min_r,max_r,av_r,std_r/100,var_r/1000])
                    file_csv_1_y.writerow([.7])


                # Tongue
                min_v=array(IM_tongue[run_idx][:,channel_eeg]).min()
                max_v=array(IM_tongue[run_idx][:,channel_eeg]).max()

                min_r=abs(min_v)/(abs(max_v)+abs(min_v))
                max_r=abs(max_v)/(abs(max_v)+abs(min_v))

                av_r=np.mean(array(IM_tongue[run_idx][:,channel_eeg]))
                std_r=np.std(IM_tongue[run_idx][:,channel_eeg])
                var_r=np.var(IM_tongue[run_idx][:,channel_eeg])
                feat_eeg_tongue[run_idx][channel_eeg]=[min_r,max_r,av_r,std_r,var_r]
                if chan_sel==channel_eeg:
                    file_csv_1.writerow([min_r,max_r,av_r,std_r/100,var_r/1000])
                    file_csv_1_y.writerow([1.0])

        file_csv_11.close()
        file_csv_11_y.close()
