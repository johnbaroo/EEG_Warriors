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

# git_Ebrahim!Eyoel_eeg_ann_main__create_network_1
# Create neural network
d = ann1.model(train_x.T, train_y.T, n_h=8, num_iters=150000, alpha=.999999, print_cost=True)
# git_Ebrahim!Eyoel_eeg_ann_main__test an_input_2
print(ann1.predict(d,[[0.46323205636283576],[0.5367679436371643],[0.15176790199608783],[.13546412479731691],[.1835052910710305]]))
print(ann1.predict(d,[[0.4007120253164557],[0.5992879746835443],[0.372924967491331],[0.12406002861735548],[0.15390890700539062]]))
print(ann1.predict(d,[[0.5163059163059163],[0.4836940836940837],[0.22758673146172312],[0.14962094150171226],[0.22386426135858803]]))
print(ann1.predict(d,[[0.4281789638932496],[0.5718210361067504],[0.4709762881212768],[0.13792605884568757],[0.1902359770870407]]))



# Save ann parameters
# np.save('parameters.npy', d)
# file_csv_11 = open('parameters.csv', 'w', newline='')
# file_csv_1 = csv.writer(file_csv_11, delimiter=',')
# file_csv_1.writerow(d['W1'])
# file_csv_1.writerow(d['b1'])
# file_csv_1.writerow(d['W2'])
# file_csv_1.writerow(d['b2'])
# file_csv_11.close()
# # load ann parameters and intilize new ann
# para=np.load('parameters.npy',allow_pickle='TRUE').item()
# print(para['W1'])
# d = ann1.model(params=para)
# # testing it
# print(ann1.predict(d,[[0.46323205636283576],[0.5367679436371643],[0.15176790199608783],[.13546412479731691],[.1835052910710305]]))


