import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# git_Ebrahim!Eyoel_eeg_ann_class
class eeg_ann:
    np.random.seed(1)
    global X
    Y=np.array(1)

# git_Eyoel_eeg_ann_class__param_init_1
    def param_init(self,p):
        parameters=p

# git_Eyoel_eeg_ann_class__sigmoid_2
    def sigmoid(self,z):
        s = 4 / (1 + np.exp(-z))
        return s

# git_Eyoel_eeg_ann_class__initialize_3
    def initialize(self,n_x, n_h, n_y):

        np.random.seed(2)
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.random.rand(n_h, 1)
        W2 = np.random.rand(n_y, n_h)
        b2 = np.random.rand(n_y, 1)
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

# git_Eyoel_eeg_ann_class__forward_prop_4
    def forward_prop(self,X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = eeg_ann.sigmoid(self,Z2)

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache

