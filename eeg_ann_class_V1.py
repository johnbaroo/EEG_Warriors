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

# git_Juilio_eeg_ann_class__compute_cost_5
    def compute_cost(self,A2, Y, parameters):
        m = Y.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']
        try:
            logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2 * 1 / np.amax(A2)), (1 - Y * 1 / np.amax(Y)))
        except:
            logprobs=Y
        logprobs[np.isnan(logprobs)] = 0
        cost = - np.sum(logprobs) / m
        cost = np.squeeze(cost)

        return cost

# git_Juilio_eeg_ann_class____back_prop_6
    def back_prop(self,parameters, cache, X, Y):
        m = Y.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']
        A1 = cache['A1']
        A2 = cache['A2']

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.square(A1))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads
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

# git_Juilio_eeg_ann_class__compute_cost_5
    def compute_cost(self,A2, Y, parameters):
        m = Y.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']
        try:
            logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2 * 1 / np.amax(A2)), (1 - Y * 1 / np.amax(Y)))
        except:
            logprobs=Y
        logprobs[np.isnan(logprobs)] = 0
        cost = - np.sum(logprobs) / m
        cost = np.squeeze(cost)

        return cost

# git_Juilio_eeg_ann_class____back_prop_6
    def back_prop(self,parameters, cache, X, Y):
        m = Y.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']
        A1 = cache['A1']
        A2 = cache['A2']

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.square(A1))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

# git_Ebrahim_eeg_ann_class__update_params_7
    def update_params(self,parameters, grads, alpha):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        return parameters

# git_Ebrahim!Eyoel_eeg_ann_class__model_0
    def model(self,X=0, Y=0, n_h=0, num_iters=0, alpha=0, print_cost=0,params=0):
        if params==0:
            print(X)
            print(Y)
            np.random.seed(3)
            n_x = X.shape[0] # width of X
            n_y = Y.shape[0] # width of Y

            parameters = eeg_ann.initialize(self,n_x, n_h, n_y)
            W1 = parameters['W1']
            b1 = parameters['b1']
            W2 = parameters['W2']
            b2 = parameters['b2']
            alpha_tmp=alpha
            costs = []
            cost_old=0
            alpha1 = 1
            for i in range(0, num_iters):

                A2, cache = eeg_ann.forward_prop(self,X, parameters)

                cost = eeg_ann.compute_cost(self,A2, Y, parameters)
                # if np.isnan(cost):
                #     m = Y.shape[1]
                #     W1 = parameters['W1']
                #     W2 = parameters['W2']
                #     print(Y*1/np.amax(Y))
                #     print("A2",np.multiply(np.log(1 - A2*1/np.amax(A2)), (1 - Y*1/np.amax(Y))))
                #
                #     logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2*1/np.amax(A2)), (1 - Y*1/np.amax(Y)))
                #     logprobs[np.isnan(logprobs)] = 0
                #     print("log", logprobs)
                #     cost = - np.sum(logprobs) / m
                #     print("np.sum(logprobs) / m", cost)
                #     cost = np.squeeze(cost)
                #     print("cost",cost)
                #     break
                cost_new = cost
                grads = eeg_ann.back_prop(self,parameters, cache, X, Y)

                diff=cost_old-cost_new
                if (diff<.006 and diff>.00001 or np.isnan(cost_old)):
                    # alpha1 = (100000 / i) * alpha
                    alpha1 = alpha-(i / num_iters)*alpha/5
                    # alpha1 = alpha
                    parameters = eeg_ann.update_params(self,parameters, grads, alpha1)
                elif (diff<0.00005):
                    # alpha1 = (100000 / i) * alpha
                    alpha1 = alpha+(i / num_iters)*alpha/5
                    # alpha1 = alpha
                    parameters = eeg_ann.update_params(self,parameters, grads, alpha1)
                else:
                    parameters = eeg_ann.update_params(self,parameters, grads, alpha)

                if i % 100 == 0:
                    cost_old=cost_new
                    costs.append(cost_new)
                # alpha=alpha-.0001
                # if alpha<.001:
                #     alpha=alpha_tmp
                if cost<.1:
                    break
                if print_cost and i % 1000 == 0:
                    print("Cost after iteration %i: %f %f" % (i, cost_old,diff))
                    if i <= 20000:
                        print("Learning rate after iteration %i: %f" % (i, alpha))
                    else:
                        print("Learning rate after iteration %i: %f  %f" % (i, alpha1,cost_new-cost_old))

            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(alpha))
            plt.show()
            # A2, cache = eeg_ann.forward_prop(self, X, parameters)
            # W2 = parameters['W2']
            # print(X)
            # X=[[0.46323205636283576],[0.5367679436371643],[0.15176790199608783],[.13546412479731691],[.1835052910710305]]
            # Z1 = np.dot(W1, X) + b1
            # print("np.dot(W1, X) + b1",Z1)
            # A1 = np.tanh(Z1)
            # print("np.tanh(Z1)", A1)
            # print("W2", W2)
            # Z2 = np.dot(W2, A1) + b2
            # print("np.dot(W2, A1) + b2", Z2)
            # A2 = eeg_ann.sigmoid(self, Z2)
            #
            # m = Y.shape[1]
            # print("m",m)
            # W1 = parameters['W1']
            # print("W1", W1)
            #
            # print("A2", A2)
            # print("Y",Y)
            # print("np.multiply(np.log(A2), Y)",np.multiply(np.log(A2), Y))
            #
            #
            # logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
            # cost = - np.sum(logprobs) / m
            # cost = np.squeeze(cost)
            # print(float(cost))

        else:
            parameters=params
        return parameters

# git_Ebrahim_eeg_ann_class__predict_8
    def predict(self,parameters, X):
        A2, cache = eeg_ann.forward_prop(self,X, parameters)
        predictions = A2

        return predictions
