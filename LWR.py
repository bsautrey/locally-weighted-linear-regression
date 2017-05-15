# Implement locally weighted linear regression from Andrew Ng's CS229 course: http://cs229.stanford.edu/notes/cs229-notes1.pdf. Batch gradient descent is used to learn the parameters, i.e. minimize the cost function.
import random
from copy import copy
from math import exp

import numpy as np
import matplotlib.pyplot as plot

# alpha - The learning rate.
# dampen - Factor by which alpha is dampened on each iteration. Default is no dampening, i.e. dampen = 1.0
# tol - The stopping criteria
# theta - The parameters to be learned.
# weights - The weighting used for each data point, for each local regression.
# index - A point, 0 - m, where the local regression is to be evaluated.
# percents - A list containing the iteration number and the percent change in theta, from one iteration to the next.

class LWR():
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.alpha = None
        self.dampen = None
        self.tao = None
        self.tol = None
        self.theta = None
        self.weights = None
        self.index = None
        self.percents = None
        
    def set_X(self,X):
        self.X = X
    
    def set_Y(self,Y):
        self.Y = Y
        
    def set_alpha(self,alpha=0.001,dampen=1.0):
        self.alpha = alpha
        self.dampen = dampen
        
    def set_tao(self,tao=1.0):
        self.tao = tao
        
    def set_tolerance(self,tol=0.001):
        self.tol = tol
        
    def initialize_theta(self,theta=None):
        if not theta:
            number_of_parameters = self.X.shape[1]
            theta = copy(self.X[0,:])
            theta.resize((1,number_of_parameters))
            
        self.theta = theta
        
    def _initialize_percents(self):
        self.percents = []
        
    def _calculate_weights(self):
        self.weights = []
        number_of_rows = self.X.shape[0]
        x_1 = self.X[self.index,:].T
        l_1 = x_1.shape[0]
        x_1.resize((1,l_1))
        for i in xrange(number_of_rows):
            x_2 = self.X[i,:].T
            l_2 = x_2.shape[0]
            x_2.resize((l_2,1))
            weight = self._calculate_weight(x_1,x_2)
            self.weights.append(weight)
                
    def _calculate_weight(self,x_1,x_2):
        x_diff = x_1.T - x_2
        x_product = np.dot(x_diff.T,x_diff)
        weight = exp(-1*x_product/(2*self.tao**2))
        return weight
        
    def run_BGD(self,index,max_iterations=1000):
        self.index = index
        self._calculate_weights()
        self._initialize_percents()
        old_theta = copy(self.theta)
        iterations = 0
        number_of_rows = self.X.shape[0]
        number_of_columns = self.X.shape[1]
        while True:
            for i in xrange(number_of_rows):
                weight = self.weights[i]
                x = self.X[i,:]
                y = self.Y[i,:][0]
                x.resize((number_of_columns,1))
                for j in xrange(number_of_columns):
                    theta_j = self.theta[0][j]
                    x_j = x[j][0]
                    dot = np.dot(self.theta,x)[0][0]
                    new_theta_j = theta_j + self.alpha*weight*(y - dot)*x_j
                    self.theta[0][j] = new_theta_j
                
                if i == self.index:
                    iterations = iterations + 1
                    percent = self._calculate_convergence(old_theta)
                    self.percents.append((iterations,percent))
                    old_theta = copy(self.theta)
                    self.alpha = self.alpha*self.dampen
                    print iterations,percent,self.alpha,self.theta
                    if percent < self.tol or iterations > max_iterations:
                        return
                
    def _calculate_convergence(self,old_theta):
        diff = old_theta - self.theta
        diff = np.dot(diff,diff.T)**0.5
        length = np.dot(old_theta,old_theta.T)**0.5
        percent = 100.0*diff/length
        return percent
        
    def generate_example(self,index,sample_size=1000):
        # assemble data
        mean = 0.0
        var = 0.15
        slope = 0.5
        X = []
        Y = []
        for x in range(sample_size):
            x = 2*np.pi*x/sample_size
            y = np.sin(x) + slope*x + random.gauss(mean,var)
            X.append(x)
            Y.append(y)
            
        X = np.array(X)
        X.resize((sample_size,1))
        intercept = np.ones((sample_size))
        X = np.column_stack((X,intercept))
        Y = np.array(Y)
        Y.resize((sample_size,1))
        
        # initialize
        self.set_X(X)
        self.set_Y(Y)
        self.set_tao(0.25)
        self.set_alpha(alpha=0.05,dampen=1.0)
        self.set_tolerance(0.1)
        self.initialize_theta()
        self.run_BGD(index)
        
        # predict
        Y_hat = []
        number_of_parameters = self.theta.shape[1]
        number_of_rows = self.X.shape[0]
        for i in xrange(number_of_rows):
            x = self.X[i,:]
            x.resize((number_of_parameters,1))
            y_hat = np.dot(self.theta,x)[0][0]
            Y_hat.append(y_hat)
            
        Y_hat = np.array(Y_hat)
        
        # plot
        plot.scatter(self.X[:,0],self.Y,s=0.5)
        plot.plot(self.X[:,0],Y_hat)
        plot.show()
        
    def plot_convergence(self,start_index,end_index=None):
        if end_index:
            X,Y = zip(*self.percents[start_index:end_index])
        else:
            X,Y = zip(*self.percents[start_index:])
            
        plot.plot(X,Y)
        plot.show()

        