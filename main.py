import pyoperon
import numpy
import scipy
import matplotlib.pyplot as plt


class Problem:
    def __init__(self, name, train_data, test_data, args):
        self.name = name
        self.train_data = train_data
        self.test_data = test_data
        self.args = args
    
    def plot(self, ax=None):

        fig, ax = plt.subplots()
        
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        
        ax.scatter(x_train, y_train, label='Train Data', color='blue')
        ax.scatter(x_test, y_test, label='Test Data', color='red')
        
        ax.set_title(self.name)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        
        return ax
