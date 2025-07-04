import pyoperon
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter




#class Solution:

class Problem:
    def __init__(self, name, train_data, test_data, args):
        self.name = name
        self.train_data = train_data
        self.test_data = test_data
        self.args = args
    
    def plot_data(self, ax=None, train=True):

        fig, ax = plt.subplots()
        
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        
        if train:
            ax.plot(x_train, y_train, label='Train Data', color='blue')
            mode = " train"
        else:
            ax.plot(x_test, y_test, label='Test Data', color='red')
            mode = " test"

        
        ax.set_title(self.name + mode)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        
        return ax
  
    
    def solve(self):
        reg = SymbolicRegressor(
            allowed_symbols=self.args['allowed_symbols'], #"add,sub,mul,aq,sin,constant,variable", 
            epsilon =self.args['epsilon'], #, 10**(-2)
            objectives= self.args["objectives"], #[ 'r2', 'length' ],
            max_evaluations= self.args["max_evaluations"], #1000000,
            max_length= self.args["max_length"], #50,
            max_time= self.args["max_time"], #900,
            n_threads= self.args["n_threads"] #8
        )   
        reg.fit(self.train_data[0].reshape(-1, 1), self.train_data[1].ravel().reshape(-1, 1))
        res = [(s['objective_values'], s['tree'], s['minimum_description_length']) for s in reg.pareto_front_]
        print(reg.pareto_front_[0].keys())
        for obj, expr, mdl in res:
            print(f'{obj}, {mdl:.2f}, {reg.get_model_string(expr, 12)}')


