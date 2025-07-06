import pyoperon
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
import sympy as sp
import re
from IPython.display import display, Math


class Solution:
    def __init__(self, name, problem, model, tree, string_expression, r2, mse, mdl, regressor):
        self.name = name
        self.problem = problem
        self.model = model
        self.tree = tree
        self.string_expression = string_expression
        self.r2 = r2
        self.mse = mse
        self.mdl = mdl
        self.regressor = regressor 


        
    def extract_and_format(self, variable_prefix='X'):
        var_names = sorted(set(re.findall(rf'{variable_prefix}\d+', self.string_expression)))
        vars_sympy = {v: sp.Symbol(v) for v in var_names}
        
        expr_for_sympy = self.string_expression
        for name in var_names:
            expr_for_sympy = re.sub(rf'\b{name}\b', name, expr_for_sympy)

        expr = sp.sympify(expr_for_sympy, locals=vars_sympy)
        expanded = sp.expand(expr)

        constants = []
        replacements = {}

        def replace_constants(e):
            if e.is_Number and e not in constants:
                constants.append(e)
                idx = len(constants) - 1
                sym = sp.Symbol(f'b{idx}')
                replacements[e] = sym
                return sym
            elif e in constants:
                return replacements[e]
            elif e.is_Atom:
                return e
            else:
                return e.func(*[replace_constants(arg) for arg in e.args])

        expr_with_b = replace_constants(expanded)
        formatted_str = str(expr_with_b)
        b_vals = {(f'b{i}'): float(c) for i, c in enumerate(constants)}

        return formatted_str, b_vals, expr_with_b

    def display_expression(self):
        formatted_expr, b_vals, sympy_expr = self.extract_and_format(self.string_expression)
        return display(Math(sp.latex(sympy_expr)))


    def plot_results(self, ax=None, train=True):
        x, y = self.problem.train_data if train else self.problem.test_data
        y_pred = self.regressor.evaluate_model(self.tree, x.reshape(-1, 1))

        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.plot(x, y, label='True', color='black')
        ax.plot(x, y_pred, label=f'Prediction ({self.name})', linestyle='--')
        ax.set_title(f'{self.name} {"Train" if train else "Test"}')
        ax.legend()
        return ax
 


    def __str__(self):
        return (f"{self.name}: expr={self.string_expression}, "
            f"R²={self.r2:.4f}, MSE={self.mse:.4f}, MDL={self.mdl:.2f}")
    






class Problem:
    def __init__(self, name, train_data, test_data, args):
        self.name = name
        self.train_data = train_data
        self.test_data = test_data
        self.args = args
        self.solutions = []
        self.symbolic_regressor = None
    

    def add_solution(self, solution):
        self.solutions.append(solution)

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

        reg.fit(self.train_data[0].reshape(-1, 1), self.train_data[1].ravel())
        res = [(s['objective_values'], s['tree'], s['minimum_description_length'], s['mean_squared_error']) for s in reg.pareto_front_]
        print(reg.pareto_front_[0].keys(), type(reg.pareto_front_[0]))
        for obj, expr, mdl, mse in res:
            print(f'{obj}, {mdl:.2f}, {reg.get_model_string(expr, 12)}, {mse:.2f}')
        
          
        self.symbolic_regressor = reg

        for idx, s in enumerate(reg.pareto_front_):
            model = s['model']
            tree = s['tree']
            string_expression = reg.get_model_string(tree, 12)
            r2 = s['objective_values'][0]  # assuming R2
            mse = s['mean_squared_error']
            mdl = s['minimum_description_length']

            solution = Solution(
                name=self.name + f": solution {idx}",
                problem=self,
                model=model,
                tree=tree,
                string_expression=string_expression,
                r2=r2,
                mse=mse,
                mdl=mdl,
                regressor=reg
            )
            self.add_solution(solution)


    def plot_mdl_vs_mse(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        lengths = [s.mdl for s in self.solutions]
        r2es = [s.mse for s in self.solutions]

        ax.scatter(lengths, r2es, color='blue')
        ax.set_xlabel('Minimal Description Length (MDL)')
        ax.set_ylabel('MSE')
        ax.set_title(f'{self.name} - Length vs MSE')

        return ax


    def __str__(self):
        return f"Problem: {self.name}, Train Data: {self.train_data[0].shape}, Test Data: {self.test_data[0].shape}, Solutions: {len(self.solutions)}"


