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
import os

class Solution:
    def __init__(self, name, problem, model, tree, string_expression, r2, mse, mdl, regressor, length):
        self.name = name
        self.problem = problem
        self.model = model
        self.tree = tree
        self.string_expression = string_expression
        self.r2 = r2
        self.mse = mse
        self.mdl = mdl
        self.regressor = regressor 
        self.length = length


        
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
    
    def to_latex(self):
        formatted_expr, b_vals, sympy_expr = self.extract_and_format(self.string_expression)
        latex_expr = sp.latex(sympy_expr)
        return latex_expr, b_vals
 


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
        self.solve_state = False
    

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
        self.solve_state = True

        for idx, s in enumerate(reg.pareto_front_):
            model = s['model']
            tree = s['tree']
            string_expression = reg.get_model_string(tree, 12)
            r2 = s['objective_values'][0]  # assuming R2
            mse = s['mean_squared_error']
            mdl = s['minimum_description_length']
            length = s['length']

            solution = Solution(
                name=self.name + f": solution {idx}",
                problem=self,
                model=model,
                tree=tree,
                string_expression=string_expression,
                r2=r2,
                mse=mse,
                mdl=mdl,
                regressor=reg,
                length = length
            )
            self.add_solution(solution)


    def plot_l_vs_mse(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        lengths = [s.length for s in self.solutions]
        r2es = [s.mse for s in self.solutions]

        ax.scatter(lengths, r2es, color='blue')
        ax.set_xlabel('Length')
        ax.set_ylabel('MSE')
        ax.set_title(f'{self.name} - Length vs MSE (Pareto Front)')

        return ax


    def export_solutions_to_latex(self, n, filename="solutions.tex"):
        """
        Export up to n solutions to a LaTeX file, with expressions and parameter tables.
        The .tex file and compilation files are stored in a folder named after the problem inside 'latex_files'.
        """
        # Create a subfolder in latex_files named after the problem
        safe_name = self.name.replace(" ", "_").replace("/", "_")+"no_solutions_{}".format(n)
        problem_dir = os.path.join("latex_files", safe_name)
        os.makedirs(problem_dir, exist_ok=True)
        filepath = os.path.join(problem_dir, filename)

        header = r"""\documentclass{article}Problem_g.export_solutions_to_latex(n=5, filename="gaussian_solutions.tex")
\usepackage{amsmath}
\usepackage{booktabs}
\begin{document}
\section*{Symbolic Regression Solutions}
"""
        footer = r"\end{document}"

        body = ""
        for i, sol in enumerate(self.solutions[:n]):
            latex_expr, b_vals = sol.to_latex()
            body += f"\\subsection*{{Solution {i+1}}}\n"
            body += f"\\[\n{latex_expr}\n\\]\n"
            if b_vals:
                body += "\\begin{center}\n\\begin{tabular}{cc}\n"
                body += "\\toprule\nParameter & Value \\\\n\\midrule\n"
                for k, v in b_vals.items():
                    body += f"${k}$ & {v:.6g} \\\\n"
                body += "\\bottomrule\n\\end{tabular}\n\\end{center}\n"
            body += "\n\\vspace{1em}\n"

        with open(filepath, "w") as f:
            f.write(header + body + footer)
        print(f"LaTeX file written to {filepath}")



    def __str__(self):
        return f"Problem: {self.name}, Train Data: {self.train_data[0].shape}, Test Data: {self.test_data[0].shape}, Solutions: {len(self.solutions)}, Solve_state: {self.solve_state}"


class hyper_parameter_search:
    def __init__(self, problem, args):
        self.problem = problem
        self.args = args
        self.results = []
    
    def run(self, max_complexity, opt_parameters):
        



