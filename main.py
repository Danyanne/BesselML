import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
import sympy as sp
import re
from IPython.display import display, Math
import os
import numpy as np
import optuna
import re
from typing import Tuple, Dict, Optional
from decimal import Decimal
from scipy import special

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
    

    def extract_and_format(self, variable_prefix: str = 'X') -> Tuple[str, Dict[str, float], sp.Expr]:
        """
        Parses the symbolic expression string stored in `self.string_expression`, replacing
        only "complex" numerical constants with symbolic parameters `b0`, `b1`, ..., and
        returns a LaTeX-formatted string along with a dictionary of these parameters.

        [Original docstring content preserved...]
        """
        
        # Input validation
        if not hasattr(self, 'string_expression') or not self.string_expression:
            raise ValueError("No expression string found in self.string_expression")
        
        expr_str = self.string_expression
        
        # Extract variable names more efficiently
        var_pattern = rf'{re.escape(variable_prefix)}\d+'
        var_names = sorted(set(re.findall(var_pattern, expr_str)), 
                        key=lambda x: int(x[len(variable_prefix):]))
        
        # Build local dictionary with only needed symbols
        local_dict = {name: sp.Symbol(name) for name in var_names}
        local_dict["sqrt"] = sp.sqrt
        
        # Additional common functions that might be needed
        local_dict.update({
            "exp": sp.exp,
            "log": sp.log,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "pi": sp.pi,
            "e": sp.E
        })
        
        try:
            expr = sp.sympify(expr_str, locals=local_dict)
            expanded = sp.expand(expr)
        except (sp.SympifyError, ValueError) as e:
            raise ValueError(f"Failed to parse expression: {e}")
        
        def is_complex_constant(c: sp.Basic) -> bool:
            """Check if a constant should be replaced with a b parameter."""
            if not c.is_Number:
                return False
                
            if isinstance(c, sp.Float):
                # More robust decimal counting using Decimal
                try:
                    dec = Decimal(str(float(c.evalf())))
                    # Get the string representation and count significant decimals
                    dec_str = str(abs(dec))
                    if '.' in dec_str:
                        # Remove trailing zeros and count remaining decimals
                        decimal_part = dec_str.split('.')[-1].rstrip('0')
                        return len(decimal_part) > 2
                    return False
                except:
                    # Fallback to original method
                    return False
                    
            elif isinstance(c, sp.Rational):
                return abs(c.q) > 10
            
            # Don't replace special constants
            if c in (sp.pi, sp.E, sp.I):
                return False
                
            return False
        
        # Collect all complex constants in a consistent order
        complex_constants = []
        seen = set()
        
        def collect_constants(expr: sp.Basic) -> None:
            """Recursively collect complex constants in preorder traversal."""
            if expr.is_Number and is_complex_constant(expr):
                # Use a hashable representation to avoid duplicates
                expr_hash = hash(expr.evalf())
                if expr_hash not in seen:
                    seen.add(expr_hash)
                    complex_constants.append(expr)
            elif hasattr(expr, 'args'):
                for arg in expr.args:
                    collect_constants(arg)
        
        collect_constants(expanded)
        
        # Create replacement mapping
        replacements = {const: sp.Symbol(f"b{i}") for i, const in enumerate(complex_constants)}
        
        # Apply replacements
        expr_with_b = expanded.subs(replacements)
        
        def clean_latex(expr: sp.Basic) -> str:
            """Clean up LaTeX formatting."""
            # Generate LaTeX with better default options
            latex = sp.latex(expr, 
                            fold_short_frac=True, 
                            fold_frac_powers=True, 
                            mul_symbol='·',
                            long_frac_ratio=3,  # Use \frac for better readability
                            mat_delim='(',
                            mat_str='matrix')
            
            # Remove redundant multiplications by 1
            latex = re.sub(r'(?<![0-9])1\s*·\s*', '', latex)  # Remove 1·X patterns
            latex = re.sub(r'\s*·\s*1(?![0-9])', '', latex)   # Remove X·1 patterns
            
            # Clean up spacing
            latex = re.sub(r'\s*·\s*', '·', latex)
            latex = re.sub(r'\s+', ' ', latex)  # Normalize whitespace
            
            # Improve subscript formatting for variables
            latex = re.sub(rf'{variable_prefix}_{{(\d+)}}', rf'{variable_prefix}_{{\1}}', latex)
            
            # Ensure consistent b parameter formatting
            latex = re.sub(r'b(\d+)', r'b_{\1}', latex)
            
            return latex.strip()
        
        formatted_str = clean_latex(expr_with_b)
        
        # Create b_vals dictionary with consistent float conversion
        b_vals = {}
        for i, const in enumerate(complex_constants):
            try:
                b_vals[f"b{i}"] = float(const.evalf())
            except:
                # Handle edge cases where evalf() might fail
                b_vals[f"b{i}"] = complex(const.evalf())
        
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

    def plot_residuals(self, ax=None, train=True):
        x, y = self.problem.train_data if train else self.problem.test_data
        y_pred = self.regressor.evaluate_model(self.tree, x.reshape(-1, 1))
        residuals = y - y_pred

        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.scatter(x, residuals, label='Residuals', color='red', s=2)
        ax.axhline(0, color='black', linestyle='--')

        ax.set_title(f'Residuals of {self.name} {"Train" if train else "Test"}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Residuals')
        ax.legend()
        
        return ax
    
    def plot_fractional_error_hypergeom(self, x_val, ax=None):
        y_pred = self.regressor.evaluate_model(self.tree, x_val.reshape(-1, 1))
        y = special.hyp2f1(2/3, 1, 7/6, x_val)  # Hypergeometric function for comparison

        fractional_error = np.abs((y - y_pred) / y)

        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.plot(abs(x_val), fractional_error, label='Fractional Error', color='orange')
        
        ax.set_title(f'Fractional error: {self.name}')
        ax.set_xlabel('-x')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Fractional Error')
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
            n_threads= self.args["n_threads"], #8
            generations= self.args["generations"], # 100,
            tournament_size= self.args["tournament_size"], # 4,
            #initialization_method= self.args["initialization_method"], # "random"
        )


        reg.fit(self.train_data[0].reshape(-1, 1), self.train_data[1].ravel())
        res = [(s['objective_values'], s['tree'], s['minimum_description_length'], s['mean_squared_error']) for s in reg.pareto_front_]
        
        #print(reg.pareto_front_[0].keys(), type(reg.pareto_front_[0]))
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
        ax.set_yscale('log')
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

        header = r"""\documentclass{article}
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
                body += "\\toprule\nParameter & Value \\\\\n\\midrule\n"
                for k, v in b_vals.items():
                    body += f"${k}$ & {v:.6g} \\\\\n"
                body += "\\bottomrule\n\\end{tabular}\n\\end{center}\n"

            body += "\n\\vspace{1em}\n"

        with open(filepath, "w") as f:
            f.write(header + body + footer)

        print(f"LaTeX file written to {filepath}")



    def __str__(self):
        return f"Problem: {self.name}, Train Data: {self.train_data[0].shape}, Test Data: {self.test_data[0].shape}, Solutions: {len(self.solutions)}, Solve_state: {self.solve_state}"


class hyper_parameter_search:
    def __init__(self, problem):
        self.problem = problem
        self.results = []
    
    def run_search_epsilon(self):
        def objective(trial):
            self.problem.solutions = []
            epsilon = trial.suggest_float('epsilon', 1e-6, 1e-2)
            self.problem.args['epsilon'] = epsilon
            self.problem.solve()

            accuracy = sum(solution.mse for solution in self.problem.solutions) / len(self.problem.solutions)

            return accuracy

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        return study.best_params
    

    def run_search_population_size(self):
        def objective(trial):
            self.problem.solutions = []
            population_size = trial.suggest_int('population_size', 50, 500)
            self.problem.args['population_size'] = population_size
            self.problem.solve()

            accuracy = sum(solution.mse for solution in self.problem.solutions) / len(self.problem.solutions)

            return accuracy

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)

        return study.best_params


