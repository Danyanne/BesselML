"""
BesselML: A Symbolic Regression Framework for Mathematical Functions

This module implements a framework for performing symbolic regression on mathematical
functions, with a particular focus on special functions like hypergeometric functions.
It provides classes for defining regression problems, finding solutions, and analyzing
results through various visualization methods.

The framework uses the Operon symbolic regression engine and includes capabilities for
parameter optimization, LaTeX export, and detailed analysis of regression results.

Done by Daniel C. Summer 2025
"""

# Scientific and numerical computing
import numpy as np
from scipy import special

# Symbolic mathematics and formatting
import sympy as sp
from decimal import Decimal

# Machine learning and optimization
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
import optuna

# Visualization and display
import matplotlib.pyplot as plt
from IPython.display import display, Math

# System and utilities
import os
import re
from typing import Tuple, Dict

class Solution:
    """
    Represents a solution found by symbolic regression.
    
    This class encapsulates a symbolic regression solution, including the mathematical
    expression, its performance metrics, and methods for visualization and analysis.

    Attributes:
        name (str): Identifier for the solution
        problem (Problem): Reference to the parent Problem instance
        model: The trained model
        tree: The expression tree representation
        string_expression (str): String representation of the mathematical expression
        r2 (float): R-squared score of the solution
        mse (float): Mean squared error
        mdl (float): Minimum description length
        regressor: The symbolic regressor instance
        length (int): Length/complexity of the expression
    """
    def __init__(self, name, problem, model, tree, string_expression, r2, mse, mdl, regressor, length):
        """
        Initialize a Solution instance with regression results and metrics.
        
        Args:
            name (str): Identifier for the solution
            problem (Problem): Reference to the parent Problem instance
            model: The trained model
            tree: The expression tree representation
            string_expression (str): String representation of the mathematical expression
            r2 (float): R-squared score
            mse (float): Mean squared error
            mdl (float): Minimum description length
            regressor: The symbolic regressor instance
            length (int): Length/complexity of the expression
        """
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
        self.sympy_expr = None
    






    def extract_and_format(self, variable_prefix: str = 'X') -> Tuple[str, Dict[str, float], sp.Expr]:
            """
            Parses the symbolic expression string stored in `self.string_expression`, replacing
            only "complex" numerical constants with symbolic parameters `b0`, `b1`, ..., and
            returns a LaTeX-formatted string along with a dictionary of these parameters.

            [Original docstring content preserved...]
            """
            
            expr_str = self.string_expression
            
            # Extract variable names more efficiently
            var_pattern = rf'{re.escape(variable_prefix)}\d+'
            var_names = sorted(set(re.findall(var_pattern, expr_str)), 
                            key=lambda x: int(x[len(variable_prefix):]))
            

            # Build local dictionary with only needed symbols
            local_dict = {name: sp.Symbol(name) for name in var_names}


            
            # Additional common functions that might be needed
            local_dict.update({
                "sqrt": sp.sqrt,
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
            

            
            # Collect all complex constants in a consistent order
            complex_constants = []
            # seen = set()



            def get_ordered_constants_positive(expanded_expr: sp.Expr) -> list[sp.Float]:
                """
                Extract numeric constants in left-to-right order by magnitude only (positive values).
                Signs are handled by the expression itself.
                """
                expr_str = sp.ccode(expanded_expr)  # sign-safe form for regex scanning

                number_pattern = r'(?<![\w.])([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)(?![\w.])'
                matches = list(re.finditer(number_pattern, expr_str))

                seen = set()
                constants_ordered = []

                for match in matches:
                    raw_val = match.group(1)
                    try:
                        dec = Decimal(raw_val)
                        # Always take absolute value for storage
                        dec_abs = abs(dec)
                        dec_str = str(dec_abs.normalize())
                        if '.' in dec_str:
                            decimal_part = dec_str.split('.')[-1].rstrip('0')
                            is_complex = len(decimal_part) > 2
                        else:
                            is_complex = False

                        if is_complex:
                            float_val = float(dec_abs)
                            if float_val not in seen:
                                seen.add(float_val)
                                constants_ordered.append(sp.Float(float_val, 12))
                                
                    except Exception:
                        continue

                return constants_ordered

            
            # collect_constants(expanded)
            complex_constants = get_ordered_constants_positive(expanded)


            rounded_expanded = expanded.xreplace({
                f: sp.Float(f, 12)
                for f in expanded.atoms(sp.Float)
            })

            replacement_pairs = [(const, sp.Symbol(f"b{i}")) for i, const in enumerate(complex_constants)]


            #expr_with_b = expanded.replace(lambda expr: expr.is_Number, replacer)
            expr_with_b = rounded_expanded.subs(replacement_pairs) 

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
            
            self.sympy_expr = expr_with_b
            
            return formatted_str, b_vals, expr_with_b





    def display_expression(self):
        """
        Display the mathematical expression in LaTeX format using IPython's Math display.
        
        Returns:
            IPython.display.Math: The formatted mathematical expression for display in notebooks.
        """
        formatted_expr, b_vals, sympy_expr = self.extract_and_format(self.string_expression)
        return display(Math(sp.latex(sympy_expr)))


    def plot_results(self, ax=None, train=True):
        """
        Plot the regression results against true values.
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, creates new figure.
            train (bool): If True, plots training data, otherwise test data.
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        # Get appropriate dataset based on train flag
        x, y = self.problem.train_data if train else self.problem.test_data
        y_pred = self.regressor.evaluate_model(self.tree, x.reshape(-1, 1))

        # Create new figure if no axes provided
        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.plot(x, y, label='True', color='black')
        ax.plot(x, y_pred, label=f'Prediction ({self.name})', linestyle='--')
        
        ax.set_title(f'{self.name} {"Train" if train else "Test"}')
        ax.legend()
        return ax


    def plot_residuals(self, ax=None, train=True):
        """
        Plot the residuals (differences between predicted and actual values) of the model.
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, creates new figure.
            train (bool): If True, uses training data, otherwise uses test data.
            
        Returns:
            matplotlib.axes.Axes: The axes containing the residuals plot.
        """
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
    
    def plot_fractional_error_hypergeom(self, x_val, coeff, ax=None):
        """
        Plot the fractional error of the regression against the hypergeometric function.
        
        This method compares the predicted values against the true hypergeometric function
        values and plots the fractional error on a log-log scale. It also highlights
        cosmologically relevant regions and the training range.
        
        Args:
            x_val (numpy.ndarray): X values for evaluation
            ax (matplotlib.axes.Axes, optional): The axes to plot on
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        # Calculate predicted and true values
        y_pred = self.regressor.evaluate_model(self.tree, x_val.reshape(-1, 1))
        y = special.hyp2f1(coeff[0], coeff[1], coeff[2], x_val)

        # Calculate fractional error
        fractional_error = np.abs((y - y_pred) / y)

        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.plot(abs(x_val), fractional_error, label=r'$\!{}_2F_1\left(\frac{1}{3},1,\frac{11}{6};x\right)$', linestyle='--', color='tab:blue')

        # Log-log scale
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Axes labels and title
        ax.set_xlabel(r'$-x$')
        ax.set_ylabel('Fractional Error')

        # Highlight relevant regions
        # Red-hatched: cosmologically relevant range (10^{-7} to 10^{0})
        ax.axvspan(1e-7, 9.0, facecolor='none', hatch='////', edgecolor='red', linewidth=0.0, zorder=0, label='Relevant for Cosmology')

        # Grey: training range (10^{-1} to 10^{1})
        ax.axvspan(0.016, 9.0, color='gray', alpha=0.3, label='Training range', zorder=0)

        # Legend and tight layout
        ax.legend(loc='lower right', frameon=True)
        ax.set_ylim(1e-12, 1e3)  # Match y-range in original figure
        ax.grid()

        return ax



    def to_latex(self):
        """
        Convert the solution's expression to LaTeX format.
        
        This method converts the symbolic expression to LaTeX format and returns both
        the formatted expression and a dictionary of parameter values.
        
        Returns:
            Tuple[str, Dict[str, float]]: A tuple containing:
                - latex_expr (str): The LaTeX-formatted expression
                - b_vals (Dict[str, float]): Dictionary mapping parameter names to their values
        """
        formatted_expr, b_vals, sympy_expr = self.extract_and_format(self.string_expression)
        latex_expr = sp.latex(sympy_expr)
        return latex_expr, b_vals
 

    def perform_functional_analysis(self):
        """
        Perform a functional analysis of the solution. Computes the relevant limits, and derivatives
        """

        formatted_expr, b_vals, sympy_expr = self.extract_and_format(self.string_expression)


        # Compute limits at 0 and infinity
        limit_at_zero = sp.limit(sympy_expr, sp.Symbol('X1'), 0)
        limit_at_infinity = sp.limit(sympy_expr, sp.Symbol('X1'), sp.oo)

        # Compute first and second derivatives
        first_derivative = sp.diff(sympy_expr, sp.Symbol('X1'))
        second_derivative = sp.diff(first_derivative, sp.Symbol('X2'))

        return {
            'limit_at_zero': limit_at_zero,
            'limit_at_infinity': limit_at_infinity,
            'first_derivative': first_derivative,
            'second_derivative': second_derivative
        }
    
    def compute_limits(self, at_value):
        """
        Compute the limit of the symbolic expression at a given value.
        
        Args:
            at_value (float): The value at which to compute the limit
            
        Returns:
            sp.Expr: The computed limit
        """
        x = sp.Symbol('X1')
        expr = sp.sympify(self.string_expression)
        return sp.limit(expr, x, at_value)
    

    
    def __str__(self):
        return (f"{self.name}: expr={self.string_expression}, "
            f"R²={self.r2:.4f}, MSE={self.mse:.4f}, MDL={self.mdl:.2f}")
    

    
    






class Problem:
    """
    Represents a symbolic regression problem with associated data and solutions.
    
    This class manages the symbolic regression problem, including training and test data,
    hyperparameters, and found solutions. It provides methods for solving the regression
    problem and visualizing results.

    Attributes:
        name (str): Name identifier for the problem
        train_data (tuple): Tuple of (X, y) arrays for training
        test_data (tuple): Tuple of (X, y) arrays for testing
        args (dict): Dictionary of hyperparameters for the symbolic regressor
        solutions (list): List of Solution objects found
        symbolic_regressor: The SymbolicRegressor instance
        solve_state (bool): Whether the problem has been solved
    """
    def __init__(self, name, train_data, test_data, args):
        """
        Initialize a Problem instance.
        
        Args:
            name (str): Name identifier for the problem
            train_data (tuple): Tuple of (X, y) arrays for training
            test_data (tuple): Tuple of (X, y) arrays for testing
            args (dict): Dictionary of hyperparameters for the symbolic regressor
        """
        self.name = name
        self.train_data = train_data
        self.test_data = test_data
        self.args = args
        self.solutions = []
        self.symbolic_regressor = None
        self.solve_state = False
    

    def add_solution(self, solution):
        """
        Add a solution to the problem's collection of solutions.
        
        Args:
            solution (Solution): A Solution instance to add to the collection
        """
        self.solutions.append(solution)

    def plot_data(self, ax=None, train=True):
        """
        Plot the training or test data of the problem.
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, creates new figure.
            train (bool): If True, plots training data, otherwise test data.
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
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
        """
        Solve the symbolic regression problem using the configured parameters.
        
        This method initializes and runs the SymbolicRegressor with the specified
        hyperparameters, then processes and stores the solutions found in the Pareto front.
        The solutions are stored as Solution objects in the solutions list.
        
        The method uses all parameters provided in the args dictionary to initialize
        the SymbolicRegressor. Any valid parameter for SymbolicRegressor can be included
        in the args dictionary and it will be passed to the constructor.
        """
        # Filter out any None values from args to avoid passing undefined parameters
        valid_args = {k: v for k, v in self.args.items() if v is not None}
        
        # Initialize SymbolicRegressor with all provided arguments
        reg = SymbolicRegressor(**valid_args)

        if self.train_data[0].ndim == 1:
            reg.fit(self.train_data[0].reshape(-1, 1), self.train_data[1].ravel())
        else:
            reg.fit(self.train_data[0], self.train_data[1].ravel())
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
            solution.extract_and_format()  # Ensure sympy_expr is set

            self.add_solution(solution)


    def plot_l_vs_mse(self, ax=None):
        """
        Plot the relationship between expression length and Mean Squared Error for all solutions.
        
        This method creates a scatter plot showing the trade-off between model complexity
        (expression length) and accuracy (MSE) for all solutions in the Pareto front.
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, creates new figure.
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        lengths = [s.length for s in self.solutions]
        mse = [s.mse for s in self.solutions]

        ax.scatter(lengths, mse, color='blue')
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
    """
    A class for performing hyperparameter optimization for symbolic regression problems.
    
    This class uses Optuna to perform hyperparameter optimization for various parameters
    of the symbolic regression process, such as epsilon and population size.
    
    Attributes:
        problem (Problem): The symbolic regression problem to optimize
        results (list): List to store optimization results
    """
    def __init__(self, problem):
        """
        Initialize the hyperparameter search.
        
        Args:
            problem (Problem): The symbolic regression problem to optimize
        """
        self.problem = problem
        self.results = []
    
    def run_search_epsilon(self):
        """
        Perform hyperparameter optimization for the epsilon parameter.
        
        This method uses Optuna to find the optimal value for the epsilon parameter,
        which controls the convergence threshold in symbolic regression.
        
        Returns:
            dict: The best parameters found during optimization
        """
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
        """
        Perform hyperparameter optimization for the population size parameter.
        
        This method uses Optuna to find the optimal value for the population size,
        which controls the number of individuals in each generation during evolution.
        
        Returns:
            dict: The best parameters found during optimization
        """
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


