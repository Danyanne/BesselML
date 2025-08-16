"""
BesselML: A Symbolic Regression Framework for Mathematical Functions

This module implements a framework for performing symbolic regression on mathematical
functions, with a particular focus on special functions like hypergeometric functions.
It provides classes for defining regression problems, finding solutions, and analyzing
results through various visualization methods. Provides additional functions for analysis and 
reoptimization of symbolic expressions.


Done by Daniel C. Summer 2025 at Oxford University, UK.
"""


import numpy as np
from scipy import special
import sympy as sp
from decimal import Decimal
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from typing import Dict, List, Any, Tuple, Optional
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
import optuna
import matplotlib.pyplot as plt
from IPython.display import display, Math
import os
import re
from typing import Tuple, Dict
from scipy.optimize import minimize



def create_arbitrary_constraint(expression_string: str, 
                            target_value: float, 
                            full_param_symbols: List[sp.Symbol],
                            constraint_type: str = 'eq') -> Dict[str, Any]:
    """Creates a single, correctly formatted and numerically robust constraint.

    This helper function takes a user-defined mathematical expression as a string
    and converts it into a constraint dictionary. The function it returns is
    "defensive," meaning it will catch numerical errors (inf/NaN) and return a
    large penalty, preventing the optimizer from crashing.

    Args:
        expression_string (str):
            The mathematical condition, e.g., "b1 + b2" or "b3 / b4".
        target_value (float):
            The value the expression should be constrained against.
        full_param_symbols (List[sp.Symbol]):
            The complete, ordered list of sympy.Symbol objects being used in the
            main optimization routine.
        constraint_type (str, optional):
            The type of constraint, 'eq' (equality) or 'ineq' (inequality).
            Defaults to 'eq'.

    Returns:
        Dict[str, Any]: A constraint dictionary ready for `scipy.optimize.minimize`.
    """
    if constraint_type not in ['eq', 'ineq']:
        raise ValueError("constraint_type must be either 'eq' or 'ineq'.")
        
    local_symbol_dict = {str(s): s for s in full_param_symbols}
    parsed_expr = sp.parse_expr(expression_string, local_dict=local_symbol_dict)
    constraint_expr = parsed_expr - target_value
    constraint_lambda = sp.lambdify(full_param_symbols, constraint_expr, 'numpy')
    
    # --- DEFENSIVE CONSTRAINT FUNCTION ---
    # This wrapper function will be the actual constraint passed to the optimizer.
    def defensive_fun(b_values):
        try:
            # Use errstate to treat numpy warnings as errors
            with np.errstate(all='raise'):
                # Calculate the constraint violation
                violation = constraint_lambda(*b_values)
                
                # If the result is not a finite number, this region is invalid.
                if not np.isfinite(violation):
                    # For constraints, returning a large value indicates a large violation.
                    return 1e12 
                
                return float(violation)

        except (FloatingPointError, ValueError, ZeroDivisionError):
            # If any numerical error occurs, penalize this parameter set heavily.
            return 1e12

    # Return the final constraint dictionary with the robust function.
    return {
        'type': constraint_type,
        'fun': defensive_fun
    }



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
        b_vals (Dict[str, float]): Dictionary of parameter values extracted from the expression
    """
    def __init__(self, name, problem, model, tree, string_expression, r2, mse, mdl, regressor, length):
        """
        Initialize a Solution instance with regression results and metrics.
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
        self.b_vals = None 
    
    def extract_and_format(self, variable_prefix: str = 'X') -> Tuple[str, Dict[str, float], sp.Expr]:
            """
            Parses the symbolic expression string stored in `self.string_expression`, replacing
            only "complex" numerical constants with symbolic parameters `b0`, `b1`, ..., and
            returns a LaTeX-formatted string along with a dictionary of these parameters and sympy expression.
            """
            
            expr_str = self.string_expression
            
            # Extract variable names more efficiently
            var_pattern = rf'{re.escape(variable_prefix)}\d+'
            var_names = sorted(set(re.findall(var_pattern, expr_str)), 
                            key=lambda x: int(x[len(variable_prefix):]))
            

            # Build local dictionary with only needed symbols
            local_dict = {name: sp.Symbol(name) for name in var_names}


            
            # Additional common functions that might be needed, defined using sympy
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

            # Parse the expression string using sympy
            try:
                expr = sp.sympify(expr_str, locals=local_dict)
                expanded = sp.expand(expr)
            except (sp.SympifyError, ValueError) as e:
                raise ValueError(f"Failed to parse expression: {e}")
            

            # Collect all complex constants in a consistent order
            complex_constants = []

            def get_ordered_constants_positive(expanded_expr: sp.Expr) -> list[sp.Float]:
                """
                Extract numeric constants in left-to-right order by magnitude only (positive values).
                Signs are handled by the expression itself, all values are therefore stored as positive.
                """

                 # sign-safe form for regex scanning
                expr_str = sp.ccode(expanded_expr) 

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
                        # Check if the decimal part has more than 2 digits - to adrress the complexity, which 
                        # in this context is defined as the number of digits after the decimal point

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

            complex_constants = get_ordered_constants_positive(expanded)
            # Rounding consistent with get_ordered_constants_positive function, because later on 
            # the precise values of the parameters need to match to subsitute b_i symbols instead of them

            rounded_expanded = expanded.xreplace({
                f: sp.Float(f, 12)
                for f in expanded.atoms(sp.Float)
            })

            # Create replacement pairs for complex constants
            replacement_pairs = [(const, sp.Symbol(f"b{i}")) for i, const in enumerate(complex_constants)]

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
            self.b_vals = b_vals  # Store the parameter values for later use
            
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

        # Handle single-dimensional input --> reshape to 2D array. If already 2D, use as is
        if x.ndim == 1:
            x_in = x.reshape(-1, 1)
        else:
            x_in = x

        y_pred = self.regressor.evaluate_model(self.tree, x_in)


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

                # Handle single-dimensional input --> reshape to 2D array. If already 2D, use as is
        if x.ndim == 1:
            x_in = x.reshape(-1, 1)
        else:
            x_in = x

        y_pred = self.regressor.evaluate_model(self.tree, x_in)
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
        ax.set_xlim(1e-7,1e7)
        ax.grid()

        return ax

    def plot_fractional_error_bessel(self, x_val, order, kind = 'first', spherical=True, ax=None):
        """
        Plot the fractional error of the regression against the Bessel spherical function.
        
        This method compares the predicted values against the true Bessel function
        values and plots the fractional error on a log-log scale.
        
        Args:
            x_val (numpy.ndarray): X values for evaluation
            ax (matplotlib.axes.Axes, optional): The axes to plot on
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        # Calculate predicted and true values
        y_pred = self.regressor.evaluate_model(self.tree, x_val.reshape(-1, 1))
        
        if kind == 'second' and spherical == False:
            y = special.yv(order, x_val)
        elif kind == 'second' and spherical == True:
            y = special.spherical_yn(order, x_val)
        elif kind == 'first' and spherical == True:
            y = special.spherical_jn(order, x_val)
            print(f"spherical bessel function of first kind of order {order}")
        else:
            y = special.jv(order, x_val)

        # Calculate fractional error
        fractional_error = np.abs((y - y_pred) / y)

        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.plot(abs(x_val), fractional_error, linestyle='--', color='tab:blue')

        # Log-log scale
        # ax.set_xscale('log')
        ax.set_yscale('log')

        # Axes labels and title
        ax.set_xlabel(r'$-x$')
        ax.set_ylabel('Fractional Error')


        # Legend and tight layout
        #ax.set_ylim(1e-12, 1e3)  # Match y-range in original figure
        #ax.set_xlim(1e-7,1e7)
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

    def int_analysis_and_modification(self, threshold: float = 0.1):
        """
        Analyzes the solution's parameters to perform 'smart rounding' and
        creates a new, re-optimized PromisingSolution if simplification is possible.

        This is a corrected and refined version of your provided function.
        """
        x_data, y_data = self.problem.test_data
        param_names = sorted(self.b_vals.keys()) # Use a sorted list for consistent order

        # Create a single, reusable numerical function from the expression
        f_lambdified = sp.lambdify(['X1'] + [sp.Symbol(name) for name in param_names], self.sympy_expr, modules='numpy')

        def objective_mse(b_values: np.ndarray) -> float:
            """Generic MSE objective function that takes a NumPy array."""
            return np.mean((y_data - f_lambdified(x_data, *b_values))**2)

        # --- Step 1: Establish the Baseline ---
        # The baseline is the best possible MSE with the full float model.
        # We assume self.b_vals contains the optimal float values.
        # If not, you would run minimize() here first.
        params_best_float = np.array([self.b_vals[name] for name in param_names])
        baseline_mse = objective_mse(params_best_float)

        # --- Step 2: Calculate the Cost of Rounding for Each Parameter ---
        substite_param = {} # Params that will be rounded
        b_vals_new = {}     # Params that will remain floats

        for i, name in enumerate(param_names):
            # Create a fresh copy of the optimal params for this test
            params_hybrid = np.copy(params_best_float)
            param_int = round(params_hybrid[i])
            params_hybrid[i] = param_int # Modify only the parameter being tested

            mse_hybrid = objective_mse(params_hybrid)
            
            # Calculate the penalty relative to our consistent baseline
            penalisation = (mse_hybrid - baseline_mse) / baseline_mse if baseline_mse > 1e-16 else float('inf')

            if penalisation < threshold:
                substite_param[name] = param_int
            else:
                b_vals_new[name] = self.b_vals[name]
        
        # --- Step 3: Create and Re-optimize a New Solution ---
        if not substite_param: # Check if the dictionary is empty
            print("  - No parameters met the rounding criteria. Keeping full float model.")
            return None # Return nothing if no changes were made
 
        print(f"  - Rounding parameters: {list(substite_param.keys())}")
        
        # Create the new expression with the integer values substituted in
        expression_w_int = self.sympy_expr.subs(substite_param)
        
        # Create the new solution object. The initial parameters for its optimization
        # will be only the ones that remained floats.
        Solution_w_integers = Promising_solution(
            expression_w_int,
            (x_data, y_data),
            self,
            initial_parameters=b_vals_new
        )
        
        print(f'Creating a new Promising solution: {Solution_w_integers.sympy_expr}')
        
        if b_vals_new:
            print(f'Re-optimizing the remaining float parameters: {list(b_vals_new.keys())}')

            # This assumes your Promising_solution class has a method to run optimization.
            # The optimization will now only work on the parameters defined in `b_vals_new`.
            Solution_w_integers.run_multiple_optimisations(
                n_runs=100,
                k_confirm=2,
                scatter_fraction=0.1,
                cluster_tolerance=1e-6
            )
            
        return Solution_w_integers

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
                            #(for k, v) construct k:v --> create the dictionary in this way  

        # Initialize SymbolicRegressor with all provided arguments
        reg = SymbolicRegressor(**valid_args)

        # Same condition as for the function solution.plot_results or .plot_residuals, identifying case for 
        # single-dimensional input and multiple-dimensional input
        if self.train_data[0].ndim == 1:
            reg.fit(self.train_data[0].reshape(-1, 1), self.train_data[1].ravel())  
        else:
            print("Training data has multiple dimensions, fitting directly.")
            reg.fit(self.train_data[0], self.train_data[1].ravel())

        res = [(s['objective_values'], s['tree'], s['minimum_description_length'], s['mean_squared_error']) for s in reg.pareto_front_]
        
        #printing the results in a formatted way
        for obj, expr, mdl, mse in res:
           print(f'{obj}, {mdl:.2f}, {reg.get_model_string(expr, 12)}, {mse:.2f}')
        
          
        self.symbolic_regressor = reg
        self.solve_state = True

        # Process the Pareto front solutions and create + store Solution objects
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
            # Ensure sympy_expr is set, the extract_and_format has implemented the logic to set it up and save 
            # to self.sympy_expr (as well as self.b_vals)
            solution.extract_and_format()  

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

        # Create a label for each point with its index (starting from 0)
        for i, (lx, my) in enumerate(zip(lengths, mse)):
            ax.text(lx, my * 1.4, str(i), fontsize=8, ha='center', va='bottom')

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
            body += f"\\subsection*{{Solution {i}}}\n"
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


class Promising_solution:
    """
    A class for handling promising symbolic regression solutions and their subsequent analysis.

    Attributes:
        expr (sp.Expr): The symbolic expression represented by this instance
        test_data (tuple): Tuple containing (x_array, y_array) for testing the expression
        initial_params (dict): Initial parameter values for optimization
        modified_parameters (dict): To store modified parameters after optimization
        numerical_expr (sp.Expr): To store the numerical expression after optimization
        solution (Solution): The original symbolic regression solution associated with this expression
    """
    def __init__(self, sympy_expr, test_data, solution, initial_parameters):
        """
        Initialize the expression with a sympy expression.
        """
                # --- Store core components ---
        self.sympy_expr = sympy_expr
        self.test_data = test_data
        self.original_solution = solution
        self.numerical_expr = None  # To store the numerical expression after optimization

        # --- Handle default parameter creation ---
        # Use a new variable to avoid the UnboundLocalError from shadowing the argument.
        params_to_process = initial_parameters

        # If no initial parameters are provided, create a default set.
        if params_to_process is None:
            print("No initial_parameters provided. Defaulting 'b*' params in expression to 1.0.")
            # Extract parameter names from expression starting with 'b'
            params_in_expr = [s.name for s in sympy_expr.free_symbols if s.name.startswith('b')]
            params_to_process = {p: 1.0 for p in params_in_expr}
        
        # --- SELF-CORRECTING PARAMETER LOGIC ---
        
        # 1. Get the set of all variable names (as strings) from the new expression.
        # We explicitly ignore 'X1' as it's the independent variable, not a parameter.
        symbols_in_expr = {str(s) for s in sympy_expr.free_symbols if str(s) != 'X1'}

        # 2. Filter the incoming parameter dictionary.
        # This ensures the parameters are always in sync with the expression.
        filtered_params = {
            key: value for key, value in params_to_process.items()
            if key in symbols_in_expr
        }

        # 3. Store the clean, filtered dictionary as the definitive set of parameters.
        # Your `optimisation` function should use this attribute.
        self.initial_params = filtered_params
        # Initialize modified_parameters with the same clean set. It will be updated by the optimizer.
        self.modified_parameters = filtered_params.copy() 

    def optimisation(self, initial_conditions: Optional[Dict[str, float]] = None, 
                 constraints_eq: Optional[List[Dict]] = None) -> Tuple[sp.Expr, Dict[str, float], Dict[str, float]]:
        """Runs a numerical optimization to fit the model's parameters to data.

        This method uses the 'trust-constr' algorithm to minimize a weighted 
        mean squared error (MSE) objective function. The objective function is
        designed to be "defensive," meaning it can handle numerical instabilities
        (like division by zero or NaN values) by returning a large penalty,
        allowing the optimizer to continue searching.

        The parameters to be optimized are derived from the keys of the
        `initial_conditions` dictionary.

        Args:
            initial_conditions (Optional[Dict[str, float]], optional):
                A dictionary mapping parameter names (str) to their initial
                numerical values (float). If None, defaults to `self.initial_params`.
            constraints_eq (Optional[List[Dict]], optional):
                A list of constraint dictionaries in the format required by
                `scipy.optimize.minimize`. Defaults to None.

        Returns:
            Tuple[sp.Expr, Dict[str, float], Dict[str, float]]: A tuple containing:
                - sp.Expr: The symbolic expression with the optimized numerical
                parameters substituted in.
                - Dict[str, float]: A dictionary of the optimized parameter values,
                keyed by parameter name (str).
                - Dict[str, float]: A dictionary of the absolute difference between
                the initial and optimized parameter values.
        """
        # --- SETUP AND VALIDATION ---
        if self.sympy_expr is None:
            raise ValueError("No symbolic expression to optimize.")

        # Default to the class's initial parameters if none are provided.
        if initial_conditions is None:
            initial_conditions = self.initial_params

        # --- ROBUST SYMBOL AND PARAMETER HANDLING ---
        # The parameters to optimize are defined by the keys of the initial_conditions dict.
        # This is the most robust way to prevent KeyErrors and symbol mismatches.
        sorted_param_names = sorted(initial_conditions.keys())
        param_syms = [sp.Symbol(name) for name in sorted_param_names]

        # Create the starting vector `b0` directly from the input dictionary.
        b0 = np.array([initial_conditions[name] for name in sorted_param_names])
        
        # Create a callable Python function from the symbolic expression.
        f_lambdified = sp.lambdify(['X1'] + param_syms, self.sympy_expr, modules='numpy')

        # Unpack test data for use in the objective function.
        X_data, Y_data = self.test_data
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        # --- DEFENSIVE OBJECTIVE FUNCTION ---
        def objective(b_values: np.ndarray) -> float:
            """
            Calculates the weighted MSE. Returns a large penalty on numerical error.
            """
            try:
                # Use errstate to treat numpy warnings (like invalid value) as errors.
                with np.errstate(all='raise'):
                    y_pred = f_lambdified(X_data, *b_values)
                    
                    # Explicitly check for NaNs or Infs which can crash the optimizer.
                    if not np.all(np.isfinite(y_pred)):
                        return 1e12 # Return a large penalty

                    # Weighting scheme to penalize errors more heavily at large X values.
                    weights = (X_data**2 + 1e-8)
                    sq_error = (y_pred - Y_data) ** 2
                    return np.mean(weights * sq_error)

            except (FloatingPointError, ValueError, ZeroDivisionError):
                # If any numerical error occurs, this parameter set is invalid.
                # Return a large penalty to guide the optimizer away.
                return 1e12

        # --- RUN OPTIMIZATION ---
        # Use the 'trust-constr' method, which is robust for complex problems.
        result = minimize(
            objective,
            b0,
            constraints=constraints_eq,
            method='trust-constr',
            options={'maxiter': 10000, 'verbose': 1}
        )

        if not result.success:
            print("Optimization did not converge:", result.message)

        # --- PROCESS AND RETURN RESULTS ---
        # Create a dictionary of optimized parameters with string keys.
        optimized_params_with_str_keys = {
            name: val for name, val in zip(sorted_param_names, result.x)
        }
        
        # Create a dictionary for substituting back into the SymPy expression.
        optimized_subs_with_sym_keys = {
            sym: val for sym, val in zip(param_syms, result.x)
        }

        # Update class attributes with the results.
        self.numerical_expr = self.sympy_expr.subs(optimized_subs_with_sym_keys)
        self.modified_parameters = optimized_params_with_str_keys

        # Calculate the absolute change for each parameter.
        abs_diff = {name: abs(result.x[i] - b0[i]) for i, name in enumerate(sorted_param_names)}

        # Print a summary if the helper method exists.
        if hasattr(self, '_print_summary_table'):
            self._print_summary_table(param_syms, b0, result.x, abs_diff)

        return self.numerical_expr, self.modified_parameters, abs_diff, result.fun

    def run_multiple_optimisations(
    self,
    n_runs: int = 20,
    k_confirm: int = 3,
    scatter_fraction: float = 0.1,
    cluster_tolerance: float = 1e-7,
    initial_conditions: Optional[Dict[str, float]] = None,
    constraints_eq: Optional[List[Dict]] = None
    ) -> Tuple[sp.Expr, Dict[str, float]]:
        """
        Performs multi-start optimization to find a robust set of parameters.

        This method addresses the sensitivity of optimization to initial conditions by
        running the optimizer multiple times from different, randomly scattered starting
        points. It then clusters the results and returns the solution from the most
        stable cluster that has the best objective score (lowest MSE).

        Args:
            n_runs (int): The total number of optimization runs to perform.
            k_confirm (int): The minimum number of results required in a cluster for
                it to be considered a stable, "confirmed" solution.
            scatter_fraction (float): The relative range (e.g., 0.1 for 10%) used
                to randomly scatter the initial parameter values for each run.
            cluster_tolerance (float): The relative tolerance used to determine if
                two sets of parameters are close enough to belong to the same cluster.
            initial_conditions (Optional[Dict[str, float]]): The central set of
                initial parameters to scatter from. Defaults to `self.initial_params`.
            constraints_eq (Optional[List[Dict]]): Constraints to be applied to all
                optimization runs.

        Returns:
            Tuple[sp.Expr, Dict[str, float]]: A tuple containing:
                - sp.Expr: The symbolic expression with the confirmed best parameters.
                - Dict[str, float]: The dictionary of the confirmed best parameters.

        Raises:
            RuntimeError: If no stable cluster of solutions can be found after all runs.
        """
        if initial_conditions is None:
            initial_conditions = self.initial_params

        all_results = []
        print(f"--- Starting Multi-Start Optimization ({n_runs} runs) ---")

        for i in range(n_runs):
            print(f"  Running optimization {i+1}/{n_runs}...")
            
            # Generate a new set of scattered initial conditions
            scattered_initials = {}
            for name, value in initial_conditions.items():
                # Avoid scattering zero or very small numbers in a multiplicative way
                if abs(value) < 1e-9:
                    scattered_initials[name] = value + (np.random.rand() - 0.5) * 2 * scatter_fraction
                else:
                    scattered_initials[name] = value * (1 + (np.random.rand() - 0.5) * 2 * scatter_fraction)

            try:
                # Call the (modified) single optimization function
                num_expr, opt_params, _, mse = self.optimisation(
                    initial_conditions=scattered_initials,
                    constraints_eq=constraints_eq
                )
                all_results.append({'params': opt_params, 'mse': mse, 'expr': num_expr})
            except Exception as e:
                print(f"    - Run {i+1} failed with an unexpected error: {e}")

        if not all_results:
            raise RuntimeError("All optimization runs failed. Unable to find a solution.")

        # --- Cluster the results to find stable solutions ---
        # Convert parameter dicts to sorted numpy arrays for reliable comparison
        for res in all_results:
            sorted_keys = sorted(res['params'].keys())
            res['params_array'] = np.array([res['params'][k] for k in sorted_keys])
        
        clusters = []
        for result in all_results:
            # Skip results with very high MSE (likely failed runs)
            if result['mse'] >= 1e10:
                continue
                
            found_cluster = False
            for cluster in clusters:
                # Check if this result is close to the representative of an existing cluster
                if np.allclose(result['params_array'], cluster[0]['params_array'], rtol=cluster_tolerance):
                    cluster.append(result)
                    found_cluster = True
                    break
            
            if not found_cluster:
                # Create a new cluster
                clusters.append([result])

        # --- Find the best result from the most stable clusters ---
        # Filter for clusters that meet the confirmation threshold `k_confirm`
        confirmed_clusters = [c for c in clusters if len(c) >= k_confirm]

        if not confirmed_clusters:
            raise RuntimeError(
                f"Optimization inconclusive. No stable parameter set was found {k_confirm} or more times."
            )

        # From the confirmed clusters, find the one containing the overall best solution (lowest MSE)
        best_overall_result = None
        best_mse = float('inf')

        for cluster in confirmed_clusters:
            # Find the best result within this cluster
            best_in_cluster = min(cluster, key=lambda x: x['mse'])
            if best_in_cluster['mse'] < best_mse:
                best_mse = best_in_cluster['mse']
                best_overall_result = best_in_cluster

        print("\n--- Multi-Start Optimization Summary ---")
        print(f"Found {len(confirmed_clusters)} stable cluster(s) with at least {k_confirm} members.")
        print(f"Best solution found with MSE = {best_overall_result['mse']:.6g}")
        print("Confirmed parameters:")
        for name, val in best_overall_result['params'].items():
            print(f"  {name}: {val}")

        # Update class attributes with the best confirmed result
        self.numerical_expr = best_overall_result['expr']
        self.modified_parameters = best_overall_result['params']

        return self.numerical_expr, self.modified_parameters

    def _print_summary_table(self, param_syms: List[sp.Symbol], old_vals: np.ndarray, 
                            new_vals: np.ndarray, abs_diff: Dict[str, float]) -> None:
        """Prints a formatted summary table of the optimization results.

        This is a private helper method used to display a clear, aligned table
        comparing the initial and final parameter values.

        Args:
            param_syms (List[sp.Symbol]): The ordered list of sympy.Symbol objects.
            old_vals (np.ndarray): The initial parameter values before optimization.
            new_vals (np.ndarray): The final, optimized parameter values.
            abs_diff (Dict[str, float]): A dictionary mapping parameter names to the
                                        absolute difference between old and new values.
        
        Returns:
            None
        """
        # Prepare all rows for the table, including headers.
        rows = []
        rows.append(['Parameter', 'Old Value', 'New Value', 'Abs Difference'])
        rows.append(['-'*9, '-'*9, '-'*9, '-'*14])
        for i, p in enumerate(param_syms):
            rows.append([
                str(p),
                f"{old_vals[i]:.6g}",
                f"{new_vals[i]:.6g}",
                f"{abs_diff[str(p)]:.6g}"
            ])

        # Dynamically compute the required width for each column for clean alignment.
        col_widths = [max(len(row[i]) for row in rows) for i in range(4)]

        # Build the final table string line by line.
        table_str = ""
        for row in rows:
            line = " | ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row))
            table_str += line + "\n"

        print("\nOptimization summary:")
        print(table_str)

    def compute_expansion_at_val(self, at_value, n=2):
        if self.sympy_expr is None:
            raise ValueError("No symbolic expression available for evaluation.")

        expr = self.sympy_expr
        X1 = sp.Symbol('X1', negative=True)

        # Substitute X1 with negative assumption
        # This ensures that the series expansion is computed correctly
        # for the negative domain, which is crucial for the hypergeometric function
        # and other expressions that may have different behavior for negative inputs.
        # All other symbols are assumed positive.
        positive_subs = {}
        for sym in expr.free_symbols:
            if sym.name == 'X1':
                # Use X1 with negative assumption
                positive_subs[sym] = X1
            else:
                # All other symbols positive
                positive_subs[sym] = sp.Symbol(sym.name, positive=True)

        expr_signed = expr.subs(positive_subs)

        return sp.series(expr_signed, X1, at_value, n).removeO()

    def compute_limits(self, at_value):
        """
        Compute the limit of the symbolic expression at a given value.
        
        Args:
            at_value (float): The value at which to compute the limit
            
        Returns:
            sp.Expr: The computed limit
        """
        x = sp.Symbol('X1')
        return sp.limit(self.sympy_expr, x, at_value)
    
    def plot_comparison(self, ax=None, train=True):
        """
        Plot the regression results against true values.
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, creates new figure.
            train (bool): If True, plots training data, otherwise test data.
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        # Get appropriate dataset based on train flag
        x, y = self.test_data

                # Calculate predicted and true values 
        X1 = sp.Symbol('X1')

        # Lambdify symbolic expression for fast evaluation
        f_lambdified = sp.lambdify(X1, self.numerical_expr, modules='numpy')

        # Evaluate predicted values from symbolic expression at x_val
        y_pred = f_lambdified(x)
        


        # Create new figure if no axes provided
        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.plot(x, y,  label = 'ground truth', color='black')
        ax.plot(x, y_pred, label='prediction of solution', linestyle='--')
        ax.legend()
        return ax
    
    def plot_fractional_error_hypergeom(self, x_val, coeff, ax=None):
        """
        Plot the fractional error of the symbolic expression against the hypergeometric function.

        Args:
            x_val (numpy.ndarray): X values for evaluation (assumed negative values)
            coeff (tuple/list): Parameters of the hypergeometric function (a, b, c)
            ax (matplotlib.axes.Axes, optional): The axes to plot on

        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """

        # Theoretical constant for the hypergeometric function, concretely its expansion as x-->inf,
        # then it behaves like a const_hypergeom times x^(-1/3) for large negative x.
        const_hypergeom = 1.437283088460994741640842 
        # Prepare symbolic variable
        X1 = sp.Symbol('X1')

        # Lambdify symbolic expression for fast evaluation
        f_lambdified = sp.lambdify(X1, self.numerical_expr, modules='numpy')

        # Evaluate predicted values from symbolic expression at x_val
        y_pred = f_lambdified(x_val)

        # Evaluate true hypergeometric values
        y_true = special.hyp2f1(coeff[0], coeff[1], coeff[2], x_val)

        # Compute fractional error (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            fractional_error = np.abs((y_true - y_pred) / (y_true + 1e-9))
            fractional_error = np.nan_to_num(fractional_error, nan=0.0, posinf=0.0, neginf=0.0)

        # Create figure and axis if not provided
        fig, ax = (plt.subplots() if ax is None else (None, ax))

        #Plot fractional error vs absolute x (x assumed negative)
        ax.plot(np.abs(x_val), fractional_error,
               label=r'Fractional error vs $_2F_1\left(\frac{1}{3},1,\frac{11}{6};x\right)$',
               linestyle='--', color='tab:blue')

        # Code for theoretical limiting behavior, if needed

        # limiting_beh = const_hypergeom/(np.abs(x_val)**(1/3))
        # ax.plot(np.abs(x_val), np.abs(limiting_beh - y_true) / np.abs(y_true),
        #         label='Limiting behavior', linestyle='-', color='green')


        # Log-log scales
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Labels and title
        ax.set_xlabel(r'$-x$')
        ax.set_ylabel('Fractional Error')

        # Highlight cosmologically relevant range: 1e-7 to 9.0 (red hatch) - This range adopted from Bartlett paper - emulators
        # for linear power spectrum

        ax.axvspan(1e-7, 9.0, facecolor='none', hatch='////', edgecolor='red',
                linewidth=0.0, zorder=0, label='Relevant for Cosmology')

        # Highlight training range: 0.016 to 9.0 (grey shaded)
        ax.axvspan(0.016, 9.0, color='gray', alpha=0.3,
                label='Training range', zorder=0)

        # Legend, grid, y-limits
        ax.legend(loc='lower right', frameon=True)
        #ax.set_xlim(1e-7, 1e7)
        ax.set_ylim(1e-16, 1e-1)
        ax.grid(True)

        return ax

    def plot_fractional_error_bessel(self, x_val, order, kind = 'first', spherical=True, ax=None):
        """
        Plot the fractional error of the regression against the Bessel spherical function.
        
        This method compares the predicted values against the true Bessel function
        values and plots the fractional error on a log-log scale.
        
        Args:
            x_val (numpy.ndarray): X values for evaluation
            ax (matplotlib.axes.Axes, optional): The axes to plot on
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        # Calculate predicted and true values
        X1 = sp.Symbol('X1')

        # Lambdify symbolic expression for fast evaluation
        f_lambdified = sp.lambdify(X1, self.numerical_expr, modules='numpy')

        # Evaluate predicted values from symbolic expression at x_val
        y_pred = f_lambdified(x_val)
        
        if kind == 'second' and spherical == False:
            y = special.yv(order, x_val)
        elif kind == 'second' and spherical == True:
            y = special.spherical_yn(order, x_val)
        elif kind == 'first' and spherical == True:
            y = special.spherical_jn(order, x_val)
            print(f"spherical bessel function of first kind of order {order}")
        else:
            y = special.jv(order, x_val)

        # Calculate fractional error
        fractional_error = np.abs((y - y_pred) / y)

        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.plot(abs(x_val), fractional_error, linestyle='--', color='tab:blue')

        # Log-log scale
        # ax.set_xscale('log')
        ax.set_yscale('log')

        # Axes labels and title
        ax.set_xlabel(r'$-x$')
        ax.set_ylabel('Fractional Error')


        # Legend and tight layout
        #ax.set_ylim(1e-12, 1e3)  # Match y-range in original figure
        #ax.set_xlim(1e-7,1e7)
        ax.grid()

        return ax
      
    def generate_constraints_from_expansion(self, var=sp.Symbol('X1'), const_target=1, linear_target=sp.Rational(2, 11)):
        """
        Generates optimization constraints based on the Taylor series expansion of the model.

        This self-contained version determines the full list of parameters internally
        from `self.initial_params` to ensure compatibility with the optimizer.

        Args:
            var (sp.Symbol): The variable to expand around (e.g., X1).
            const_target (float): The target value for the constant term (c0).
            linear_target (float): The target value for the linear coefficient (c1).

        Returns:
            tuple: A tuple containing (list of constraint dicts, const_term_expr, linear_coeff_expr).
        """
        if self.sympy_expr is None:
            raise ValueError("Symbolic expression `self.sympy_expr` must be set first.")
        if not hasattr(self, 'initial_params'):
            raise AttributeError("`self.initial_params` dictionary must exist to determine the full parameter list.")

        # 1. Create the full, ordered list of parameter symbols from `self.initial_params`.
        # This is the crucial step to ensure the list matches the one used in the main
        # optimization routine, preventing argument count mismatches.
        sorted_param_names = sorted(self.initial_params.keys())
        full_param_symbols = [sp.Symbol(name) for name in sorted_param_names]

        # 2. Compute the series expansion around var=0 up to a sufficient order (n=2 for linear term).
        series_expr = self.sympy_expr.series(var, 0, 2).removeO()

        # 3. Robustly extract the constant and linear coefficients.
        const_term = series_expr.subs(var, 0)
        linear_coeff = series_expr.coeff(var, 1)

        # 4. Create the constraint dictionaries for the optimizer.
        # Use the internally generated `full_param_symbols` in the lambdify call.
        # This creates a function that accepts all parameters from the optimizer.
        # constr_target and linear target are theoretically derived to match limiting behaviour
        # of hypergeometric function around points of interest (0, and in the limit x --> inf)
        constraints = [
            {
                'type': 'eq',
                'fun': lambda b, expr=const_term: float(sp.lambdify(full_param_symbols, expr)(*b) - const_target)
            },
            {
                'type': 'eq',
                'fun': lambda b, expr=linear_coeff: float(sp.lambdify(full_param_symbols, expr)(*b) - linear_target)
            }
        ]

        return constraints, const_term, linear_coeff

    def log_derivative(self, var=sp.Symbol('X1')):
        """
        Compute the symbolic derivative of log(expr) with respect to var.
        
        Parameters:
            expr (sympy.Expr): The expression to differentiate.
            var (sympy.Symbol): The variable to differentiate with respect to.
        
        Returns:
            sympy.Expr: The symbolic derivative.
        """
        return sp.simplify(sp.diff(sp.log(self.sympy_expr), var))

    def compute_expansion_at_infinity(self):
        """
        Compute the series expansion of the symbolic expression at infinity.
        
        This method computes the series expansion of the symbolic expression
        at infinity (X1 -> oo) and returns the simplified expression (without other terms).
        
        Returns:
            sp.Expr: The series expansion at infinity.
        """
        X1 = sp.Symbol('X1', negative=True)
        expr = self.sympy_expr.subs({s: X1 for s in self.sympy_expr.free_symbols if s.name == 'X1'})

        # Substitute X1 with 1/X1 and expand around 0, common method to find 
        # series expansion at infinity (we take just first two orders)
        expr_subs = expr.subs(X1, 1/X1)
        return sp.series(expr_subs, X1, 0, 2).removeO()    

    def int_analysis_and_modification(self, threshold: float = 0.01, abs_mse_tol: float = 1e-18):
        """
        Analyzes the solution's parameters to perform 'smart rounding' and
        creates a new, re-optimized PromisingSolution if simplification is possible.

        This version automatically detects which parameters are present in the
        expression and only analyzes them.
        """
        # --- Setup: Identify Active Parameters ---
        x_data, y_data = self.original_solution.problem.test_data
        
        # Automatically detect which parameters are actually in the expression
        all_symbols_in_expr = self.sympy_expr.free_symbols
        all_param_names = self.modified_parameters.keys()
        
        active_param_symbols = [s for s in all_symbols_in_expr if s.name in all_param_names]
        active_param_names = sorted([s.name for s in active_param_symbols])

        if not active_param_names:
            print("  - No optimizable parameters found in the expression. No changes made.")
            return None

        print(f"  - Found active parameters in expression: {active_param_names}")

        # Create a single, reusable numerical function from the expression
        f_lambdified = sp.lambdify(['X1'] + active_param_symbols, self.sympy_expr, modules='numpy')

        def objective_mse(b_values: np.ndarray) -> float:
            """Generic MSE objective function that takes a NumPy array."""
            return np.mean((y_data - f_lambdified(x_data, *b_values))**2)

        # --- Step 1: Establish the Baseline ---
        # The baseline is the best possible MSE with the full float model.
        params_best_float = np.array([self.modified_parameters[name] for name in active_param_names])
        baseline_mse = objective_mse(params_best_float)
        print(baseline_mse)

        # --- Step 2: Calculate the Cost of Rounding for Each Parameter ---
        substite_param = {} # Params that will be rounded
        b_vals_new = {}     # Params that will remain floats

        for i, name in enumerate(active_param_names):
            # Create a fresh copy of the optimal params for this test
            params_hybrid = np.copy(params_best_float)
            param_int = round(params_hybrid[i])
            params_hybrid[i] = param_int # Modify only the parameter being tested

            mse_hybrid = objective_mse(params_hybrid)
            
            absolute_increase = abs(mse_hybrid - baseline_mse)
            relative_increase = absolute_increase / baseline_mse if baseline_mse > 1e-22 else float('inf')

            print(f"  - Testing '{name}': Rel cost = {relative_increase}, {mse_hybrid}")
        

            if relative_increase < threshold or absolute_increase < abs_mse_tol:
                substite_param[name] = param_int
            else:
                b_vals_new[name] = self.modified_parameters[name]
        
        # --- Step 3: Create and Re-optimize a New Solution ---
        if not substite_param: # Check if the dictionary is empty
            print("  - No parameters met the rounding criteria. Keeping full float model.")
            return None # Return nothing if no changes were made

        print(f"  - Rounding parameters: {list(substite_param.keys())}")
        
        # Create the new expression with the integer values substituted in
        expression_w_int = self.sympy_expr.subs(substite_param)
        
        # Create the new solution object. The initial parameters for its optimization
        # will be only the ones that remained floats.
        Solution_w_integers = Promising_solution(
            expression_w_int,
            (x_data, y_data),
            self,
            initial_parameters=b_vals_new
        )
        
        print(f'Creating a new Promising solution: {Solution_w_integers.sympy_expr}')
        
        if b_vals_new:
            print(f'Re-optimizing the remaining float parameters: {list(b_vals_new.keys())}')

            # This assumes your Promising_solution class has a method to run optimization.
            # The optimization will now only work on the parameters defined in `b_vals_new`.
            Solution_w_integers.run_multiple_optimisations(
                n_runs=100,
                k_confirm=3,
                scatter_fraction=0.1,
                cluster_tolerance=1e-6
            )
        else:
            print("  - All parameters were rounded. No re-optimization needed.")
            
        return Solution_w_integers

    def int_analysis_and_modification_iterative_optimisation(self, threshold: float = 0.01, abs_mse_tol: float = 1e-18):
        """
        Analyzes the solution's parameters to perform 'smart rounding' and
        creates a new, re-optimized PromisingSolution if simplification is possible.

        This version uses a robust "iterative refinement" strategy. At each step,
        it rounds the single cheapest parameter and then re-optimizes the others
        before deciding whether to accept the change.
        """
        # --- Setup: Identify Active Parameters ---
        x_data, y_data = self.original_solution.problem.test_data
        all_symbols_in_expr = self.sympy_expr.free_symbols
        all_param_names = self.modified_parameters.keys()
        active_param_symbols = [s for s in all_symbols_in_expr if s.name in all_param_names]
        active_param_names = sorted([s.name for s in active_param_symbols])

        if not active_param_names:
            print("  - No optimizable parameters found in the expression. No changes made.")
            return None

        print(f"  - Found active parameters in expression: {active_param_names}")
        f_lambdified = sp.lambdify(['X1'] + active_param_symbols, self.sympy_expr, modules='numpy')

        def objective_mse(b_values: np.ndarray) -> float:
            return np.mean((y_data - f_lambdified(x_data, *b_values))**2)

        # --- Step 1: Establish the Baseline ---
        params_best_float = np.array([self.modified_parameters[name] for name in active_param_names])
        baseline_mse = objective_mse(params_best_float)
        print(f"  - Baseline Best MSE: {baseline_mse:.6e}")

        # --- Step 2: Iterative Refinement Loop ---
        print("\n--- Starting Iterative Refinement Process ---")
        
        # Keep track of which parameters are floats and which are fixed as ints
        current_float_params = {name: val for name, val in zip(active_param_names, params_best_float)}
        final_int_params = {}
        
        # This will be updated after each successful rounding step
        current_best_mse = baseline_mse

        while True:
            if not current_float_params:
                print("\n  - All parameters have been rounded.")
                break

            # Find the cost of rounding each of the *remaining* float parameters
            costs = []
            for name_to_test in current_float_params.keys():
                # The other floats that need re-optimizing
                params_to_reopt_names = [name for name in current_float_params.keys() if name != name_to_test]

                # If there are other floats, re-optimize them to see the true cost
                if params_to_reopt_names:
                    def reduced_objective(free_vals):
                        b_values = []
                        free_vals_iter = iter(free_vals)
                        # Build the full parameter array for the objective function
                        for name in active_param_names:
                            if name == name_to_test:
                                b_values.append(round(current_float_params[name]))
                            elif name in final_int_params:
                                b_values.append(final_int_params[name])
                            else: # It's a float to be optimized
                                b_values.append(next(free_vals_iter))
                        return objective_mse(np.array(b_values))

                    initial_guess_reduced = [current_float_params[name] for name in params_to_reopt_names]
                    res = minimize(reduced_objective, x0=initial_guess_reduced, method='L-BFGS-B')
                    cost_mse = res.fun
                else: # This is the last float parameter, no re-optimization needed
                    b_values_list = []
                    for name in active_param_names:
                        if name == name_to_test:
                            b_values_list.append(round(current_float_params[name]))
                        else:
                            b_values_list.append(final_int_params[name])
                    cost_mse = objective_mse(np.array(b_values_list))

                costs.append({'name': name_to_test, 'cost_mse': cost_mse})

            # Find the cheapest parameter to round in this iteration
            if not costs: break # Should not happen, but as a safeguard
            cheapest_param = min(costs, key=lambda x: x['cost_mse'])
            name_to_round = cheapest_param['name']
            new_mse = cheapest_param['cost_mse']
            
            # *** CORRECTED LOGIC V3: Properly define cost and handle improvements ***
            increase = new_mse - current_best_mse
            
            # The cost is the increase in error. If error decreases, cost is 0.
            cost = max(0, increase)
            relative_cost = cost / current_best_mse if current_best_mse > 1e-22 else float('inf')
            
            print(f"\n  - Cheapest candidate to round is '{name_to_round}'.")
            print(f"    - If rounded, new MSE would be {new_mse:.6e} (Abs change: {increase:.6e}, Rel cost: {relative_cost:.2%})")

            # Check if this cheapest move is acceptable. Any improvement (increase <= 0) is acceptable.
            if increase <= 0 or relative_cost < threshold or abs(increase) < abs_mse_tol:
                print(f"    - Cost is acceptable. Locking in '{name_to_round}' as an integer.")
                # Lock it in: move from float dict to int dict
                final_int_params[name_to_round] = round(current_float_params[name_to_round])
                del current_float_params[name_to_round]
                # Update the baseline for the next iteration
                current_best_mse = new_mse
            else:
                print("    - Cost of cheapest move exceeds threshold. Stopping refinement.")
                break # The best move is too expensive, so we stop

        # --- Step 3: Create and Finalize the New Solution ---
        if not final_int_params:
            print("\n  - No parameters met the rounding criteria. Keeping full float model.")
            return None

        print(f"\n  - Final parameters to be rounded: {list(final_int_params.keys())}")
        
        expression_w_int = self.sympy_expr.subs(final_int_params)
        
        # The remaining floats are in current_float_params
        b_vals_new = current_float_params
        
        Solution_w_integers = Promising_solution(
            expression_w_int,
            (x_data, y_data),
            self,
            initial_parameters=b_vals_new
        )
        
        print(f'Creating a new Promising solution: {Solution_w_integers.sympy_expr}')
        
        if b_vals_new:
            print(f'Re-optimizing the final float parameters: {list(b_vals_new.keys())}')
            Solution_w_integers.run_multiple_optimisations(
                n_runs=100,
                k_confirm=2,
                scatter_fraction=0.1,
                cluster_tolerance=1e-6
            )
        else:
            print("  - All parameters were rounded. No final re-optimization needed.")
            
        return Solution_w_integers
