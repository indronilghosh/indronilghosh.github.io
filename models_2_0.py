## Package imports and utility functions written by Indro:
# Installation note - conda install the following packages:
# jupyterlab numpy scipy sympy matplotlib pandas conda-forge::openpyxl conda-forge::lmfit conda-forge::numdifftools

import pandas as pd
import numpy as np
from lmfit import Model, Parameters
from lmfit.models import PolynomialModel
import sympy
from sympy import Symbol, Poly, N, latex
from IPython.display import display, HTML, Markdown, Latex

# Load the data from the Excel file:

# 3rd argument is header_rows_list, containing a minimum of 1 and a maximum of 2 row integers
# Denotes what rows to consider in constructing variable names -
# If two row integers are given, then the first row (e.g. material name) is used as a prefix,
# while the second row (e.g. indepedent or dependent variable) is used as a suffix.
# If two row integers are given, it is assume that the format is:
# |                  Material Name 1              |   ...   |                  Material Name n              |
# | Independent variable 1 | Dependent variable 1 |   ...   | Independent variable n | Dependent variable n |

# If only one row integers are given, it is assume that the format is:
# | Independent variable 1 | Dependent variable 1 |   ...   | Independent variable n | Dependent variable n |

# Row integers start counting from 1 like in an Excel spreadsheet.

translation_table = {ord(' ') : None, ord(',') : None, ord('.') : None, ord('-') : None, ord('°') : None, ord('\'') : None, ord('"') : None,\
                     ord('(') : None, ord(')') : None, ord('[') : None, ord(']') : None, ord('{') : None, ord('}') : None}

def load_data(filename, sheet_name, header_rows_list=[1, 3], num_data_rows=20, room_temp=293, display_data_table_bool=True):

    # Column labels, used to construct variable names
    # Variable names that mirror the column labels, only removing non-alphanumeric characters in the translation_table
    if len(header_rows_list) == 1:
        
        col_names = list(pd.read_excel(filename, sheet_name=sheet_name, header=header_rows_list[0]-1, nrows=1))
        cols_with_data = [i for i in range(0, len(col_names)) if "Unnamed" not in col_names[i]]
        col_names = [col_name for col_name in col_names if "Unnamed" not in col_name]
        var_names = [col.translate(translation_table) for col in col_names]
        
    elif len(header_rows_list) == 2:
        
        col_names_prefix = list(pd.read_excel(filename, sheet_name=sheet_name, header=header_rows_list[0]-1, nrows=1))
        col_names_prefix = [col_name for col_name in col_names_prefix if "Unnamed" not in col_name]
        var_names_prefix = [col.translate(translation_table) for col in col_names_prefix]
        
        col_names_suffix = list(pd.read_excel(filename, sheet_name=sheet_name, header=header_rows_list[1]-1, nrows=1))
        cols_with_data = [i for i in range(0, len(col_names_suffix)) if "Unnamed" not in col_names_suffix[i]]
        col_names_suffix = [col_name for col_name in col_names_suffix if "Unnamed" not in col_name]
        var_names_suffix = [col.translate(translation_table) for col in col_names_suffix]

        var_names = []
        for i in range(0, len(col_names_prefix)):
            for j in range(2*i, 2*i+2):
                var_names.append(var_names_prefix[i] + var_names_suffix[j])
    else:
        raise ValueError("header_rows_list must be a list containing a minimum of 1 and a maximum of 2 row integers!")
        
    # Temporarily store the columnar Excel data
    # with pd.option_context('future.no_silent_downcasting', True):
    temp_data = pd.read_excel(filename, sheet_name=sheet_name, header=header_rows_list[-1]-1, nrows=num_data_rows, usecols=cols_with_data).replace('RT', room_temp)
    
    temp_data.columns = var_names
    if display_data_table_bool:
        display(pd.DataFrame(temp_data).dropna(how='all'))

    variables = [temp_data[col].dropna().to_numpy() for col in var_names]

    return var_names, variables, temp_data


# Concatenate and sort lists of x and y data
def concatenate_and_sort(x_list, y_list):

    x_concat = np.concatenate(x_list)
    y_concat = np.concatenate(y_list)

    x_sorted_indices = x_concat.argsort()
    x_sorted = x_concat[x_sorted_indices]
    y_sorted = y_concat[x_sorted_indices]

    return x_sorted, y_sorted


##################################################################################################
# Weibull distribution model function
# When x < p_1: the function is zero because the distribution starts at p_1
# This is a key characteristic, as the Weibull distribution is defined only for x ≥ p_1.
# For x ≥ p_1: The distribution takes on different shapes depending on the value of :
# p_2 < 1: The hazard function is decreasing, indicating a higher probability of failure (or event) early on, with the rate decreasing over time.
# The PDF has a peak at  x=p_1 and decreases monotonically.
# p_2 = 1: The distribution simplifies to an exponential distribution with a constant hazard function.
# The PDF is a decreasing exponential function.
# p_2 > 1: The hazard function is increasing, indicating a lower probability of failure early on, with the rate increasing over time.
# The PDF initially increases, reaches a peak, and then decreases, showing a typical "bell" shape.
# Impact of p_0:
# Larger values of p_0 stretch the distribution, making it wider and less peaked.
# Smaller values of p_0 compress the distribution, making it narrower and more peaked.
# Impact of p_1:
# The location parameter p_1 shifts the entire distribution along the x-axis.
# Changing p_1 does not affect the shape of the distribution but changes the starting point.

p_0_default_w = 1
p_1_default_w = 0
p_2_default_w = 1
def weibull(x, p_0=p_0_default_w, p_1=p_1_default_w, p_2=p_2_default_w):
    return (p_2 / p_0) * ((x - p_1) / p_0)**(p_2 - 1) * np.exp(-((x - p_1) / p_0)**p_2)

####################################################################################################
# Generalized Exponential model function
# Constant Term (p_0): This term shifts the entire function vertically. 
# It determines the baseline value of the function when all other terms are zero.
# Linear Term (p_1 + p_2 x): introduces a linear component that depends on x
# p_1 is the y-intercept of this linear component.
# p_2 is the slope of the linear component, determining how quickly the value of 
# p_1 + p_2 x changes with x
# Exponential Term (exp(p_3 x)): It controls the rate of exponential growth or decay:
# If p_3 > 0 , the function grows exponentially.
# If p_3 < 0, the function decays exponentially.

p_0_default_e = 0
p_1_default_e = 0
p_2_default_e = 1
p_3_default_e = 1
def exponential(x, p_0=p_0_default_e, p_1=p_1_default_e, p_2=p_2_default_e, p_3=p_3_default_e):
    return p_0 + (p_1 + p_2 * x) * np.exp(p_3 * x)

###################################################################################################
# Transition function
# The function starts near a for large negative x. It transitions smoothly around x = p_3, 
# controlled by the parameter p_4. It ends up behaving like p_1 + p_2 x for large positive x
# p_0: Shifts the entire function vertically.
# p_1: Controls the final value for large positive x
# p_2: Controls the slope of the linear component added to p_1
# p_3: Controls the center of the transition.
# p_4: Controls the steepness of the transition

p_0_default_t = 0
p_1_default_t = 1
p_2_default_t = 0
p_3_default_t = 100
p_4_default_t = 10
def transition(x, p_0=p_0_default_t, p_1=p_1_default_t, p_2=p_2_default_t, p_3=p_3_default_t, p_4=p_4_default_t):
    return p_0 + 0.5 * (p_1 - p_0 + p_2 * x) * (1 + np.tanh((x - p_3) / p_4))

###################################################################################################
# Dip function
# The function has two transitions smoothly around x = p_2, and x = p_5, 
# p_0: Shifts the entire function vertically.
# p_1: scale value
# p_4: second scale value

p_0_default_d = 0
p_1_default_d = 1
p_2_default_d = 1
p_3_default_d = 1
p_4_default_d = 1
p_5_default_d = 1
p_6_default_d = 1
def dip(x, p_0=p_0_default_d, p_1=p_1_default_d, p_2=p_2_default_d, p_3=p_3_default_d, p_4=p_4_default_d, p_5=p_5_default_d, p_6=p_6_default_d):
    return p_0 + p_1 * (1 + np.tanh((x - p_2) / p_3))+ p_4 * (1 + np.tanh((x - p_5) / p_6))

###################################################################################################
# Hardening function
p_0_default_h = 0
p_1_default_h = 1
p_2_default_h = 1
def hardening(x, p_0=p_0_default_h, p_1=p_1_default_h, p_2=p_2_default_h):
    return (p_0 + p_1 * x**(1/2)) * (1 - np.exp(-x/p_2))**(1/2)

###################################################################################################

# Swelling function
p_0_default_s = 1
p_1_default_s = 1
def swelling(x, p_0=p_0_default_s, p_1=p_1_default_s):
    return p_0 * x * (1 - np.exp(-x/p_1))

###################################################################################################

# Calculate confidence intervals
def get_model_fit_and_print_it(x, y, sigma=3, fit_func='poly', method='leastsq', param_initials=None, param_defaults=None,
                               material_name=None, property_name=None, eq_digits=4, print_bool=False, print_params_bool=True, fit_symbol='T'):

    # Utility function to assemble parameters for LMFIT
    def assemble_params(num_params, func_suffix):
        params = Parameters()
        
        for i in range(0, num_params):
            if not np.isnan(param_initials[i]):
                params.add('p_' + str(i), value=param_initials[i])
            else:
                if param_defaults is None:
                    params.add('p_' + str(i), value=eval('p_' + str(i) + '_default_' + func_suffix), vary=False)
                else:
                    params.add('p_' + str(i), value=param_defaults[i], vary=False)
                        
        return params

    # Model fitting based on specified fitting function ('fit_func') and fitting method ('method')
    if fit_func == 'poly':
        
        poly_deg = len(param_initials)-1
        model = PolynomialModel(degree=poly_deg, nan_policy='propagate')

        args = ''
        for i in range(0, poly_deg):
            args += 'c' + str(i) + '=' + str(param_initials[i]) + ", "
        args += 'c' + str(poly_deg) + '=' + str(poly_deg)

        params = eval("model.make_params("  + args + ")") 
        result = model.fit(y, x=x, method=method, params=params)
        
        # Regression curve
        fit_for_x = result.eval(result.params, x=x)
        # Confidence interval
        dely = result.eval_uncertainty(sigma=sigma, x=x)
        # Fit results for extremes of confidence interval
        result_min = model.fit(fit_for_x-dely, x=x, method=method, params=params)
        result_max = model.fit(fit_for_x+dely, x=x, method=method, params=params)
        
    elif fit_func == 'weibull':
        
        model = Model(weibull)
        params = Parameters()

        if not len(param_initials) == 3:
            raise ValueError("Must give 3 initial parameters to use the weibull fitting function. Initial params may include NaN to fix a variable to its constant default.")

        if not param_defaults is None:
            if not len(param_defaults) == 3:
                raise ValueError("If param_defaults is not None, must give 3 default parameters to prescribe defaults to the weibull fitting function.")

        params = assemble_params(3, 'w')
        result = model.fit(y, x=x, method=method, params=params, nan_policy='propagate')

        # Regression curve
        fit_for_x = result.eval(result.params, x=x)
        # Confidence interval
        dely = result.eval_uncertainty(sigma=sigma, x=x)
        # Fit results for extremes of confidence interval
        result_min = model.fit(fit_for_x-dely, x=x, method=method, params=params, nan_policy='propagate')
        result_max = model.fit(fit_for_x+dely, x=x, method=method, params=params, nan_policy='propagate')

    elif fit_func == 'exponential':
        
        model = Model(exponential)
        params = Parameters()

        if not len(param_initials) == 4:
            raise ValueError("Must give 4 initial parameters to use the exponential fitting function. Initial params may include NaN to fix a variable to its constant default.")

        if not param_defaults is None:
            if not len(param_defaults) == 4:
                raise ValueError("If param_defaults is not None, must give 4 default parameters to prescribe defaults to the exponential fitting function.")

        params = assemble_params(4, 'e')
        result = model.fit(y, x=x, method=method, params=params, nan_policy='propagate')

        # Regression curve
        fit_for_x = result.eval(result.params, x=x)
        # Confidence interval
        dely = result.eval_uncertainty(sigma=sigma, x=x)
        # Fit results for extremes of confidence interval
        result_min = model.fit(fit_for_x-dely, x=x, method=method, params=params, nan_policy='propagate')
        result_max = model.fit(fit_for_x+dely, x=x, method=method, params=params, nan_policy='propagate')
        
    elif fit_func == 'transition':
        
        model = Model(transition)
        params = Parameters()

        if not len(param_initials) == 5:
            raise ValueError("Must give 5 initial parameters to use the transition fitting function. Initial params may include NaN to fix a variable to its constant default.")

        if not param_defaults is None:
            if not len(param_defaults) == 5:
                raise ValueError("If param_defaults is not None, must give 5 default parameters to prescribe defaults to the transition fitting function.")

        params = assemble_params(5, 't')
        result = model.fit(y, x=x, method=method, params=params, nan_policy='propagate')

        # Regression curve
        fit_for_x = result.eval(result.params, x=x)
        # Confidence interval
        dely = result.eval_uncertainty(sigma=sigma, x=x)
        # Fit results for extremes of confidence interval
        result_min = model.fit(fit_for_x-dely, x=x, method=method, params=params, nan_policy='propagate')
        result_max = model.fit(fit_for_x+dely, x=x, method=method, params=params, nan_policy='propagate')
        
    elif fit_func == 'dip':
    
        model = Model(dip)
        params = Parameters()

        if not len(param_initials) == 7:
            raise ValueError("Must give 7 parameters to use the dip fitting function. Parameters may include NaN to fix a variable to its constant default.")

        if not param_defaults is None:
            if not len(param_defaults) == 7:
                raise ValueError("If param_defaults is not None, must give 7 default parameters to prescribe defaults to the dip fitting function.")

        params = assemble_params(7, 'd')
        result = model.fit(y, x=x, method=method, params=params, nan_policy='propagate')

        # Regression curve
        fit_for_x = result.eval(result.params, x=x)
        # Confidence interval
        dely = result.eval_uncertainty(sigma=sigma, x=x)
        # Fit results for extremes of confidence interval
        result_min = model.fit(fit_for_x-dely, x=x, method=method, params=params, nan_policy='propagate')
        result_max = model.fit(fit_for_x+dely, x=x, method=method, params=params, nan_policy='propagate')

    elif fit_func == 'hardening':
    
        model = Model(hardening)
        params = Parameters()

        if not len(param_initials) == 3:
            raise ValueError("Must give 3 parameters to use the hardening fitting function. Parameters may include NaN to fix a variable to its constant default.")

        if not param_defaults is None:
            if not len(param_defaults) == 3:
                raise ValueError("If param_defaults is not None, must give 3 default parameters to prescribe defaults to the hardening fitting function.")

        params = assemble_params(3, 'h')
        result = model.fit(y, x=x, method=method, params=params, nan_policy='propagate')

        # Regression curve
        fit_for_x = result.eval(result.params, x=x)
        # Confidence interval
        dely = result.eval_uncertainty(sigma=sigma, x=x)
        # Fit results for extremes of confidence interval
        result_min = model.fit(fit_for_x-dely, x=x, method=method, params=params, nan_policy='propagate')
        result_max = model.fit(fit_for_x+dely, x=x, method=method, params=params, nan_policy='propagate')

    elif fit_func == 'swelling':
    
        model = Model(swelling)
        params = Parameters()

        if not len(param_initials) == 2:
            raise ValueError("Must give 2 parameters to use the hardening fitting function. Parameters may include NaN to fix a variable to its constant default.")

        if not param_defaults is None:
            if not len(param_defaults) == 2:
                raise ValueError("If param_defaults is not None, must give 2 default parameters to prescribe defaults to the hardening fitting function.")

        params = assemble_params(2, 's')
        result = model.fit(y, x=x, method=method, params=params, nan_policy='propagate')

        # Regression curve
        fit_for_x = result.eval(result.params, x=x)
        # Confidence interval
        dely = result.eval_uncertainty(sigma=sigma, x=x)
        # Fit results for extremes of confidence interval
        result_min = model.fit(fit_for_x-dely, x=x, method=method, params=params, nan_policy='propagate')
        result_max = model.fit(fit_for_x+dely, x=x, method=method, params=params, nan_policy='propagate')

    else:
        raise ValueError("Please give a valid fit_func string among: 'poly', 'weibull', 'exponential', 'transition', 'dip', 'hardening', 'swelling'!")

    def make_latex_poly(params, c_list, poly_deg):
        sym = Symbol(fit_symbol)
                
        latex_list = []
        for i in range(0, poly_deg+1):
            latex_list.insert(0, latex(N(params[c_list[poly_deg-i]].value, eq_digits) * sym**(poly_deg-i), min=0, max=0))
            
        latex_to_print = ''
        for i in range(0, poly_deg+1):
            if "+" == latex_list[i].lstrip()[0] or "-" == latex_list[i].lstrip()[0]:
                latex_to_print += latex_list[i] + " "
            else:
                latex_to_print += "+ " + latex_list[i] + " "
                
        latex_to_print = '\\boxed{ ' + latex_to_print + ' }'
        
        if '\cdot' in latex_to_print:
            latex_to_print = latex_to_print.replace('\cdot', '\\times')

        return latex_to_print

    def make_latex_weibull(params):
        sym = Symbol(fit_symbol)
        
        p_0_fit = N(result.params['p_0'].value, eq_digits)
        p_1_fit = N(result.params['p_1'].value, eq_digits)
        p_2_fit = N(result.params['p_2'].value, eq_digits)
        
        latex_to_print = '\\boxed{ '\
        + latex((p_2_fit / p_0_fit) * ((sym - p_1_fit) / p_0_fit)**(p_2_fit - 1) * sympy.exp(-((sym - p_1_fit) / p_0_fit)**p_2_fit), min=0, max=0)\
        + ' }'
        if '\cdot' in latex_to_print:
            latex_to_print = latex_to_print.replace('\cdot', '\\times')

        return latex_to_print

    def make_latex_exponential(params):
        sym = Symbol(fit_symbol)
        
        p_0_fit = N(result.params['p_0'].value, eq_digits)
        p_1_fit = N(result.params['p_1'].value, eq_digits)
        p_2_fit = N(result.params['p_2'].value, eq_digits)
        p_3_fit = N(result.params['p_3'].value, eq_digits)

        a_term_latex = latex(p_0_fit)
        exp_term_latex = latex((p_1_fit + p_2_fit * sym) * sympy.exp(p_3_fit * sym), min=0, max=0)

        if "+" != a_term_latex.lstrip()[0] and "-" != a_term_latex.lstrip()[0]:
            a_term_latex = "+ " + a_term_latex + " "

        if "+" != exp_term_latex.lstrip()[0] and "-" != exp_term_latex.lstrip()[0]:
            exp_term_latex = "+ " + exp_term_latex
        
        latex_to_print = '\\boxed{ ' + a_term_latex + exp_term_latex + ' }'
        if '\cdot' in latex_to_print:
            latex_to_print = latex_to_print.replace('\cdot', '\\times')

        return latex_to_print

    def make_latex_transition(params):
        sym = Symbol(fit_symbol)
        
        p_0_fit = N(result.params['p_0'].value, eq_digits)
        p_1_fit = N(result.params['p_1'].value, eq_digits)
        one_half_times_p_1_minus_p_0_fit = N(0.5*(result.params['p_1'].value - result.params['p_0'].value), eq_digits)
        one_half_times_p_2_fit = N(0.5*result.params['p_2'].value, eq_digits)
        p_3_fit = N(result.params['p_3'].value, eq_digits)
        p_4_fit = N(result.params['p_4'].value, eq_digits)
        
        a_term_latex = latex(p_0_fit)
        tanh_term_latex = latex((one_half_times_p_1_minus_p_0_fit + one_half_times_p_2_fit * sym) * (1 + sympy.tanh((sym - p_3_fit) / p_4_fit)), min=0, max=0)

        if "+" != a_term_latex.lstrip()[0] and "-" != a_term_latex.lstrip()[0]:
            a_term_latex = "+ " + a_term_latex + " "

        if "+" != tanh_term_latex.lstrip()[0] and "-" != tanh_term_latex.lstrip()[0]:
            tanh_term_latex = "+ " + tanh_term_latex
        
        latex_to_print = '\\boxed{ ' + a_term_latex + tanh_term_latex + ' }'
        if '\cdot' in latex_to_print:
            latex_to_print = latex_to_print.replace('\cdot', '\\times')

        return latex_to_print

    def make_latex_dip(params):

        # Define the symbol for the variable in the equation
        sym = Symbol(fit_symbol)

        p_0_fit = N(result.params['p_0'].value, eq_digits)
        p_1_fit = N(result.params['p_1'].value, eq_digits)
        p_2_fit = N(result.params['p_2'].value, eq_digits)
        p_3_fit = N(result.params['p_3'].value, eq_digits)
        p_4_fit = N(result.params['p_4'].value, eq_digits)
        p_5_fit = N(result.params['p_5'].value, eq_digits)
        p_6_fit = N(result.params['p_6'].value, eq_digits)
        
        # Create the LaTeX representations for each term
        a_term_latex = latex(p_0_fit) # Constant term
        
        # Tanh terms with proper handling of multiplication and parentheses
        tanh1_term = p_1_fit * (1 + sympy.tanh((sym - p_2_fit) / p_3_fit))
        tanh2_term = p_4_fit * (1 + sympy.tanh((sym - p_5_fit) / p_6_fit))
        
        # Convert the tanh terms to LaTeX
        tanh1_term_latex = latex(tanh1_term, min=0, max=0)
        tanh2_term_latex = latex(tanh2_term, min=0, max=0)
        
        # Ensure the terms have a "+" or "-" sign at the beginning if necessary
        if a_term_latex.lstrip()[0] not in ["+", "-"]:
            a_term_latex = "+ " + a_term_latex + " "
        
        if tanh1_term_latex.lstrip()[0] not in ["+", "-"]:
            tanh1_term_latex = "+ " + tanh1_term_latex
            
        if tanh2_term_latex.lstrip()[0] not in ["+", "-"]:
            tanh2_term_latex = "+ " + tanh2_term_latex
        
        # Combine the LaTeX expressions inside a \boxed environment
        latex_to_print = '\\boxed{ ' + a_term_latex + tanh1_term_latex + tanh2_term_latex + ' }'

        return latex_to_print

    def make_latex_hardening(params):
        sym = Symbol(fit_symbol)
        
        p_0_fit = N(result.params['p_0'].value, eq_digits)
        p_1_fit = N(result.params['p_1'].value, eq_digits)
        p_2_fit = N(result.params['p_2'].value, eq_digits)
        
        latex_to_print = '\\boxed{ '\
        + latex((p_0_fit + p_1_fit * sym**(1/2)) * (1 - sympy.exp(-sym/p_2_fit))**(1/2), min=0, max=0)\
        + ' }'
        if '\cdot' in latex_to_print:
            latex_to_print = latex_to_print.replace('\cdot', '\\times')

        return latex_to_print

    def make_latex_swelling(params):
        sym = Symbol(fit_symbol)
        
        p_0_fit = N(result.params['p_0'].value, eq_digits)
        p_1_fit = N(result.params['p_1'].value, eq_digits)

        latex_to_print = '\\boxed{ '\
        + latex(p_0_fit * sym * (1 - sympy.exp(-sym/p_1_fit)), min=0, max=0)\
        + ' }'
        if '\cdot' in latex_to_print:
            latex_to_print = latex_to_print.replace('\cdot', '\\times')

        return latex_to_print
        
    # Printing the fitting parameters and equations
    if print_bool:
        if print_params_bool:
            display(HTML("<hr>"))
            display(Markdown(f'**Fitting parameters for {material_name} {property_name}** \n'))
            print(result.fit_report())
            display(HTML("<hr>"))
            display(Markdown(f'**The equations for {material_name} {property_name} are:**\n'))
        
        if fit_func == 'poly':
            display(Latex(f'Fit: ${make_latex_poly(result.params, ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"], poly_deg)}$'))
            if print_params_bool:
                display(Latex(f'Minimum of confidence interval: ${make_latex_poly(result_min.params, ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"], poly_deg)}$'))
                display(Latex(f'Maximum of confidence interval: ${make_latex_poly(result_max.params, ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"], poly_deg)}$'))

        elif fit_func == 'weibull':
            display(Latex(f'Fit: ${make_latex_weibull(result.params)}$'))
            if print_params_bool:
                display(Latex(f'Minimum of confidence interval: ${make_latex_weibull(result_min.params)}$'))
                display(Latex(f'Maximum of confidence interval: ${make_latex_weibull(result_max.params)}$'))
            
        elif fit_func == 'exponential':
            display(Latex(f'Fit: ${make_latex_exponential(result.params)}$'))
            if print_params_bool:
                display(Latex(f'Minimum of confidence interval: ${make_latex_exponential(result_min.params)}$'))
                display(Latex(f'Maximum of confidence interval: ${make_latex_exponential(result_max.params)}$'))

        elif fit_func == 'transition':
            display(Latex(f'Fit: ${make_latex_transition(result.params)}$'))
            if print_params_bool:
                display(Latex(f'Minimum of confidence interval: ${make_latex_transition(result_min.params)}$'))
                display(Latex(f'Maximum of confidence interval: ${make_latex_transition(result_max.params)}$'))
            
        elif fit_func == 'dip':
            display(Latex(f'Fit: ${make_latex_dip(result.params)}$'))
            if print_params_bool:
                display(Latex(f'Minimum of confidence interval: ${make_latex_dip(result_min.params)}$'))
                display(Latex(f'Maximum of confidence interval: ${make_latex_dip(result_max.params)}$'))
            
        elif fit_func == 'hardening':
            display(Latex(f'Fit: ${make_latex_hardening(result.params)}$'))
            if print_params_bool:
                display(Latex(f'Minimum of confidence interval: ${make_latex_hardening(result_min.params)}$'))
                display(Latex(f'Maximum of confidence interval: ${make_latex_hardening(result_max.params)}$'))

        elif fit_func == 'swelling':
            display(Latex(f'Fit: ${make_latex_swelling(result.params)}$'))
            if print_params_bool:
                display(Latex(f'Minimum of confidence interval: ${make_latex_swelling(result_min.params)}$'))
                display(Latex(f'Maximum of confidence interval: ${make_latex_swelling(result_max.params)}$'))
            
        else:
            pass # valid fit_func string checked in previous if block
            
    return result