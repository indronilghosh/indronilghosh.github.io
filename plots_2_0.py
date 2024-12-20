## Package imports and utility functions written by Indro:
# Installation note - conda install the following packages:
# jupyterlab numpy scipy sympy matplotlib pandas conda-forge::openpyxl conda-forge::lmfit conda-forge::numdifftools

import numpy as np
from itertools import product, cycle
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

marker_cycle = cycle(('o', '^', 'v', '<', '>', 'd', 's', '*')) 
fsize1 = 24
fsize2 = 20
fsize3 = 14
def data_plot(x_data_list, y_data_list, x_fit_list=None, y_fit_list=None,
              x_label='X-axis', y_label='Y-axis', title='Plot',
              scale='linear', x_lim=None, y_lim=None, 
              grid=True, legend=True, data_labels=None, fit_labels=None,
              data_colors=None, fit_colors=None, marker_size=25, 
              fit_line_width=2, x_label_font_size=12, y_label_font_size=12, 
              title_font_size=16, legend_font_size=8, legend_loc='best', legend_num_cols=2):

    plt.figure(figsize=(6, 4), dpi=300)

    # Ensure the data lists have the same length, and scatter plot data if provided
    if x_data_list is not None and y_data_list is not None:

        # Set default values if not provided
        if data_labels is None:
            data_labels = [f'Data {i+1}' for i in range(len(x_data_list))]

        if data_colors is None:
            data_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        if len(x_data_list) != len(y_data_list):
            raise ValueError("x_data_list and y_data_list must have the same length.")
        
        for i, (x_data, y_data) in enumerate(zip(x_data_list, y_data_list)):
            plt.scatter(x_data, y_data, label=data_labels[i], color=data_colors[i], s=marker_size, marker = next(marker_cycle))

    # Ensure the fit lists have the same length, and scatter plot fit lines if provided
    if x_fit_list is not None and y_fit_list is not None:

        # Set default values if not provided
        if fit_labels is None and x_fit_list is not None and y_fit_list is not None:
            fit_labels = [f'Fit {i+1}' for i in range(len(x_fit_list))]
        
        if fit_colors is None and x_fit_list is not None and y_fit_list is not None:
            fit_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        if x_fit_list is not None and y_fit_list is not None and len(x_fit_list) != len(y_fit_list):
            raise ValueError("x_fit_list and y_fit_list must have the same length.")
        
        for i, (x_fit, y_fit) in enumerate(zip(x_fit_list, y_fit_list)):
            plt.plot(x_fit, y_fit, label=fit_labels[i], color=fit_colors[i], linewidth=fit_line_width)

    plt.xlabel(x_label, fontsize=x_label_font_size)
    plt.ylabel(y_label, fontsize=y_label_font_size)
    plt.title(title, fontsize=title_font_size)
    
    if scale == 'linear':
        plt.xscale('linear')
        plt.yscale('linear')
    elif scale == 'log-log':
        plt.xscale('log')
        plt.yscale('log')
    elif scale == 'log-x':
        plt.xscale('log')
        plt.yscale('linear')
    elif scale == 'log-y':
        plt.xscale('linear')
        plt.yscale('log')
    else:
        raise ValueError("Scale must be 'linear', 'log-log', 'log-x', or 'log-y'")
        
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    if grid:
        plt.grid(True)
    if legend:
        plt.legend(loc=legend_loc, fontsize=legend_font_size, ncol=legend_num_cols)

# Plot fit and confidence intervals from fitting result
def fit_plot(x, fit_result, sigma=3, legend=True, fit_label='Data Fit', legend_font_size=8, legend_loc='best', legend_num_cols=2, fit_line_width=2, fit_line_color='black', conf_int=True, conf_int_fill_color='blue'):
    # Regression curve
    fit_for_x = fit_result.eval(fit_result.params, x=x)
    plt.plot(x, fit_for_x, linewidth=fit_line_width, color=fit_line_color, label=fit_label)

    # Confidence interval
    if conf_int:
        dely = fit_result.eval_uncertainty(sigma=sigma, x=x)
        plt.fill_between(x, fit_for_x - dely, fit_for_x + dely, color=conf_int_fill_color, alpha=0.2,
                         label=str(sigma) + "σ Conf. Int.")

    if legend:
        plt.legend(loc=legend_loc, fontsize=legend_font_size, ncol=legend_num_cols)

# Wrapper function for convenient plotting
def plot_data(x_data_list, y_data_list, x_for_fit_plot, fit_result, font_size, marker_size, x_label, y_label, x_lim, y_lim, data_labels, title, sigma=3, legend_loc='best', legend_font_size=8, scale=False):
    # Call the plotting functions
    if not scale:
        scale = 'linear'
    data_plot(x_data_list, y_data_list, marker_size=marker_size, x_label_font_size=font_size, y_label_font_size=font_size, x_label=x_label, y_label=y_label, x_lim=x_lim, y_lim=y_lim, data_labels=data_labels, title_font_size=font_size, legend_font_size=legend_font_size, title=title, legend_loc=legend_loc, scale=scale)
    if not fit_result is None:
        fit_plot(x_for_fit_plot, fit_result, sigma=sigma, legend_font_size=legend_font_size, legend_loc=legend_loc)

# Define a function to plot multiple curves
def plot_multiple_curves(temperatures, shift_percent=0.2):
    plt.figure()
    
    # Plot each curve with a shift in strain values
    for i, T in enumerate(temperatures):
        shift = i * shift_percent * TotalElongation(T)  # Shift by 20% of Total Elongation for clarity
        x_values, y_values = stress_strain_curve(T, shift=shift)
        plt.plot(x_values, y_values, marker='', linestyle='-', label=f'{T-273} °C', linewidth=1)
    
    plt.xlabel('Strain [%]', fontsize=12)  # Set font size for x-axis label
    plt.ylabel('Stress [MPa]', fontsize=12)  # Set font size for y-axis label
    plt.title('Stress-Strain Curves at Different Temperatures', fontsize=16)  # Set font size for title
    
    # Modify legend position and line thickness, and remove data points
    plt.legend(loc='best', fontsize=8)  # Control legend position (e.g., 'upper right', 'lower left')
    
    # Control line thickness
    for line in plt.gca().get_lines():
        line.set_linewidth(2)  # Set the line thickness to 2
    
    # Control graph size and center it in the page
    plt.gcf().set_size_inches(10, 6)  # Set figure size (width, height in inches)
    plt.gcf().subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Centering the graph
    
    plt.grid(True)
    plt.xlim(0, max(x_values) + 5)  # Adjust x-axis limit for better visualization
    plt.ylim(0, 700)  # Adjust y-axis limit as needed
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set font size for tick labels

    plt.show()

# Generic function for plotting
def generic_plot(x_data, y_data, shift, title, xlabel, ylabel, xlim_range, ylim_range, label=None, fsize1=fsize1, fsize2=fsize2, fsize3 = fsize3):
    plt.plot(x_data + shift, y_data, label=label, linewidth=2)
    
    # Set x- and y-axis labels font size
    plt.xlabel(xlabel, fontsize=fsize1)
    plt.ylabel(ylabel, fontsize=fsize1)
    
    # Set x- and y-limits
    plt.xlim(xlim_range)
    plt.ylim(ylim_range)

    # Set tick marks font size
    # plt.xticks(fontsize=fsize2)
    # plt.yticks(fontsize=fsize2)
    
    # Set plot title
    plt.title(title, fontsize=fsize1)
    
    # Enable grid
    plt.grid(True)
    
    # Add legend if a label is provided, with specific font size for the legend
    if label:
        plt.legend(fontsize=fsize3)