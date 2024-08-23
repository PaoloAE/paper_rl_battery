from __future__ import print_function
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from pathlib import Path
from IPython import display
import sac_epi
import scipy

"""
This module contains functions to visualize data that was logged with sac_epi.SacTrain. In includes
visualizing plots in Jupyter notebook, exporting them as pdfs.
"""

#most useful functions

def plot_sac_logs(log_dir, actions_per_log=6000, running_reward_file=None, running_loss_file=None,
                running_multi_obj_file=None, actions_file=None, actions_to_plot=150,plot_to_file_line = None,
                suppress_show=False, actions_ylim=None, running_reward_ylim=None,
                dont_clear_output=False,rescale_action=False,k_notation=True,constant_actions_steps=False,
                plot_actions_separately=False,export_performance_pdf_location=None):
    """
    Produces and displays in a Jupyter notebook a single plot with the running reward G, running multi objectives,
    (in this case energy and ergotropy), the Q loss function, the Pi loss function, the alpha temperature parameter,
    the entropy of the policy, and the last chosen actions. It can also save the plot as a pdf if the argument
    export_performance_pdf_location is specified.
    
    Args:
        log_dir (str): location of the folder with all the logging
        actions_per_log (int): number of actions taken between logs. Corresponds to LOG_STEPS hyperparam while training
        running_reward_file (str): location of the txt file with the running reward. If None, default location is used
        running_loss_file (str): location of the txt file with the loss function. If None, default location is used
        running_multi_obj_file (str): location of the txt file with the running multi objectives. If Nonde, default
            location is used
        actions_file (str): location of the txt file with the actions. If None, default location is used
        actions_to_plot (int): how many of the last actions to show
        plot_to_file_line (int): plots logs only up to a given file line. In None, all data is used
        suppress_show (bool): if True, it won't display the plot
        actions_ylim (tuple): a 2 element tuple specifying the y_lim of the actions plot
        running_reward_ylim (tuple): a 2 element tuple specifying the y_lim of the reward plot
        dont_clear_output (bool): if False, it will clear the previous plots shown with this function
        rescale_action (bool): if true, the actions are rescaled all between 0 and 1
        k_notation (bool): if true, uses "k" for 1000 when counting training steps
        constant_actions_steps (bool): if true, the actions are displayed as piece-wise constant. Otherwise dots
        plot_actions_separately (book): if true, a separate panel for each action is used. Otherwise all together
        export_performance_pdf_location(str): if specified, saves the plot in this location        
    """
    #i create the file location strings if they are not passed it
    running_reward_file, running_loss_file, running_multi_obj_file, actions_file = \
        sac_logs_file_location(log_dir, running_reward_file, running_loss_file, running_multi_obj_file, actions_file)

    #check if the logging files exist
    running_reward_exists = Path(running_reward_file).exists()
    running_loss_exists = Path(running_loss_file).exists()
    actions_exists = Path(actions_file).exists()
    
    #count quantities to plot to make the right size plot
    running_multi_obj_quantities = count_quantities(running_multi_obj_file)
    loss_elements = count_quantities(running_loss_file)
    if plot_actions_separately:
        action_panels = count_quantities(actions_file)
    else:
        action_panels = 1
    
    #sum the quantities to plot
    quantities_to_log = int(running_reward_exists) + loss_elements*int(running_loss_exists) + action_panels + \
        running_multi_obj_quantities

    #create the matplotlib subplots
    fig, axes = plt.subplots(quantities_to_log, figsize=(7,quantities_to_log*2.2))
    if quantities_to_log == 1:
        axes = [axes]
    axis_ind = 0
    
    #plot the running reward
    if running_reward_exists:
        plot_running_reward_on_axis(running_reward_file, axes[axis_ind], plot_to_file_line,ylim=running_reward_ylim,
                        k_notation=k_notation)
        axis_ind += 1
    
    #plot the running multi objectives
    if running_multi_obj_quantities > 0:
        plot_running_multi_obj_on_axes(running_multi_obj_file, axes[axis_ind : axis_ind+running_multi_obj_quantities],
                                            plot_to_file_line)
        axis_ind += running_multi_obj_quantities
    
    #plot the running loss
    if running_loss_exists:
        plot_running_loss_on_axis(running_loss_file, axes[axis_ind:axis_ind+loss_elements], plot_to_file_line)
        axis_ind += loss_elements
    
    #plot the last actions
    if actions_exists:
        if plot_actions_separately:
            temp_ax = axes[axis_ind:axis_ind+action_panels]
        else:
            temp_ax = axes[axis_ind]
        plot_actions_on_axis(actions_file, temp_ax, actions_to_plot=actions_to_plot,
                plot_to_file_line= None if plot_to_file_line is None else int(plot_to_file_line*actions_per_log),
                actions_ylim=actions_ylim, rescale_action=rescale_action, k_notation=k_notation,
                constant_steps=constant_actions_steps,plot_actions_separately=plot_actions_separately )
        axis_ind += action_panels
    
    #compact view
    fig.tight_layout()  
    
    #save the plot if requested
    if export_performance_pdf_location is not None:
        fig.savefig(export_performance_pdf_location, bbox_inches='tight')
    
    #display the plot if requested
    if not suppress_show:
        if not dont_clear_output:
            display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.close() 

def save_spectrum_weights_plot_from_file(data_location, plot_location):
    """
    Given a .npz file in data_location that was produced using extra.export_performance_and_protocol_files,
    saves a plot with 7 panels: first 3 are colorplots of the square projection psi(t) onto the spectrum
    respectively of the sum of the cavity and qubits Hamiltonian (without interaction), of the cavity Hamiltonian,
    and the qubits Hamiltonian. Time on x, and energy on y. Then, the entropy of all qubits, the single qubit entropy,
    the single qubit ergotropy and the control are all plotted as a function of time. 

    Args:
        data_location (str): where the .npz file is located
        plot_location (str): location and file name for the plot that will be created
    """
    #load the data from the .npz file
    spectrum_weights_list = np.load(data_location)["data"]
    objs_location = data_location.replace("spectrum_weights", "objs_extra_data").replace(".npz", ".txt")
    objs_data = np.loadtxt(objs_location)

    #create quantities to help looping over the 3 Hamiltonians
    hamil_labels = [r"$H_{tot}$", r"$H_{c}$", r"$H_{q}$"]
    hamil_type = ["tot", "cavity", "qubits"]
    quantity_idx = {"time": 0, "actions": 1, "energy": 2, "ergotropy": 3, "qubit_energy": 4,
                    "qubit_entropy": 5, "qubit_variance": 6, "qubit_purity": 7, "qubits_energy": 8,
                    "qubits_entropy": 9, "qubits_variance": 10, "cavity_energy": 11, "cavity_entropy": 12,
                    "cavity_variance": 13, "tot_energy": 14, "tot_variance": 15}

    #create the figure
    fig, axes = plt.subplots(7,1, sharex=True, figsize=(6,17))

    #prepare arrays for time plotting
    min_t, max_t = np.min(objs_data[:, quantity_idx["time"]]), np.max(objs_data[:, quantity_idx["time"]])
    t_smooth = np.linspace(min_t, max_t, 100)

    #loop over the 3 hamiltonians
    for hamil_index in range(3): #0 htot, 1 hc, 2 hq
        t_vals = []
        e_vals = []
        weight_vals = []
        #loop over timesteps
        for i, spectrum_weight in enumerate(spectrum_weights_list):
            weights = spectrum_weight[hamil_index,:,1]
            energies = spectrum_weight[hamil_index,:,0]
            
            e_vals += list(energies)
            t_vals += [ objs_data[i,quantity_idx["time"]] ] * len(energies)
            weight_vals += list(weights)
            
        #now I make the density
        data = np.vstack([t_vals, e_vals])
        weights = np.array(weight_vals)
        
        # Create Gaussian KDE with weights
        kde = scipy.stats.gaussian_kde(data, weights=weights, bw_method=0.15)
            
        # Define the grid over which to evaluate the KDE
        xmin, xmax = data[0,:].min(), data[0,:].max()
        ymin, ymax = data[1,:].min(),  data[1,:].max()
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        # Evaluate the KDE on the grid
        Z = np.reshape(kde(positions).T, X.shape)

        # Plot the result
        axes[hamil_index].imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax], cmap=plt.cm.jet, aspect='auto')
        axes[hamil_index].set_title(f"Proj. on {hamil_labels[hamil_index]}")
        axes[hamil_index].set_ylabel("Energy")
        
        #add the mean and std
        h_str = hamil_type[hamil_index]
        energy_interp = scipy.interpolate.interp1d(objs_data[:, quantity_idx["time"]],
                                                objs_data[:, quantity_idx[f"{h_str}_energy"]], kind="cubic")
        std_interp = scipy.interpolate.interp1d(objs_data[:, quantity_idx["time"]],
                                np.sqrt(objs_data[:, quantity_idx[f"{h_str}_variance"]]), kind="cubic")
        axes[hamil_index].plot(t_smooth,energy_interp(t_smooth), c="black", linewidth=1.3)
        axes[hamil_index].fill_between(t_smooth, energy_interp(t_smooth) -0.5*std_interp(t_smooth),
                            energy_interp(t_smooth) +0.5*std_interp(t_smooth), color="black", alpha=0.13)

    #add all qubits entropy
    axes[3].plot(objs_data[:, quantity_idx["time"]],objs_data[:, quantity_idx["qubits_entropy"]])
    axes[3].set_ylabel("all qubits entropy")

    #add single qubit entropy
    axes[4].plot(objs_data[:, quantity_idx["time"]],objs_data[:, quantity_idx["qubit_entropy"]])
    axes[4].set_ylabel("single qubit entropy")

    #add ergotropy
    axes[5].plot(objs_data[:, quantity_idx["time"]],objs_data[:, quantity_idx["ergotropy"]])
    axes[5].set_ylabel("single qubit ergotropy")

    #plot the control in last axis
    axes[6].plot(objs_data[:, quantity_idx["time"]],objs_data[:, quantity_idx["actions"]])
    axes[6].set_ylabel("control")

    #time label
    axes[-1].set_xlabel("time")

    #save the figure
    fig.tight_layout()
    fig.savefig(plot_location)

def plot_spectrum_density_on_axis(spectrum_weights_file, axis, energy_eigspaces, hamil_index = 2, g=1.,
                                add_initial_null_value = False):
    """
    Given a .npz file in data_location that was produced using extra.export_performance_and_protocol_files, and given an axis,
    it plots the square of the projections of psi(t) onto the spectrum either of the cavity plus qubit Hamiltonian (no interaction),
    of the cavity Hamiltonian, or qubit Hamiltonian depending on hamil_index. The x is time, the y is the energy, and 
    the color is the square of the projection. As opposed to save_spectrum_weights_plot_from_file, this produces a properly 
    normalized density.

    Args:
        spectrum_weights_file(str): the location of the .npz file
        axis (matplotlib axis): the axis where to do this plot
        energy_eigspaces (np.Array): 1D array with the complete energy spectrum of the Hamiltonian
        hamil_index (int): 0=H_tot, 1=H_cavity, 2=H_qubits
        g(float): constant by which time will be multiplied
        add_initial_null_value (bool): if true, it adds an initial time t=0 with all energy in lowest eigenstate 

    Returns:
        im: the matplotlib object returned by imshow that allows to make a colorbar
    """

    #load the data 
    spectrum_weights_list = np.load(spectrum_weights_file)["data"]
    objs_location = spectrum_weights_file.replace("spectrum_weights", "objs_extra_data").replace(".npz", ".txt")
    objs_data = np.loadtxt(objs_location)

    #useful indices to read data files correctly
    quantity_idx = {"time": 0, "actions": 1, "energy": 2, "ergotropy": 3, "qubit_energy": 4,
                    "qubit_entropy": 5, "qubit_variance": 6, "qubit_purity": 7, "qubits_energy": 8,
                    "qubits_entropy": 9, "qubits_variance": 10, "cavity_energy": 11, "cavity_entropy": 12,
                    "cavity_variance": 13, "tot_energy": 14, "tot_variance": 15}
    hamil_type = ["tot", "cavity", "qubits"]

    #if necessary, I add the initial t=0 values
    if add_initial_null_value:
        #add values to the objectives array
        objs_data = np.vstack([np.zeros((1,objs_data.shape[1])),objs_data])
        #add values to the spectrum file
        new_entry = spectrum_weights_list[0:1,...] *0.
        spectrum_weights_list = np.concatenate([new_entry,spectrum_weights_list], axis=0)
        #set the weights to 1
        spectrum_weights_list[0,:,:,1] = 1. / spectrum_weights_list.shape[2]

    #prepare arrays for time plotting
    min_t, max_t = np.min(objs_data[:, quantity_idx["time"]]), np.max(objs_data[:, quantity_idx["time"]])
    t_smooth = np.linspace(min_t, max_t, 100)

    #create the matrix that will be plotted. rows are energies, columns are timesteps
    img_mat = np.zeros((len(energy_eigspaces), len(spectrum_weights_list)))

    #create the time-step array 
    t_vals =[] 
    #loop over timesteps
    for i, spectrum_weight in enumerate(spectrum_weights_list):
        #get weights and energies for the current timestep
        weights = spectrum_weight[hamil_index,:,1]
        energies = spectrum_weight[hamil_index,:,0]
        
        #add the current timestep
        t_vals.append(objs_data[i,quantity_idx["time"]])
        
        #loop over the eigenenergies
        counted_states = 0
        for j, eig_energy in enumerate(energy_eigspaces):
            #get a mask corresponding to the current eigenenergy
            mask = np.isclose(energies, eig_energy)

            #sum all the weights with this energy and put it in the matrix
            img_mat[j,i] = np.sum(weights[mask])

            #update how many states i counted, to check that I didnt miss any
            counted_states += np.sum(mask)

        #check that i got all states. If not, it measn that energy_eigspaces didnt have all energies
        assert counted_states == len(energies)

    #convert t_vals to a numpy array
    t_vals = np.array(t_vals)
    
    #rescale time axis multiplying the time by g
    t_vals *= g

    # Plot the result
    im = axis.imshow(np.log10(img_mat+1.e-8), extent=[t_vals[0], t_vals[-1], energy_eigspaces[0], energy_eigspaces[-1]],
                      cmap=plt.cm.jet, interpolation="bilinear", aspect='auto',  origin="lower", vmin=-2., vmax=0.,) 
        
    #add the mean and std
    h_str = hamil_type[hamil_index]
    energy_interp = scipy.interpolate.interp1d(objs_data[:, quantity_idx["time"]],
                                            objs_data[:, quantity_idx[f"{h_str}_energy"]], kind="cubic")
    std_interp = scipy.interpolate.interp1d(objs_data[:, quantity_idx["time"]],
                            np.sqrt(objs_data[:, quantity_idx[f"{h_str}_variance"]]), kind="cubic")
    axis.plot(g*t_smooth,energy_interp(t_smooth), c="black", linewidth=1.3)
    axis.fill_between(g*t_smooth, energy_interp(t_smooth) -0.5*std_interp(t_smooth),
                        energy_interp(t_smooth) +0.5*std_interp(t_smooth), color="black", alpha=0.16)
    
    #return quantity for colorbar
    return im

#functions mainly used internally

def sac_logs_file_location(log_dir, running_reward_file, running_loss_file,
    running_multi_obj_file, actions_file): 
    """
    Returns the location of the logging files. If they are passed, it doesn't change them.
    If they are None, it returns the default location

    Args:
        log_dir (str): location of the logging folder
        running_reward_file (str): location of reward file. If None, default location is returned
        running_loss_file (str): location of loss file. If None, default location is returned
        running_multi_obj_file (str): location if multi obj file. If none, default location is returned
        actions_file (str): location of actions file. If None, default location is returned

    Returns:
        running_reward_file (str): location of reward file
        running_loss_file (str): location of loss file
        running_multi_obj_file (str): location of multi objective file
        actions_file (str): location of actions file
    """

    sac_module = sac_epi
    if running_reward_file is None:
        running_reward_file = os.path.join(log_dir, sac_module.SacTrain.RUNNING_REWARD_FILE_NAME)
    if running_loss_file is None:
        running_loss_file = os.path.join(log_dir, sac_module.SacTrain.RUNNING_LOSS_FILE_NAME)
    if running_multi_obj_file is None:
        running_multi_obj_file = os.path.join(log_dir, sac_module.SacTrain.RUNNING_MULTI_OBJ_FILE_NAME)
    if actions_file is None:
        actions_file = os.path.join(log_dir, sac_module.SacTrain.ACTIONS_FILE_NAME)
    return (running_reward_file, running_loss_file, running_multi_obj_file, actions_file)

def plot_running_reward_on_axis(file_location, axis, plot_to_file_line = None,ylabel = None,
                                xlabel = None, xticks = None, xticklabels=None, yticks = None, yticklabels=None,
                                k_notation=True, linewidth=None, custom_color=None, lines_to_mark = None, custom_mark_color = None,
                                ylim=None,plot_extra_args=None,plot_extra_kwargs=None,legend_labels=None):
    """
    Produces a plot of the running reward on a given matplot lib axis

    Args:
        file_location (str): location of the file with the running reward
        axis (matplotlib axis): the axis on which to do the plot
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
        ylabel (str): custom string for y axis
        xlabel (str): custom string for x axis
        xticks (list(float)): custom list of x ticks
        xticklabels (list(str)): custom list of x tick strings
        yticks (list(float)): custom list of y ticks
        yticklabels (list(str)): custom list of y tick strings
        k_notation (bool): if True, displays number of x axis using "k" to represent 1000
        linewidth (float): linewidth of line
        custom_color: color of the line
        lines_to_mark (list(int)): adds a circle around the points corresponding to the specified lines in the file
        custom_mark_color: color of the circles at the points specified by lines_to_mark
        ylim (tuple(int)): ylim delimiter
        plot_extra_args: will call the function axis.plot passing in these custom args
        plot_extra_kwargs: will call the function axis.plot passing in plot_extra_args and plot_extra_kwargs
        legend_labels (list(str)): list of strings for the legend labels
    """
    #setup the plot
    if legend_labels is None:
        label_iter = itertools.cycle([None])
    else:
        label_iter = itertools.cycle(legend_labels)
    if ylabel is None:
        ylabel = "$G$"
    if xlabel is None:
        xlabel = "step"
    
    #load the data
    plot_data = np.loadtxt(file_location).reshape(-1,2)
    if lines_to_mark is not None:
        points_to_mark = plot_data[lines_to_mark]
    if plot_to_file_line is not None:
        plot_data = plot_data[:plot_to_file_line+1]
    
    #perform the plot and set the labels
    axis.plot(plot_data[:,0],plot_data[:,1], linewidth=linewidth, color=custom_color, label=next(label_iter))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    
    #additional details of the plot
    if k_notation:
        axis.xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
    if lines_to_mark is not None:
        axis.scatter(points_to_mark[:,0],points_to_mark[:,1], color=custom_mark_color)
    if ylim is not None:
        axis.set_ylim(ylim)
    if plot_extra_args is not None:
        if plot_extra_kwargs is None:
            axis.plot(*plot_extra_args, label=next(label_iter))
        else:
            axis.plot(*plot_extra_args, **plot_extra_kwargs, label=next(label_iter))
    if xticks is not None:
        axis.set_xticks(xticks)
    if xticklabels is not None:
        axis.set_xticklabels(xticklabels)
    if yticks is not None:
        axis.set_yticks(yticks)
    if yticklabels is not None:
        axis.set_yticklabels(yticklabels)
    if legend_labels is not None:
        axis.legend(loc="best",fancybox=True, framealpha=0.,borderaxespad=  0.1,handlelength=1.1, ncol=1)

def plot_running_multi_obj_on_axes(file_location, axes, plot_to_file_line = None):
    """
    Produces a plot of the running multiple objective, putting each quantity on a different axis.
    axes must contain the correct number of axis.

    Args:
        file_location (str): location of the file with the running reward
        axes (matplotlib axis): list of axis for each objective
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
    """
    #load the data
    plot_data = np.loadtxt(file_location)
    if len(plot_data.shape) == 1:
        plot_data = plot_data.reshape(1,-1)
    if plot_to_file_line is not None:
        plot_data = plot_data[:plot_to_file_line+1]

    #loop over each quantity to plot
    for i,axis in enumerate(axes):
        #perform the plot and set the labels
        axis.plot(plot_data[:,0],plot_data[:,i+1])
        axis.set_xlabel("step")
        axis.set_ylabel(f"Obj {i}")

def plot_running_loss_on_axis(file_location, axes, plot_to_file_line = None):
    """
    Produces a series of plots with the quantities logged in the running losses file. Based on
    the number of axes provided, it plots the Q running loss, the Pi running loss, the alpha temperature
    parameters, and the entropy of the policy
    Args:
        file_location (str): location of the file with the running losses
        axes (list(matplotlib axis)): a list of axis on which to plot the quantities mentioned above
        plot_to_file_lin (int): plot data up to this file line. If None, plots till the end
    """
    #load data
    plot_data = np.loadtxt(file_location).reshape(-1,len(axes)+1)
    if plot_to_file_line is not None:
        plot_data = plot_data[:plot_to_file_line+1]
    #plot q loss on first axis
    if len(axes) > 0:
        axes[0].set_yscale("log")
        axes[0].plot(plot_data[:,0],plot_data[:,1])
        axes[0].set_xlabel("steps")
        axes[0].set_ylabel("Q Running Loss")
    #plot pi loss on second axis
    if len(axes) > 1:
        axes[1].plot(plot_data[:,0],plot_data[:,2])
        axes[1].set_xlabel("steps")
        axes[1].set_ylabel("Pi Running Loss")
    #plot alpha on third axis
    if len(axes) > 2:
        axes[2].set_yscale("log")
        axes[2].plot(plot_data[:,0],plot_data[:,3])
        axes[2].set_xlabel("steps")
        axes[2].set_ylabel("alpha")
    #plot the entropy on fourth axis
    if len(axes) > 3:
        axes[3].plot(plot_data[:,0],plot_data[:,4])
        axes[3].set_xlabel("steps")
        axes[3].set_ylabel("entropy")
    #plot the rest (if present)
    if len(axes) > 4:
        for i in range(4, len(axes)):
            axes[i].plot(plot_data[:,0],plot_data[:,i+1])
            axes[i].set_xlabel("steps")
            axes[i].set_ylabel(f"loss {i}")

def plot_actions_on_axis(file_location, axis, actions_to_plot=1200, plot_to_file_line = None,
                        actions_ylim = None, ylabel = None, xlabel = None, xticks = None, xticklabels=None,
                        yticks = None, yticklabels=None, k_notation = True, constant_steps=False,
                        x_count_from_zero=False,  two_xticks=False, legend_lines=None,  legend_text=None, 
                        legend_location="best", legend_cols = None, rescale_action = False,
                        plot_actions_separately=False):
    """
    Produces a plot of the last chosen actions on a given matplot lib axis

    Args:
        file_location (str): location of the file with the actions
        axis (matplotlib axis): the axis on which to do the plot, or a list of them if plot_actions_separately=True
        actions_to_plot (int): how many actions to display in the plot
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
        actions_ylim (list(float)): delimiter for the y axis. If plot_actions_separately=True, this is a list of bounds
        ylabel (str): custom string for y axis
        xlabel (str): custom string for x axis
        xticks (list(float)): custom list of x ticks
        xticklabels (list(str)): custom list of x tick strings
        yticks (list(float)): custom list of y ticks
        yticklabels (list(str)): custom list of y tick strings
        k_notation (bool): if True, displays number of x axis using "k" to represent 1000
        constant_steps (bool): if True, it plots the actions as piecewise constant with dashed line
            for the jumps. Otherwise each action is just a dot
        x_count_from_zero (bool): if true, sets x axis from zero. Otherwise from step index
        two_xticks (bool): if True, it only places 2 x_ticks at smallest and largest value
        legend_lines(list(Line2D)): list of Line2D objects to generate the legend
        legend_text (list(str)): list of strings for the legend
        legend_location: matplotlib legend_location
        legend_cols (int): number of columns for the legend
        rescale_action(bool): if true, rescales actions between 0 and 1
        plot_actions_separately(bool): if true, it plots each action on a separate axis        
    """
    
    #set labels
    if ylabel is None:
        ylabel = "$u$"
    if xlabel is None:
        xlabel = "$t$" if x_count_from_zero else "step"   
    
    #handle separate panels for each action
    if plot_actions_separately:
        axes = axis
        actions_ylim_list = actions_ylim
    else:
        axes = [axis]
        actions_ylim_list = [actions_ylim]

    for i,ax in enumerate(axes):
        #prevent the scientific notation on plots
        ax.ticklabel_format(useOffset=False)
        #set y lims
        if actions_ylim is not None:
            ax.set_ylim(actions_ylim_list[i])
    
    #load data
    plot_data = np.loadtxt(file_location)
    if plot_to_file_line is None:
        plot_to_file_line = plot_data.shape[0]-1
    plot_to_file_line = min(plot_to_file_line,plot_data.shape[0]-1)
    actions_to_plot = min(actions_to_plot, plot_data.shape[0])
    data_to_plot = plot_data[(plot_to_file_line-actions_to_plot+1):(plot_to_file_line+1)]
    
    #plot actions
    if rescale_action:
        for i in range (1,data_to_plot.shape[1]):
            min_val = np.min(data_to_plot[:,i])
            max_val = np.max(data_to_plot[:,i])
            data_to_plot[:,i] = (data_to_plot[:,i]-min_val)/(max_val-min_val)

    if constant_steps:
        #if counting from zero, i replace the x values with growing numbers from zero
        if x_count_from_zero:
            data_to_plot[:,0] = np.arange(data_to_plot.shape[0])
        #compute the step to draw the length of the last point
        dt = data_to_plot[-1,0] - data_to_plot[-2,0]
        #first I do the vertical dahsed line, so they get covered
        for i in range(data_to_plot.shape[0]-1):
            axis.plot([data_to_plot[i+1,0],data_to_plot[i+1,0]],[data_to_plot[i,1],data_to_plot[i+1,1]], color="lightgray",
                linewidth=0.8)
        #horizontal lines
        data_to_plot = np.concatenate( (data_to_plot, np.array([[data_to_plot[-1,0]+dt] + (data_to_plot.shape[1]-1)*[0.] ]) ))
        for a in range(1,data_to_plot.shape[1]):
            for i in range(data_to_plot.shape[0]-1):  
                axis.plot([data_to_plot[i,0],data_to_plot[i+1,0]],[data_to_plot[i,a],data_to_plot[i,a]], color="black", linewidth=2.)
    else:
        for i in range(1,data_to_plot.shape[1]):
            if plot_actions_separately:
                temp_ax = axis[i-1]
            else:
                temp_ax = axis
            temp_ax.scatter(data_to_plot[:,0],data_to_plot[:,i], color = "black")
    #additional details of plot
    for ax in axes:
        ax.set_ylabel(ylabel)
        if xlabel != "":
            ax.set_xlabel(xlabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        elif two_xticks:
            ax.set_xticks([ data_to_plot[0,0], data_to_plot[-1,0] ])
        if yticks is not None:
            ax.set_yticks(yticks)
        if k_notation:
            ax.xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)

    #do the legend if necessary
    if legend_lines is not None:
        if legend_cols is None:
            legend_cols = len(legend_lines)
        axis.legend(legend_lines,legend_text,loc=legend_location,fancybox=False, framealpha=0.,borderaxespad=0.,
                    ncol=legend_cols,handlelength=0.5) 

def num_to_k_notation(tick, tex=True):
    """
    Used to produce ticks with a "k" indicating 1000

    Args:
        tick (float): value of the tick to be converted to a string
        tex (bool): if true, it will wrap the string in $$, making it look more "latex-like"

    Returns:
        tick_str (str): the string for the tick
    """
    tick_str = str(int(tick // 1000))
    end_not_stripped = str(int(tick % 1000))
    end = end_not_stripped.rstrip('0')
    if len(end) > 0:
        end = ("0"*(3-len(end_not_stripped))) + end
        tick_str += f".{end}"
    if tex:
        tick_str = "$" + tick_str + "$"
    tick_str += "k"
    return tick_str

def nearest_int(num):
    """ return the nearest integer to the float number num """
    return int(np.round(num))

def get_number_lines(running_reward_file, running_loss_file, action_count_file):
    """
    Returns the number of lines in the reward and loss files
    
    Args:
        running_reward_file (str): location of the reward file
        running_loss_file (str): location of the loss file
        action_count_file (str): location of the actions file

    Raises:
        NameError: if all files don't exist
    
    Returns:
        lines (int): number of lines in the file
     """
    if Path(running_reward_file).exists():
        data = np.loadtxt(running_reward_file).reshape(-1,2)
        return data.shape[0]
    if Path(running_loss_file).exists():
        data = np.loadtxt(running_loss_file).reshape(-1,2)
        return data.shape[0]
    if Path(action_count_file).exists():
        data = np.loadtxt(action_count_file).reshape(-1,2)
        return data.shape[0]
    raise NameError("No files to count lines")

def count_quantities(running_multi_obj_file):
    """
    counts the number of quantities in the multi_objective log. If it doesn't exist, 
    return 0.
    
    Args:
        running_multi_obj_file (str): file location

    Returns:
        quantities (int): number of quantities
    """
    if Path(running_multi_obj_file).exists():
        loaded_data = np.loadtxt(running_multi_obj_file)
        if len(loaded_data.shape) == 1:
            quantities = loaded_data.shape[0] - 1
        else:
            quantities = loaded_data.shape[1] - 1
    else:
        quantities = 0
    return quantities

