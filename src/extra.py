import time
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import os
import sac_epi
import plotting
import bz2
import _pickle as cPickle
import pickle
import shutil
import sac_epi_envs
import torch


"""
This module contains support and extra functions.
"""

class MeasureDuration:
    """ Used to measure the duration of a block of code.

    to use this:
    with MeasureDuration() as m:
        #code to measure
    """
    def __init__(self, what_str=""):
        self.start = None
        self.end = None
        self.what_str = what_str
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self):
        self.end = time.time()
        print(f"Time:  {self.duration()}  for {self.what_str}")
    def duration(self):
        return str((self.end - self.start)) + ' s'

@dataclass
class SacTrainState:
    """
    Data class. It is used internally by sac_epi.SacTrain to save the internal
    training state. When saving and loading a training session, it is pickled and unpickled.
    """
    device = None
    save_data_dir = None
    env_params = None
    training_hyperparams = None
    log_info = None
    state = None
    log_session = None
    steps_done = None
    running_reward = None
    running_loss = None
    actions = None
    running_multi_obj = None
    current_episode_return = None 
    current_episode_multi_obj = None

class LogSession(object):
    """
    Data object used to store the location of all the logging and training state data.
    It is used internally by sac_epi.SacTrain to handle training state saving 
    and logging.
    """
    def __init__(self, log_dir, state_dir, log_running_reward, log_running_loss, log_running_multi_obj,
                log_actions, running_reward_file, running_loss_file, running_multi_obj_file, actions_file):
        self.log_dir = log_dir
        self.state_dir = state_dir
        self.log_running_reward = log_running_reward
        self.log_running_loss = log_running_loss
        self.log_running_multi_obj = log_running_multi_obj
        self.log_actions = log_actions
        self.running_reward_file = running_reward_file
        self.running_loss_file = running_loss_file
        self.running_multi_obj_file = running_multi_obj_file
        self.actions_file = actions_file

def export_performance_and_protocol_files(main_dir, x_quantity, conditions_dict={},
                                quantities_for_indexing=[], show_single_plot=False, exp_folder_name="",
                                multiply_nc = None):
    """
    See 1_evaluate_and_export_performance.ipynb for details.
    Evaluates and exports data for multiple training folders in main_dir that satisfy the condition in conditions_dict.
    Typically, one would export a given value of nq. For each log satisfying the criteria, it will:
    - create a folder named exp_folder_name;
    - create a file in main_dir called "performance_{extra}.txt", where {extra} is the x_quantity, and all the parameters
        in conditions_dict, e.g. "nq={nq_val}".

    The file "performance_{extra}.txt" contains the performance across each training file. In particular, each row
    is a different value of the quantity x_quantity, and each column represents the same exact quantities in
    "objs_extra_data{extra}.txt" described below, but evaluated at the final time, and the value of the time-dependent
    control (index=1), is replaced with the return.
    
    The folder will contain multiple files. Let us denote with {extra} the parameters in quantities_for_indexing,
    and their value, e.g. {tau={tau_val}}, that distinguishes the log directories. For each training, the following
    files will be produced:
        - "eval_{extra}.pdf": plot with the return G, obj 0, i.e. the energy, obj 1, i.e. the ergotropy, and u,
            the time-dependent control, as a function of the training step. The final deterministic policy is used
            to choose the actions. This plot is also shown in this Jupyter Notebook during the execution.
        - "spectrum_weights_{extra}.pdf". Plot analyzing the performance of the final deterministic policy.
            First 3 panels correspond respectively to the projection of the quantum state onto the eigenstates of
            $\mathcal{\hat H}(t)$, the eigenstates of $\mathcal{\hat H}_{\rm C}$, and the eigenstates of 
            $\mathcal{\hat H}_{\rm B}$; then the y represents the energy, and the x is time, and the color is the square
            of the projection of the state onto the eigenstates. The following panels show the entropy of all the TLS,
            the entropy of a single TLS, the single TLS ergotropy, and the time dependent control as a function of time.
        - "train_{extra}.pdf": pdf plot representing the training behaviour. This corresponds to the plot outputted
            during training, and it is shown in the Jupyter Notebook during execution.
        - "objs_extra_data_{extra}.txt": text file with the performance of the final deterministic policy. Every row is
            a time-step, and each of the 16 columns reprents respectively [0 = time step, 1 = time-dependent control,
            2 = obj 0 (total battery energy), 3 = obj 1 (single TLS unit ergotropy), 4 = single TLS energy,
            5 = single TLS entropy, 6 = single TLS energy variance, 7 = single TLS purity, 8 = battery energy, 
            9 = battery entropy, 10 = battery energy variance, 11 = cavity energy, 12 = cavity entropy, 
            13 = cavity energy variance, 14 = total energy (including coupling), 15 = total energy variance]
        - "spectrum_weights_{extra}.npz": numpy compressed object with a "data" key. This is an array with shape
            (time_steps,hamil_index,spectrum,2). Time_steps indexes each timestep. hamil_index represents which
            hamiltonian we are projecting on, with 0=$\mathcal{\hat H}_{\rm C} + \mathcal{\hat H}_{\rm B}$,
            1=$\mathcal{\hat H}_{\rm C}$, 2=$\mathcal{\hat H}_{\rm B}$. Spectrum is an integer for each eigenstate of the
            corresponding Hamiltonian. The final index is 0=(energy value of the eingenstate), 1=(square projection of the
            state on the corresponding eigenstate).

    Args:
        main_dir(str): directory containing the training folders (log folders)
        x_quantity(str): name of the parameter distinguishing each training folder. "tau" is a good choice (see above)
        conditions_dict(dict): dictionary of parameter values and names. Only trainings satisfying these will be considered
            {"nq": f{nq_val}} is a common choice
        quantities_for_indexing(list): list of strings representing the parameter values used to distinguish the produced file
            (see above)
        show_single_plot(bool): if true, when running in a Jupyter Notebook, just the plot of the last training folder is 
            shown. Otherwise all plots are appended to the output.
        exp_folder_name(str): name of the folder that contains all files outputted for each folder individually
        multiply_nc(int): if specified, it evaluates the protocol performance creating a battery environment with a larger
            cutoff of the Fock space that is multiplied by this value.
    """
    det_data_list = []
    #get log dirs satisfying the right criteria
    log_dirs = log_dirs_given_criteria(main_dir, conditions_dict)
    
    #create the folder where files will be exported
    exp_folder = os.path.join(main_dir, exp_folder_name) 
    Path(exp_folder).mkdir(parents=True, exist_ok=True)
        
    #loop over each training
    for (i, log_dir) in enumerate(log_dirs): 
        print(f"Evaluating i = {i+1} of {len(log_dirs)}: {log_dir}")
        
        #load the data
        loaded_train = sac_epi.SacTrain()
        loaded_train.load_train(log_dir, no_train=True)
        
        #create the file extra str with all quantities_for_indexing
        extra_str = params_string_from_logdir(log_dir,quantities_for_indexing)
        
        #get dt
        dt = float(params_from_log_dir(log_dir)["dt"])
        
        #evaluate the model and export files
        det_eval = loaded_train.evaluate_current_policy(actions_to_plot=500, 
                    save_policy_to_file_name="det_policy.txt", actions_ylim=None,
                    suppress_show=False, dont_clear_output=not show_single_plot,
                    export_performance_pdf_location=os.path.join(exp_folder, f"eval{extra_str}.pdf"),
                    export_physical_quantities_location=os.path.join(exp_folder, f"objs_extra_data{extra_str}.txt"),
                    export_spectrum_weights_location = os.path.join(exp_folder, f"spectrum_weights{extra_str}.npz"),
                    export_spetrum_weights_pdf_location = os.path.join(exp_folder, f"spectrum_weights{extra_str}.pdf"),
                    dt=dt, multiply_nc=multiply_nc)

        #produce the training data plot
        plotting.plot_sac_logs(log_dir,actions_per_log=1000,rescale_action=False,
                            plot_actions_separately=True,dont_clear_output=not show_single_plot,
                            actions_ylim=[[loaded_train.env.action_space.low[i],loaded_train.env.action_space.high[i]]
                            for i in range(len(loaded_train.env.action_space.high))],
                            export_performance_pdf_location=os.path.join(exp_folder, f"train{extra_str}.pdf"))
        
        #append data
        model_parameters = params_from_log_dir(log_dir)
        det_data_list.append([model_parameters[x_quantity]] + list(det_eval))
    
    #save the performance to txt file
    np.savetxt(os.path.join(main_dir,f"performance_{x_quantity}{''.join([f'_{k}={v}' for (k,v) in conditions_dict.items()])}.txt"),
                            np.array(det_data_list, dtype=np.float32))

def create_best_of_multiple_runs(compare_folders, destination_folder, nq_vals):
    """
    This function allows to collect the best run out of multiple identical runs. In particular, if one uses
    0_train_coupling_case.ipynb or 0_train_detuning_case.ipynb to run multiple identical trainings, and then 
    exports the performance using export_performance_and_protocol_files as detailed in 
    1_evaluate_and_export_performance.ipynb, this function creates a new folder with the same type of exported files,
    but only selecting the best run, where best means the highest final ergotropy.

    It further create a file called ergo_mean_std_nq={nq_val}.txt for each value of nq that has different values of tau
    by row, and the 3 columns are the value of tau, the average ergotropy across mutiple runs, and the standard deviation
    of the ergotropy across multiple runs.

    Args:
        compare_folders(list(str)): list of the folders each containing a set of identical runs
        destination_folder(str): where the best outputs should be collected
        nq_vals(list): list of values of nq that were used during training. Only these will be compared
    """

    last_shape = None
    all_ergotropies = [] #first index is nq. Second is tau, and then the 4 repetitions as [nq, tau, ergo]

    #choose the best at each value of nq
    for nq in nq_vals:
        datas = []
        all_ergotropies.append([])
        #load all performance files
        for folder in compare_folders:
            #load data in current folder
            data = np.loadtxt(os.path.join(folder,f"performance_tau_nq={nq}.txt"))
            #sort it
            datas.append(data[data[:, 0].argsort()])
            #check that dimensions of files are all the same
            if last_shape is not None:
                assert last_shape == datas[-1].shape
            last_shape = datas[-1].shape
        
        #now that data is loaded, lets create the best file at this given nq
        best_performance = np.zeros_like(datas[0])
        #create folder to copy extra nq data
        current_nq_folder = os.path.join(destination_folder, f"nq={nq}")
        Path(current_nq_folder).mkdir(parents=True, exist_ok=True)
        #loop over each row
        for row_index in range(datas[0].shape[0]):
            all_ergotropies[-1].append([])
            best_data_index = -1
            best_ergotropy = -1.
            #search through the various files
            for data_index in range(len(datas)):
                if datas[data_index][row_index,1] > best_ergotropy:
                    best_ergotropy = datas[data_index][row_index,1]
                    best_data_index = data_index
                all_ergotropies[-1][-1].append([nq, datas[data_index][row_index,0],datas[data_index][row_index,1]])
            best_performance[row_index,:] = datas[best_data_index][row_index,:]
            
            #move files
            #since tau is rounded differently, i need to find the nearest
            target_tau = float(best_performance[row_index,0])
            tau_file_list = os.listdir(os.path.join(compare_folders[best_data_index], f"nq={nq}"))
            tau_str_vals = np.array([file_name.split("tau=")[1].split(".pdf")[0].split(".txt")[0].split(".npz")[0] for file_name in tau_file_list])
            tau_float_vals = tau_str_vals.astype(float)
            closest_index = np.argmax(np.isclose(tau_float_vals,target_tau,rtol = 1.e-3, atol = 1.e-3))
            tau = tau_str_vals[closest_index]
            for copy_file in [f"eval_tau={tau}.pdf",f"train_tau={tau}.pdf",f"objs_extra_data_tau={tau}.txt",
                              f"spectrum_weights_tau={tau}.npz", f"spectrum_weights_tau={tau}.pdf"]:
                shutil.copyfile(os.path.join(compare_folders[best_data_index], f"nq={nq}",copy_file), os.path.join(current_nq_folder,copy_file))
            np.savetxt(os.path.join(destination_folder, f"performance_tau_nq={nq}.txt"), best_performance)
        
    # now i compute the mean and standard deviation over the repetitions
    all_ergotropies = np.array(all_ergotropies)

    mean = np.mean(all_ergotropies[:,:,:,2], axis=2)
    std = np.std(all_ergotropies[:,:,:,2], axis=2, ddof=1)

    mean_reshape = mean.reshape(mean.shape + (1,))
    std_reshape = std.reshape(mean_reshape.shape)
    tau_reshape = all_ergotropies[:,:,0,1].reshape(mean_reshape.shape)

    tau_mean_std_array = np.concatenate([tau_reshape, mean_reshape, std_reshape], axis=2)

    #loop over the nq vals
    for i in range(all_ergotropies.shape[0]):
        np.savetxt(os.path.join(destination_folder, f"ergo_mean_std_nq={nq_vals[i]}.txt"), tau_mean_std_array[i])

def test_policy(env_class, env_params, policy,actions_to_plot=400,suppress_show=False,
                save_policy_to_file_name=None, actions_ylim=None,dont_clear_output=False,
                export_performance_pdf_location=None, export_physical_quantities_location = None, 
                export_spectrum_weights_location = None, export_spetrum_weights_pdf_location= None,
                dt=1., multiply_nc=None):
    """
    Function to evaluate the performance of a given policy. It creates a new instance of the environment and runs one
    episode on it using the given policy. If run in a jupyter notebook, it displays a plot with the reward, energy, and ergotropy
    and control as a function of time during the evaluation of the policy. It return the performance of the policy in terms 
    of the return, the multi objectives, and other physical quantities (see "Returns" below). It can also export txt, npz and 
    pdf files with further statistics and quantities (see "Args" below).

    shows the eval plot

    Args:
        env_class: class of the environment
        env_params(dict): dictionary of parameters to initialize the environment
        policy: the policy to test, i.e. a function taking a state as input, and outputting an action
        actions_to_plot (int): how many of the last actions to show in the plot
        suppress_show (bool): if False, it will show in a jupyter notebook the avg return, the multiobjectives, and the 
            and the last chosen actions
        save_policy_to_file_name (str): if specified, it will save the chosen actions to this file
        actions_ylim ((float,float)): y_lim for the plot of the chosen actions
        dont_clear_output (bool): if False, it will clear the previous plots produced by this function
        export_performance_pdf_location (str): if specified, it saves a plot with the sum of the rewards, the objectives
            (battery energy and TSL ergotropy),and the control, as a function of time.
        export_physical_quantities_location(str): if specified, it export a txt file with time-steps by row,
            and 16 columns representing respectively [0 = time step, 1 = time-dependent control,
            2 = obj 0 (total battery energy), 3 = obj 1 (single TLS unit ergotropy), 4 = single TLS energy,
            5 = single TLS entropy, 6 = single TLS energy variance, 7 = single TLS purity, 8 = battery energy,
            9 = battery entropy, 10 = battery energy variance, 11 = cavity energy, 12 = cavity entropy,
            13 = cavity energy variance, 14 = total energy (including coupling), 15 = total energy variance]
        export_spectrum_weights_location(str): if specified, it exports a numpy compressed object with a "data" key.
            This is an array with shape (time_steps,hamil_index,spectrum,2). Time_steps indexes each timestep.
            hamil_index represents which hamiltonian we are projecting on, with 0=$\mathcal{\hat H}(t)$,
            1=$\mathcal{\hat H}_{\rm C}$, 2=$\mathcal{\hat H}_{\rm B}$. Spectrum is an integer for each eigenstate of 
            the corresponding Hamiltonian. The final index is 0=(energy value of the eingenstate), 1=(square projection 
            of the state on the corresponding eigenstate).
        export_spetrum_weights_pdf_location(str): if specified, exports a plot where the first 3 panels correspond
            respectively to the projection of the quantum state onto the eigenstates of 
            $\mathcal{\hat H}_{\rm C} + \mathcal{\hat H}_{\rm B}$, the eigenstates of $\mathcal{\hat H}_{\rm C}$, and the
            eigenstates of $\mathcal{\hat H}_{\rm B}$; then the y represents the energy, and the x is time, and the color is
            the square of the projection of the state onto the eigenstates. The following panels show the entropy of all the
            TLS, the entropy of a single TLS, the single TLS ergotropy, and the time dependent control as a function of time.
            It will only work if also export_spectrum_weights_location is specified.
        dt(float): duration of a single time-step.
        multiply_nc(int): if specified, it increases the fock space cutoff multiplying it by this factor before evaluating 
            the performance.
        
    Returns:
        If there environment doesnt support multiple objectives
            return(float): the final return

        If it does support multiple objectives (which quantum battery environments should)
            performance(np.array(float)): 1D array containing the final time values of [0=reward, 1 = obj 0 (total battery energy),
                2 = obj 1 (single TLS unit ergotropy), 3 = single TLS energy, 4 = single TLS entropy, 5 = single TLS energy variance,
                6 = single TLS purity, 7 = battery energy, 8 = battery entropy, 9 = battery energy variance, 10 = cavity energy,
                11 = cavity entropy, 12 = cavity energy variance, 13 = total energy (including coupling), 14 = total energy variance]

    """
    #create an instance of the environment
    env = env_class(env_params)

    #create an extra environment, if necessary, with a larger cutoff nc
    if multiply_nc is not None:
        #load the environment
        eval_env_params = env_params.copy()
        eval_env_params["nc"] = int(env_params["nc"]*multiply_nc)
        eval_env = env_class(eval_env_params)

    #initialize variables to compute the running reward and mulit_obj without bias
    state = env.reset()
    done = False
    reward = 0.
    multi_obj = None
    actions = []
    running_rewards = []
    running_multi_objs = []
    running_physical_quantities = []
    running_spectrum_weights = []

    i = -1
    #execute one deterministic episode
    while not done:
        #do a step
        i += 1
        act = policy(state)
        state, ret, done, info_dict =  env.step(act)
        
        #if using a different env for evaluation
        if multiply_nc is not None:
            _, ret, _, info_dict = eval_env.step(act)
        
        #update the reward
        reward += ret
        running_rewards.append([i,reward])

        #update the actions
        actions.append([i] +  list(act))

        #update the multiobjective
        if "multi_obj" in info_dict:
            if multi_obj is None:
                multi_obj = np.zeros(len(info_dict["multi_obj"]))
            multi_obj += info_dict["multi_obj"]
            running_multi_objs.append([i] + list(multi_obj))

        #update the additional physical quantities
        if not (export_physical_quantities_location is None):
            if multiply_nc is None:
                running_physical_quantities.append(sac_epi_envs.return_physical_quantities(env))
            else:
                running_physical_quantities.append(sac_epi_envs.return_physical_quantities(eval_env))
        
        #update the spectrum and weights
        if not (export_spectrum_weights_location is None):
            if multiply_nc is None:
                running_spectrum_weights.append(sac_epi_envs.return_spectrum_weights(env))
            else:
                running_spectrum_weights.append(sac_epi_envs.return_spectrum_weights(eval_env))
        
    #if necessary, saves the chosen actions to file
    if save_policy_to_file_name is not None:
        f_actions_name = save_policy_to_file_name
        Path(f_actions_name).parent.mkdir(parents=True, exist_ok=True)
    else:
        f_actions_name = None

    #if necessary, export a file with all the additional physical quantities (see documentation above)
    if export_physical_quantities_location is not None:
        actions_np = np.array(actions)
        multi_obj_np = np.array(running_multi_objs)
        physical_quantities_np = np.array(running_physical_quantities)
        #convert step index to time before concatenating
        actions_np[:,0] += 1.
        actions_np[:,0] *= dt 
        #prepare data and save it
        data_to_save = np.concatenate([actions_np, multi_obj_np[:,1:], physical_quantities_np], axis=1)
        os.makedirs(os.path.split(export_physical_quantities_location)[0], exist_ok=True)
        np.savetxt(export_physical_quantities_location, data_to_save)

    #if necessary, export a file with spectrum and weights
    if export_spectrum_weights_location is not None:
        data_to_save = np.array(running_spectrum_weights)
        np.savez_compressed(export_spectrum_weights_location, data = data_to_save)
        #if necessary, I also save the pdf of this data
        if not export_spetrum_weights_pdf_location is None:
            plotting.save_spectrum_weights_plot_from_file(export_spectrum_weights_location, 
                                                 export_spetrum_weights_pdf_location)
    
    #show a plot of the performance, i.e. return, objectives, and control, as a function of time.
    #to do this, save data to a temp file in order to call the plotting functions which loads data from files
    #running reward file
    f_running_rewards = tempfile.NamedTemporaryFile()
    f_running_rewards_name = f_running_rewards.name
    f_running_rewards.close()
    #running actions file
    if f_actions_name is None:
        f_actions = tempfile.NamedTemporaryFile()
        f_actions_name = f_actions.name
        f_actions.close()
    np.savetxt(f_running_rewards_name, np.array(running_rewards))
    np.savetxt(f_actions_name, np.array(actions))
    #if its multi_objective, i save that file too
    if not running_multi_objs is None:
        f_running_multi_objs = tempfile.NamedTemporaryFile()
        f_running_multi_objs_name = f_running_multi_objs.name
        f_running_multi_objs.close()
        np.savetxt(f_running_multi_objs_name, np.array(running_multi_objs))
    #plot the data, and eventually save it as a pdf too
    plotting.plot_sac_logs(Path(f_running_rewards_name).parent.name, running_reward_file=f_running_rewards_name,
        running_loss_file=None,running_multi_obj_file= f_running_multi_objs_name, actions_file=f_actions_name,
        actions_per_log=1, plot_to_file_line = None, suppress_show=suppress_show, 
        actions_to_plot=actions_to_plot,actions_ylim=actions_ylim,dont_clear_output=dont_clear_output,
        k_notation=False, constant_actions_steps=False,plot_actions_separately=True,
        export_performance_pdf_location=export_performance_pdf_location)

    #return  performances
    return reward if multi_obj is None else np.concatenate([np.array([reward]), multi_obj, physical_quantities_np[-1,:]])

def clear_memory(train):
    """
    Frees memory after a training is complete

    Args:
        train (SacTrain): training object to be cleared
    """
    del train.s
    del train.env
    del train.memory
    del train.ac
    del train.ac_targ
    del train
    torch.cuda.empty_cache()

def log_dirs_given_criteria(main_dir, conditions_dict):
    """
    Returns a list of directories with logs satisfying all parameter requests given in conditions_dict

    Args:
        main_dir (str): location of the directory containing all log directories
        conditions_dict (dict): dictionary with all parameter requests to be satisfied

    Return:
        list of log directories satisfying the requests
    """
    ret_list = []
    #loop through all folders
    for sub_dir in os.listdir(main_dir):
        #current log directory
        log_dir = os.path.join(main_dir,sub_dir)
        #check if it's a folder and not a file
        if os.path.isdir(log_dir):  
            #check if the params.txt file exists 
            params_file = os.path.join(log_dir, sac_epi.SacTrain.PARAMS_FILE_NAME)
            if os.path.exists(params_file):
                #load all parameters in a dict
                params_dict = params_from_log_dir(log_dir)
                all_conditions_met = True
                #check if all conditions are met
                for key in conditions_dict:
                    if conditions_dict[key] != params_dict[key]:
                        all_conditions_met = False
                        break
                if all_conditions_met:
                    ret_list.append(log_dir)
    return ret_list

def print_last_rewards(log_dir, number_of_rewards=3):
    """
    Prints the last number_of_rewards in a given log_dir
    """
    file = os.path.join(log_dir, sac_epi.SacTrain.RUNNING_REWARD_FILE_NAME)
    data = np.loadtxt(file)
    last_rewards = data[-number_of_rewards:,1]
    print(f"last rewards: {last_rewards}; avg reward: {np.mean(last_rewards)}")

def params_from_log_dir(log_dir):
    """
    given a log_dir, it returns a dictionary with all the parameters loaded.
    Both key and value are strings
    """
    #initialize dict
    params_dict = {}
    #load a dictionary with all parameters
    params_file = os.path.join(log_dir, sac_epi.SacTrain.PARAMS_FILE_NAME)
    params_np = np.genfromtxt(params_file, dtype=str, delimiter=":\t")
    for (key, value) in params_np:
        params_dict[key] = value
    return params_dict

def pickle_data(file_location, data):
    """
    saved an object to file compressing it with bz2

    Args:
        file_location (str): where the file should be stored
        data(obj): data to be saved
    """
    with bz2.BZ2File(file_location, "w") as f: 
        cPickle.dump(data, f)

def unpickle_data(file, uncompressed_file = None):
    """
    Loads the file and return the unpickled data.

    Args:
        file(str): the compressed file to load
        uncompressed_file(str): if specified, it loads it as an uncompressed file if it cant find file
    """
    #if the compressed file exists
    if os.path.exists(file):
        data = bz2.BZ2File(file, "rb")
        data = cPickle.load(data)
    else:
        #if the uncompressed file must be loaded (backward compatibility)
        with open(uncompressed_file, 'rb') as input:
            data = pickle.load(input)

    return data
        
def params_string_from_logdir(log_dir, params_list):
    """
    Give a log_dir, it returns a string with the parameter names in params_list
    and their corresponding value

    Args:
        log_dir(str): location of the log dir
        params_list(list): list of parameters (strings) to be included in the string
    """
    my_str = ""
    params_dict = params_from_log_dir(log_dir)
    for param in params_list:
        my_str += f"_{param}={params_dict[param]}"
    return my_str
        