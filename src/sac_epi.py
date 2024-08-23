from __future__ import print_function
import os
import numpy as np
import torch
import torch.optim as optim
import shutil
import sys
import warnings
import logging
from itertools import chain
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import plotting
import core_epi
import sac_epi_envs
import extra

"""
This mudule contains the objects used to train the charging of quantum batteries with an arbitrary
number of continuous controls (wrapped into a 1D array).
All torch tensors that are not integers are torch.float32.
It was written making many changes from the code:
J. Achiam, Spinning Up in Deep Reinforcement Learning, https://github.com/openai/spinningup (2018).
Furthermore, automatic choosing of temperature is implemented according to: https://arxiv.org/abs/1812.05905
"""


def state_to_tensor(state, device):
    """ Coverts a numpy state to a torch.tensor with the right dimension """
    return torch.as_tensor(state, device=device, dtype=torch.float32).view(-1)

def action_to_tensor(state, device):
    """ Coverts a numpy action to a torch.tensor with the right dimension """
    return torch.as_tensor(state, device=device, dtype=torch.float32).view(-1)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents with continuous actions.

    Args:
        obs_dim (int): number of continuous parameters of observation space.
        act_dim (int): number of continuous parameters of action space.
        size (int): size of the buffer.
        device (torch.device): which torch device to use.
        objectives(int): number of multi-objectives.
    """

    def __init__(self, obs_dim, act_dim, size, device, objectives=1):  
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.obs2_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.objs_buf = torch.zeros((size, objectives), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, objs, next_obs, done):
        """
        stores a transition into the buffer. All args are torch.float32.

        Args:
            obs(torch.tensor): the initial state
            act(torch.tensor): the continuous action
            objs(torch.tensor): the multi-objectives received
            next_obs(torch.tensor): the next state   
            done(bool): whether the episode is over or not     
        """
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        #the reshape ensures that it's an array even if just a single number
        self.objs_buf[self.ptr] = torch.tensor(objs.reshape(-1), dtype=torch.float32, device=self.device)
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        """
        Return a random batch of experience from the buffer.
        The batch index is the leftmost index.

        Args:
            batch_size (int): size of batch
        """
        idxs = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     objs=self.objs_buf[idxs],
                     done=self.done_buf[idxs])
        return batch

class SacTrain(object):
    """
    Main class to train the RL agent on a quantum battery environment
    with an arbitrary number of continuous actions.
    This class can create a new training session, or load an existing one.
    It takes care of logging and of saving the training session all in 1 folder.

    Usage:
        After initialization either
        - call initialize_new_train() to initialize a new training session
        - call load_train() to load an existing training session
    """
        
    #define some constants defining the filestructure of the logs and saved state.
    PARAMS_FILE_NAME = "params.txt"
    S_FILE_NAME = "s.dat"
    S_FILE_NAME_BZ2 = "s_bz2.dat"
    MEMORY_FILE_NAME = "memory.dat"
    POLICY_NET_FILE_NAME = "policy_net.dat"
    TARGET_NET_FILE_NAME = "target_net.dat"
    ERROR_LOG_FILE_NAME = "caught_errors.log"
    STATE_FOLDER_NAME = "state"
    SAVE_DATA_DIR = os.path.join("..", "data")
    SAVED_LOGS_FOLDER = "logs"
    RUNNING_REWARD_FILE_NAME = "running_reward.txt"
    RUNNING_LOSS_FILE_NAME = "running_loss.txt"
    RUNNING_MULTI_OBJ_FILE_NAME = "running_multi_obj.txt"
    ACTIONS_FILE_NAME = "actions.txt"
    SAVED_POLICY_DIR_NAME = "saved_policies"
    
    #Methods that can be called:

    def initialize_new_train(self, env_class, env_params, training_hyperparams, log_info):
        """ Initializes a new training session. Should be called right after initialization.

        Args:
            env_class (gym.Env): class representing the quantum battery environment
            env_params (dict): parameters used to initialize env_class. See specific env requirements.
            training_hyperparams (dict): dictionary with training hyperparameters. Must contain the following
                "BATCH_SIZE" (int): batch size
                "LR" (float): learning rate to update the value and policy functions
                "ALPHA_LR" (float): learning rate to update the alpha parameter (the SAC temperature)
                "H_START" (float): initial value of the entropy of the policy
                "H_END" (float): final value of the entropy of the policy
                "H_DECAY" (float): the entropy decays exponentially from H_START to H_END with this typical width
                "C_START" (np.array(float32)): array whose length is the number of multiobjectives. It corresponds
                    to the initial weights assigned to each multiobjective
                "C_END" (np.array(float32)): same size as C_START. Final values of the multiobjective weights.
                "C_MEAN" (float): the multiobjective weights vary from C_START to C_END according to a Fermi distribution
                    with mean C_MEAN and width C_WIDTH (in training steps units)
                "C_WIDTH" (float): see above 
                "REPLAY_MEMORY_SIZE" (int): size of replay buffer
                "POLYAK" (float): polyak coefficient
                "LOG_STEPS" (int): save logs and display training every number of steps
                "GAMMA" (float): RL discount factor
                "RETURN_GAMMA" (float): when plotting the return during training, a running exponential average with
                    with this parameter is used
                "LOSS_GAMMA" (float): when plotting the loss functions during training, a running exponential average with
                    with this parameter is used
                "HIDDEN_SIZES" tuple(int): size of hidden layers 
                "SAVE_STATE_STEPS" (int): saves complete state of trainig every number of steps
                "INITIAL_RANDOM_STEPS" (int): number of initial uniformly random steps
                "UPDATE_AFTER" (int): start minimizing loss function after initial steps
                "UPDATE_EVERY" (int): performs this many updates every this many steps
                "USE_CUDA" (bool): use cuda for computation
                "MIN_COV_EIGEN" (float): small value to stabilize the covariance matrix of the multivariate distribution 
                    used in the policy function. See core_epi.MLPActorCritic for details
                "DONT_SAVE_MEMORY" (bool): if True, it won't save the replay buffer when saving the state of the training.
                    This means that training cannot be later resumed. However, the policy can still be loaded for evaluation.
                    Not saving the replay buffer saves space on disk.
            log_info (dict): specifies logging info. Must contain
                    "log_running_reward" (bool): whether to log the running return (True is recommended) 
                    "log_running_loss" (bool):  whether to log the running loss (True is recommended)
                    "log_actions" (bool):  whether to log the actions (True is recommended)
                    "extra_str" (str): extra string to append to training folder name
        """
        #initialize a SacTrainState to store the training state 
        self.s = extra.SacTrainState()

        #save input parameters
        self.s.save_data_dir = self.SAVE_DATA_DIR
        self.s.env_params = env_params
        self.s.training_hyperparams = training_hyperparams
        self.s.log_info = log_info 

        #add a default value for backward compatibility
        if not "GAMMA" in self.s.training_hyperparams:
            self.s.training_hyperparams["GAMMA"] = 1. 

        #setup the torch device
        if self.s.training_hyperparams["USE_CUDA"]:
            if torch.cuda.is_available():
                self.s.device = torch.device("cuda")
            else:
                warnings.warn("Cuda is not available. Will use cpu instead.")
                self.s.device = torch.device("cpu")
        else:
            self.s.device = torch.device("cpu")

        #create the rl environment
        self.env = env_class(self.s.env_params)

        #add the environment name to the env_params dictionary
        self.s.env_params["env_name"] = self.env.__class__.__name__

        #set the training steps_done to zero
        self.s.steps_done = 0

        #set  the return of the current episode to zero
        self.s.current_episode_return = 0.
        self.s.running_reward = 0.

        #reset the environment and save the initial state
        self.s.state = state_to_tensor(self.env.reset(), self.s.device)

        #initialize logging session
        self.s.log_session = self.initialize_log_session()

        #setup the memory replay buffer
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.memory = ReplayBuffer(obs_dim, act_dim, self.s.training_hyperparams["REPLAY_MEMORY_SIZE"],
                                     self.s.device, self.get_obj_num())

        #initialize the NNs
        self.initialize_nns()

        #setup the optimizer
        self.create_optimizer()

        #create the weight scheduler
        self.create_weight_scheduler()

    def load_train(self, log_folder, specific_state_folder = None, no_train=False):
        """
        Loads a training session that had been previously saved. The training
        sessions are saved as folders numbered as "0", "1",... By default, the latest
        one is loaded, but this can be overridden by specific_state_folder
        
        Args:
            log_folder (str): folder of the training session
            specific_state_folder (str): can load a specific save. If None, loads the latest one.
            no_train (bool): if False, it creates a new logging folder where all new saves and loggings
                are located. If true, it doesn't create a new folder, but cannot train anymore. If the 
                replay buffer was not saved, this must be True.    
        """
        save_dir_path = os.path.join(log_folder, self.STATE_FOLDER_NAME)
        if specific_state_folder is not None:
            #load the folder where the save is
            save_dir_path = os.path.join(save_dir_path, specific_state_folder)
        else:
            #must find the latest folder if not specificed
            path = Path(save_dir_path)
            folders = [dir.name for dir in path.iterdir() if dir.is_dir()]
            index = int(folders[0])
            for folder in folders:
                index = max(index, int(folder))
            save_dir_path = os.path.join(save_dir_path, str(index))
        
        #load self.s
            self.s = extra.unpickle_data(os.path.join(save_dir_path, self.S_FILE_NAME_BZ2),
                                    uncompressed_file = os.path.join(save_dir_path, self.S_FILE_NAME))
    
        #load the memory
        if not no_train:
            #for back compatibility, check if the memory is in a separate file or not
            #if it's new
            if os.path.exists(os.path.join(save_dir_path, self.MEMORY_FILE_NAME)):        
                #load memory
                self.memory = extra.unpickle_data(os.path.join(save_dir_path, self.MEMORY_FILE_NAME))
            else:
                self.memory = self.s.memory
            
        #add a default value for backward compatibility
        if not "GAMMA" in self.s.training_hyperparams:
            self.s.training_hyperparams["GAMMA"] = 1.  

        #load the environment
        env_method = self.return_env_class_from_name()
        self.env = env_method(self.s.env_params)
        try:
            self.env.set_current_state(self.s.state.cpu().numpy())
        except:
            self.env.set_current_state(self.s.state)

        #create the nns
        self.initialize_nns()
        
        #load the policy net
        self.ac.load_state_dict(torch.load(os.path.join(save_dir_path, self.POLICY_NET_FILE_NAME)))
        
        #create the weight scheduler
        self.create_weight_scheduler()

        #i stop here if i don't want to further optimize the model
        if no_train:
            self.s.log_session.log_dir = log_folder
        else:
            #load the target net
            self.ac_targ.load_state_dict(torch.load(os.path.join(save_dir_path, self.TARGET_NET_FILE_NAME)))
            #load the optimizer
            self.create_optimizer()
            #now that everything is loaded, i create a new LogSession, and copy in the old logs
            self.s.log_session = self.initialize_log_session(reset_running_vars = False)
            for file in Path(os.path.join(save_dir_path, self.SAVED_LOGS_FOLDER)).iterdir():
                shutil.copy(str(file), os.path.join(self.s.log_session.log_dir, file.name))

    def train(self, steps, output_plots = True):
        """
        Runs "steps" number of training steps. Takes care of saving and logging. It can be called multiple
        times and it will keep training the same model.

        Args:
            steps (int): number of training steps to perform
            output_plots (bool): if true, it will output a plot with all the running logs every LOG_STEPS.
        """
        
        for _ in range(steps):  
            
            #choose an action (random uniform for first INITIAL_RANDOM_STEPS, then according to policy )
            if self.s.steps_done > self.s.training_hyperparams["INITIAL_RANDOM_STEPS"]:
                a = self.get_action(self.s.state)
            else:
                a = action_to_tensor(self.env.action_space.sample(), self.s.device)

            #perform the action on environment
            o2_np, r, d, info_dict = self.env.step(a.cpu().numpy())
            o2 = state_to_tensor(o2_np,self.s.device)
            
            # Store experience to replay buffer
            if "multi_obj" in info_dict:
                #if i get at least 2 objectives, i store them instead of the reward
                if len(info_dict["multi_obj"]) > 1:
                    data = info_dict["multi_obj"]
                else:
                    data = np.array([r], dtype=np.float32)
            else:
                data = np.array([r], dtype=np.float32)
            self.memory.store(self.s.state, a, data, o2, d)

            #compute the "real reward" as a weighed average of the objectives
            r = np.tensordot(self.weight_scheduler(self.s.steps_done, return_numpy_cpu=True), data, axes=([0],[0]))
            
            #move to the next state
            self.s.state = o2

            #increase the step counter
            self.s.steps_done += 1

            # Perform NN parameters updates
            if self.s.steps_done > self.s.training_hyperparams["UPDATE_AFTER"] and \
                self.s.steps_done % self.s.training_hyperparams["UPDATE_EVERY"] == 0:

                for _ in range(self.s.training_hyperparams["UPDATE_EVERY"]):
                    #collect a batch of experience to use for training
                    batch = self.memory.sample_batch(self.s.training_hyperparams["BATCH_SIZE"])
                    try:
                        #perform the update using the batch
                        q_loss, pi_loss, entropy = self.update(data=batch)
                        #update logging: running loss
                        self.s.running_loss[0] += (1.-self.s.training_hyperparams["LOSS_GAMMA"])*(q_loss - self.s.running_loss[0])
                        self.s.running_loss[1] += (1.-self.s.training_hyperparams["LOSS_GAMMA"])*(pi_loss - self.s.running_loss[1])
                        self.s.running_loss[2] += (1.-self.s.training_hyperparams["LOSS_GAMMA"])*(self.current_alpha() - self.s.running_loss[2])
                        self.s.running_loss[3] += (1.-self.s.training_hyperparams["LOSS_GAMMA"])*(entropy - self.s.running_loss[3])
                        
                    except RuntimeError as e:
                        #there could be an error doing updates, e.g. covariance singular. In such case i log it
                        logging.error(f"Exception at step {self.s.steps_done} during self.update: {e}")

            #update the return of the current episode
            self.s.current_episode_return += r

            #if present, update the sum of the multiobjectives of the current episode
            if "multi_obj" in info_dict:
                if self.s.current_episode_multi_obj is None:
                    self.s.current_episode_multi_obj = np.zeros(len(info_dict["multi_obj"]) ,dtype=np.float32)
                self.s.current_episode_multi_obj += info_dict["multi_obj"]

            #update actions list
            self.s.actions.append([self.s.steps_done] + a.tolist() ) 

            #handle end of episode
            if d:
                #update running average of total reward
                self.s.running_reward += (1.-self.s.training_hyperparams["RETURN_GAMMA"])*(
                    self.s.current_episode_return - self.s.running_reward) 
                #reset total reward
                self.s.current_episode_return = 0.
                #if present, update running average of total multi objectives
                if "multi_obj" in info_dict:
                    if self.s.running_multi_obj is None:
                        self.s.running_multi_obj = np.zeros(len(info_dict["multi_obj"]) ,dtype=np.float32)
                    self.s.running_multi_obj += (1.-self.s.training_hyperparams["RETURN_GAMMA"])*(self.s.current_episode_multi_obj
                                                                                            -self.s.running_multi_obj  )
                #reset the total multi obj
                self.s.current_episode_multi_obj = np.zeros(len(info_dict["multi_obj"]) ,dtype=np.float32)
                #reset the environment
                self.s.state = state_to_tensor(self.env.reset(), self.s.device)

            #if its time to log
            if self.s.steps_done % self.s.training_hyperparams["LOG_STEPS"] == 0 :
                #update log files
                self.update_log_files()
                
                #plot the logs
                if output_plots:
                    self.plot_logs()

            #if it's time to save the full training state   
            if self.s.steps_done % self.s.training_hyperparams["SAVE_STATE_STEPS"] == 0:
                self.save_full_state()

    def save_full_state(self):
        """
        Saves the full state to file, such that the policy can be later loaded, and if DONT_SAVE_MEMORY=False,
        it is also possible to keep training later.
        The saved session is placed in a folder inside STATE_FOLDER_NAME, named using an ascending index
        0, 1, ... Largest index is the most recent save.
        """
        #folder where the session is saved
        path_location = os.path.join(self.s.log_session.state_dir, str(len(list(Path(self.s.log_session.state_dir).iterdir()))))
        #create the folder to save the state
        Path(path_location).mkdir(parents=True, exist_ok=True)
        #save self.s state object
        extra.pickle_data(os.path.join(path_location, self.S_FILE_NAME_BZ2), self.s)    
        #save memory object
        if "DONT_SAVE_MEMORY" in self.s.training_hyperparams:
            dont_save_memory = self.s.training_hyperparams["DONT_SAVE_MEMORY"]
        else:
            dont_save_memory = False
        if not dont_save_memory:
            extra.pickle_data(os.path.join(path_location, self.MEMORY_FILE_NAME), self.memory)    
        #save policy_net params
        torch.save(self.ac.state_dict(), os.path.join(path_location, self.POLICY_NET_FILE_NAME))
        #save target_net params
        torch.save(self.ac_targ.state_dict(), os.path.join(path_location, self.TARGET_NET_FILE_NAME))
        #copy over the logging folder 
        saved_logs_path = os.path.join(path_location, self.SAVED_LOGS_FOLDER)
        Path(saved_logs_path).mkdir(parents=True, exist_ok=True)
        for file in Path(self.s.log_session.log_dir).iterdir():
            if not file.is_dir() :
                shutil.copy(str(file), os.path.join(saved_logs_path, file.name))

    def evaluate_current_policy(self, suppress_show=False,actions_to_plot=400,
                                save_policy_to_file_name = None,actions_ylim=None,dont_clear_output=False,
                                export_performance_pdf_location=None,export_physical_quantities_location=None,
                                export_spectrum_weights_location=None,export_spetrum_weights_pdf_location=None,
                                dt=1., multiply_nc=None):
        """
        Function that evaluates the performance of the current policy. It can show the performance in jupyter notebook, and save
        many statistic and properties as .txt, .npz and .pdf plots. See extra.test_policy for all details.

        Args:
            suppress_show (bool): if False, it will show in a jupyter notebook the avg return, the multiobjectives, and the 
                and the last chosen actions
            actions_to_plot (int): how many of the last actions to show in the plot
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
        if save_policy_to_file_name is not None:
            save_policy_to_file_name = os.path.join( self.s.log_session.log_dir, self.SAVED_POLICY_DIR_NAME, save_policy_to_file_name)
        
        if actions_ylim is None:
            actions_ylim = [[self.env.action_space.low[i],self.env.action_space.high[i]]
                            for i in range(len(self.env.action_space.high))]
        #evaluates the policy
        return extra.test_policy(self.return_env_class_from_name(), self.s.env_params,
                     lambda o: self.get_action(torch.as_tensor(o,dtype=torch.float32,device=self.s.device),
                     deterministic=True).cpu().numpy(), False,suppress_show=suppress_show,
                     actions_to_plot=actions_to_plot,dont_clear_output=dont_clear_output,
                     save_policy_to_file_name=save_policy_to_file_name, actions_ylim=actions_ylim,
                     export_performance_pdf_location=export_performance_pdf_location,
                     export_physical_quantities_location=export_physical_quantities_location,
                     export_spectrum_weights_location=export_spectrum_weights_location,
                     export_spetrum_weights_pdf_location=export_spetrum_weights_pdf_location,dt=dt,
                     multiply_nc=multiply_nc)

    #Methods that should only be used internally:

    def initialize_log_session(self, reset_running_vars = True):
        """
        creates a folder, named with the current time and date, for logging and saving a training session,
        and saves all the physical parameters and hyperparameters in file PARAMS_FILE_NAME. 
        
        Args:
            reset_running_vars (bool): wether to reset the logged data

        Raises:
            Exception: if the folder for logging already exists
        
        Returns:
            log_session (extra.LogSession): info used by this class to do logging and saving state in the right place
        """
        #reset the running variables
        if reset_running_vars:
            self.s.running_reward = 0.
            self.s.running_loss = np.zeros(4, dtype=np.float32)
            self.s.running_multi_obj = None
            self.s.actions =[]

        #create folder for logging
        now = datetime.now()
        log_dir = os.path.join(self.s.save_data_dir, now.strftime("%Y_%m_%d-%H_%M_%S") + self.s.log_info["extra_str"])
        Path(log_dir).mkdir(parents=True, exist_ok=False)
            
        #create a file with all the environment params and hyperparams
        param_str = ""
        for name, value in chain(self.s.env_params.items(), self.s.training_hyperparams.items()):
            param_str += f"{name}:\t{value}\n"
        param_file = open(os.path.join(log_dir, self.PARAMS_FILE_NAME),"w") 
        param_file.write(param_str)
        param_file.close()
        
        #create files for logging
        running_reward_file = os.path.join(log_dir, self.RUNNING_REWARD_FILE_NAME)
        running_loss_file = os.path.join(log_dir, self.RUNNING_LOSS_FILE_NAME)
        running_multi_obj_file = os.path.join(log_dir, self.RUNNING_MULTI_OBJ_FILE_NAME)
        actions_file = os.path.join(log_dir, self.ACTIONS_FILE_NAME)
        
        #create folder for saving the state
        state_dir = os.path.join(log_dir, self.STATE_FOLDER_NAME)
        Path(state_dir).mkdir(parents=True, exist_ok=True)

        #initialize the logging for errors
        logging.basicConfig(filename= os.path.join(log_dir, self.ERROR_LOG_FILE_NAME), 
            level=logging.ERROR, format="%(asctime)s:\t%(message)s")

        #set default value for logging multiobjective if it's not passed in (for compatibility)
        if not "log_running_multi_obj" in self.s.log_info:
            self.s.log_info["log_running_multi_obj"] = True
        return extra.LogSession(log_dir, state_dir, self.s.log_info["log_running_reward"], self.s.log_info["log_running_loss"],
                            self.s.log_info["log_running_multi_obj"], self.s.log_info["log_actions"], running_reward_file,
                            running_loss_file, running_multi_obj_file, actions_file)                
  
    def initialize_nns(self):
        """ Initializes the NNs for the soft actor critic method """
        #create the main NNs
        self.ac = core_epi.MLPActorCritic(self.env.observation_space, self.env.action_space,
                                     hidden_sizes=self.s.training_hyperparams["HIDDEN_SIZES"],
                                     min_cov_eigen=self.s.training_hyperparams["MIN_COV_EIGEN"]).to(self.s.device)     
        
        #create the target NNs
        self.ac_targ = deepcopy(self.ac)
        
        #Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        #List of parameters for both Q-networks (saved for convenience)
        self.q_params = chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Count and print number of variables 
        var_counts = tuple(core_epi.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    def create_optimizer(self):
        """ Setup the ADAM optimizer for pi and q, and the SGD optimized for the alpha parameter"""
        #for backward compatibility
        if not "ALPHA_LR" in self.s.training_hyperparams:
            self.s.training_hyperparams["ALPHA_LR"] = self.s.training_hyperparams["LR"]
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.s.training_hyperparams["LR"])
        self.q_optimizer = optim.Adam(self.q_params, lr=self.s.training_hyperparams["LR"]) 
        self.alpha_optimizer = optim.SGD(self.ac.alpha.parameters(), lr=self.s.training_hyperparams["ALPHA_LR"])

    def current_h(self):
        """ returns the current value of H, the entropy of the policy, which decreases exponentially """
        return self.s.training_hyperparams["H_END"] + \
            (self.s.training_hyperparams["H_START"] - self.s.training_hyperparams["H_END"]) * \
            np.exp(-1. * self.s.steps_done / self.s.training_hyperparams["H_DECAY"])

    def current_alpha(self):
        """ return the current value of alpha, not as a trainable quantity """
        return self.ac.alpha_no_grad()

    def create_weight_scheduler(self):
        """ create the schedulers for the weights between the different multiobjectives """
        obj_num = self.get_obj_num()
        self.weight_scheduler = core_epi.WeightScheduler(self.s.training_hyperparams, obj_num).to(self.s.device)

    def get_obj_num(self):
        """ get number of objectives from the environment """
        info_dict = self.env.step(self.env.action_space.sample())[3]
        if "multi_obj" in info_dict:
            obj_num = len(info_dict["multi_obj"])
        else:
            obj_num = 1
        self.env.reset()
        return obj_num

    def get_action(self, o, deterministic=False):
        """ Returns an on-policy action based on the state passed in.
        This computation does not compute the gradients.

        Args:
            o (torch.Tensor): state from which to compute action
            deterministic (bool): wether the action should be sampled or deterministic  
        """
        return self.ac.act(o, deterministic)

    def compute_loss_q(self, data):
        """
        Compute the loss function of the q-value functions given a batch of data. 

        Args:
            data(dict): dictionary with the following
                "obs" (torch.Tensor): batch of states
                "act" (torch.Tensor): batch of continuous actions
                "objs" (torch.Tensor): batch of multi-objectives
                "obs2" (torch.Tensor): batch of next states
                "done" (torch.Tensor): batch of "done" flags

        Returns:
            (torch.Tensor): the sum of the loss function for both q-values
        """
        #unpack the batched data
        o, a, objs, o2, d = data['obs'], data['act'], data['objs'], data['obs2'], data['done'] 

        #value Q(s,u)
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions. The gradient is not taken respect to the target
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            #prepare the reward, as a weighted sum of the objectives, to pass to bellmann
            r = torch.tensordot(self.weight_scheduler(self.s.steps_done), objs, dims=([0],[1]))  

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.s.training_hyperparams["GAMMA"]*(1-d)* (q_pi_targ - self.current_alpha() * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, data):
        """
        Compute the loss function for the policy given a batch of data.

        Args:
            data(dict): dictionary with the following
                "obs" (torch.Tensor): batch of states

        Returns:
            loss(torch.Tensor): the loss function for the policy
            -log_pi(torch.Tensor): the negative average log probability 
        """
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.current_alpha() * logp_pi - q_pi).mean()

        return loss_pi, - logp_pi.mean()

    def compute_loss_alpha(self, data):
        """
        Compute the loss function for the alpha parameter given a batch of data. 

        Args:
            data(dict): dictionary with the following
                "obs"(torch.Tensor): batch of states

        Returns:
            (torch.Tensor): the loss function for the policy
        """

        o = data['obs']

        with torch.no_grad():
            _, logp_pi = self.ac.pi(o)

        # alpha loss function
        loss_alpha = -self.ac.alpha()*( logp_pi + self.current_h() ).mean()
        return loss_alpha

    def update(self, data):
        """
        Performs an update of the parameters of both Q, Pi, and the alpha parameter.

        Args:
            data (dict): batch of experience drawn from replay buffer. See compute_loss_q, compute_loss_pi
                and compute_loss_alpha for details
        
        Return:
            loss_q(float): the numerical value of the loss function for the value function
            loss_pi(float): the numerical value of the loss function for the policy function
            entropy(float): the average entropy of the policy
               
        """

        #optimize the value function
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data) 
        loss_q.backward()
        self.q_optimizer.step()

        #optimize the policy function
        # Freeze Q-networks since they will not be updated
        for p in self.q_params:
            p.requires_grad = False

        # optimze the policy params
        self.pi_optimizer.zero_grad()
        loss_pi, entropy = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so they can be optimized at the next step
        for p in self.q_params:
            p.requires_grad = True

        #optimize the temperature alpha
        self.alpha_optimizer.zero_grad()
        loss_alpha = self.compute_loss_alpha(data)
        loss_alpha.backward()
        self.alpha_optimizer.step()

        # Update target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # in-place operations
                p_targ.data.mul_(self.s.training_hyperparams["POLYAK"])
                p_targ.data.add_((1 - self.s.training_hyperparams["POLYAK"]) * p.data)

        return loss_q.item(), loss_pi.item(), entropy

    def update_log_files(self):
        """ updates all the log files with the current running reward (which is actually the running return), current 
            running losses (which actually contains more plotted info), and actions"""
        #update running reward
        if self.s.log_session.log_running_reward:
            self.append_log_line(f"{self.s.running_reward}", self.s.log_session.running_reward_file, self.s.steps_done)
        #update running loss
        if self.s.log_session.log_running_loss:
            self.append_log_line(np.array_str(self.s.running_loss,999999).replace("[", "").replace("]","")
                                ,self.s.log_session.running_loss_file, self.s.steps_done)
        #update running multi objective (if present)
        if self.s.log_session.log_running_multi_obj and self.s.running_multi_obj is not None:
            self.append_log_line(np.array_str(self.s.running_multi_obj,999999).replace("[", "").replace("]","")
                                ,self.s.log_session.running_multi_obj_file, self.s.steps_done)
        #update the actions
        if self.s.log_session.log_actions: 
            f=open(self.s.log_session.actions_file,'ab')
            np.savetxt(f, self.s.actions)
            f.close()
            self.s.actions = []

    def append_log_line(self, data, file, count):
        """appends count and data to file as plain text """
        file_object = open(file, 'a')
        file_object.write(f"{count}\t{data}\n")
        file_object.close()
  
    def plot_logs(self):
        """ plots important logs of the current train"""
        plotting.plot_sac_logs(self.s.log_session.log_dir, running_reward_file=self.s.log_session.running_reward_file,
            running_loss_file=self.s.log_session.running_loss_file, actions_file=self.s.log_session.actions_file,
                plot_to_file_line = None, suppress_show=False, rescale_action=False)

    def return_env_class_from_name(self):
        """
        Return the class to create a new environment, given the string
        of the environment class name in self.s.env_params['env_name'].
        Looks in sac_epi_envs for the environment class.

        Raises:
            NameError: if env_name doesn't exist

        Returns:
            Returns the class to create the environment
        """
        if hasattr(sac_epi_envs, self.s.env_params['env_name']):
            return getattr(sac_epi_envs, self.s.env_params['env_name'])
        else:
            raise NameError(f"Environment named {self.s.env_params['env_name']} not found in sac_envs")






