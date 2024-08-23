import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

"""
This module defines the NNs used to parameterize the value and policy function. Multiple
continuous actions are supported.
"""

#Constants used by the NNs to prevent numerical instabilities
UNIFORM_SIGMA =3.46

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    return a sequential net of fully connected layers

    Args:
        sizes(tuple(int)): sizes of all the layers
        activation: activation function for all layers except for the output layer
        output_activation: activation to use for output layer only

    Returns:
        stacked fully connected layers
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    """counts the variables of a module """
    return sum([np.prod(p.shape) for p in module.parameters()])

class RescaleInput(nn.Module):
    """
    This module rescales the input such that, if the input data was uniformly distributed
    between lower_bounds and upper_bouds, it would have variance 1. 
        
    Args:
        lower_bounds(np.array(float)): 1D array, with the same size of input, with all the
            lower bounds
        upper_bounds(np.array(float)): 1D array, with the same size of input, with all the
            upper bounds
    
    """
    def __init__(self, lower_bounds, upper_bounds):
        super().__init__()
        self.register_buffer("lower_bounds", torch.tensor(lower_bounds, dtype=torch.float32))
        self.register_buffer("upper_bounds", torch.tensor(upper_bounds, dtype=torch.float32))
    
    def forward(self, x):
        return ((x-self.lower_bounds)/(self.upper_bounds-self.lower_bounds)-0.5)*UNIFORM_SIGMA
        
class MLPActorCritic(nn.Module):
    """
    Module that contains two value functions, self.q1 and self.q2, the policy self.pi,
    and a module for the "temperature" parameter self.alpha.
    Both the value function and the policy are parameterizes using an arbitrary number of fully connected layers.
  
    Args:
        observation_space(gym.spaces): observation space of the environment (must be 1D array)
        action_space (gym.spaces): action space of sac_envs environments (must be 1D array)
        hidden_sizes (tuple(int)): size of each of the hidden layers that will be addded to 
            both the value and policy functions
        activation: activation function for all layers except for the output layer
        min_cov_eigen (float): used for stability when computing the covariance matrix of the multivariate
            normal distribution.

    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, min_cov_eigen = 1.e-8):
        super().__init__()

        # build policy, value functions, and module for the value of alpha
        self.pi = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation,
                                            min_cov_eigen=min_cov_eigen)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.alpha = Alpha()

    def act(self, obs, deterministic=False):
        """ return the action, chosen according to deterministic, given a single unbatched observation obs """
        with torch.no_grad():
            a, _ = self.pi(obs.view(1,-1), deterministic, False)
            return a[0]

    def alpha_no_grad(self):
        """
        returns the current value of the temperature parameter alpha without gradients
        """
        with torch.no_grad():
            return self.alpha()

class MLPQFunction(nn.Module):
    """
    Class representing a q-value function, implemented with fully connected layers
    that stack the state-action as input, and output the value of such state-action

    Args:
        observation_space(gym.spaces): observation space of the RL environment. Must be 1D 
            array        
        action_space(gym.spaces): action space of the RL environment. Must be 1D 
            array
        hidden_sizes(tuple(int)): list of sizes of hidden layers
        activation: activation function for all layers except for output
    """
    def __init__(self, observation_space, action_space, hidden_sizes, activation):
        super().__init__()

        #determine action and observation spaces
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        obs_act_lower_bounds = np.concatenate([observation_space.low, action_space.low ])
        obs_act_upper_bounds = np.concatenate([observation_space.high, action_space.high ])

        #define the input layer that places data in a bound
        self.rescale_input = RescaleInput(obs_act_lower_bounds, obs_act_upper_bounds)

        #define the main fully connected NN
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        """
        Args:
            obs(torch.Tensor): batch of observations
            act(torch.Tensor): batch of continuous actions

        Returns:
            (torch.Tensor): 1D tensor with value of each state-action in the batch
        """
        out = self.rescale_input(torch.cat([obs, act], dim=-1))
        out = self.q(out)
        return torch.squeeze(out, -1) 

class SquashedGaussianMLPActor(nn.Module):
    """
    Class representing the policy, implemented with fully connected layers
    that take the state as input, and outputs the averages and the covariance matrix
    of a multivariate normal distribution. The action is then computed by sampling 
    from this distribution, and passing the output thourgh a "squashing function" 
    (a hyperbolic tangent, and then a rescaling to the desired range).

    Args:
        observation_space(gym.spaces): observation space of the RL environment. Must be 1D 
            array        
        action_space(gym.spaces): action space of the RL environment. Must be 1D 
            array
        hidden_sizes(tuple(int)): list of sizes of hidden layers
        activation: activation function for all layers except for output
        min_cov_eigen (float): used for stability when computing the covariance matrix of the multivariate
            normal distribution.
    """
    def __init__(self, observation_space, action_space, hidden_sizes, activation,
                min_cov_eigen=1.e-8):
        super().__init__()

        #load state and action dimensions and bounds
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        obs_lower_bounds = observation_space.low
        obs_upper_bounds = observation_space.high
        self.register_buffer("act_lower_bounds", torch.tensor(action_space.low, dtype=torch.float32))
        self.register_buffer("act_upper_bounds", torch.tensor(action_space.high, dtype=torch.float32))
        self.act_dim = act_dim

        #define the input layer that places data in a bound
        self.rescale_input = RescaleInput(obs_lower_bounds, obs_upper_bounds)

        #main network taking the state as input, and passing it through all hidden layers
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        
        #output layer producing the average of the multivariate gaussina
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        
        #output layer producing a matrix M such that the covariance matrix is
        #M * M^T + min_cov_eigen * Id, where min_cov_eigen is a small constant to ensure strict positivity
        self.m_layer = nn.Linear(hidden_sizes[-1], act_dim*act_dim)
        
        #I create the matrix l_id = min_cov_eigen * Id for the covariance matrix
        self.register_buffer("l_id", torch.eye(act_dim)*min_cov_eigen)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Args:
            obs(torch.Tensor): batch of observations
            deterministic(bool): if the actions should be chosen deterministally or not
            with_logprob(bool): if the log of the probability should be computed and returned

        Returns:
            pi_action(torch.Tensor): the chosen continuous actions
            logp_pi(torch.Tensor): the log probability of such continuous actions
        """
        #run the state through the network
        net_out = self.rescale_input(obs)
        net_out = self.net(net_out)
        mu = self.mu_layer(net_out)
        m = self.m_layer(net_out).view(-1,self.act_dim,self.act_dim)
        cov_mat = (torch.transpose(m,1,2) @ m) + self.l_id

        # Pre-squash (tanh) distribution and sample
        pi_distribution = MultivariateNormal(mu, cov_mat)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        #if necessary, compute the log of the probabilities
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action)

            #change of distribution when going from gaussian to Tanh
            logp_pi -= (2.*(np.log(2.) - pi_action - F.softplus(-2.*pi_action))).sum(axis=1)
            
            #notice that we don't account for the rescaling of the action in logp_pi. This is to
            #make the entropy more universal, and not dependent on the specific interval size.
        else:
            logp_pi = None
       
        #apply tanh to the sampled gaussian
        pi_action = torch.tanh(pi_action)

        #apply the shift of the action to the correct interval
        pi_action = self.act_lower_bounds + 0.5*(pi_action + 1.)*(self.act_upper_bounds-self.act_lower_bounds)

        return pi_action, logp_pi

class Alpha(nn.Module):
    """
    This module is used to return the current value of the "temperature parameter" alpha, and to update
    it using gradient descent in order to ensure a given target entropy of the distribution. Since alpha>0,
    we store log_alpha, which can be any R number, and then exponentiate it to get alpha = e*log_alpha.
    This ensures that any update cannot make alpha negative.
       
    """
    def __init__(self):
        super().__init__()

        #initialize it to 1., an arbitrary initial value
        self.log_alpha_val = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        
    def forward(self):
        """
        return the value of alpha
        """
        return torch.exp(self.log_alpha_val)

class WeightScheduler(nn.Module):
    """
    Module that returns a 1D array with weights to assign to each objective. It is used to schedule the weight
    between objectives during training. The scheduling interpolates by a set of initial weights 
    training_hyperparams["C_START"] to some final weights training_hyperparams["C_END"] using a Fermi distribution
    with mean training_hyperparams["C_MEAN"] and typical width training_hyperparams["C_WIDTH"]

    Args:
        training_hyperparams(dict): dictionary containing
            "C_START"(np.array): 1D array, with same length as objectives, representing the initial weight for each objective
            "C_END"(np.array): 1D array, with same length as objectives, representing the final weight for each objective
            "C_MEAN"(float): mean point of the fermi distribution in units of steps
            "C_WIDTH"(float): typical width of the Fermi distribution in units of steps
            If these are missing, it will simply return a (1,0...,0) vector with no scheduling
        obj_num(int): number of objectives. If the previous hyperparameters are not passed, this must be specified, otherwise
            in gives an error
    """

    def __init__(self, training_hyperparams, obj_num=None):
        super().__init__()
        
        #create the main tensors that will produce the weights
        if "C_START" in training_hyperparams:
            self.register_buffer("c_start", torch.tensor(training_hyperparams["C_START"], dtype=torch.float32))
            self.register_buffer("c_end", torch.tensor(training_hyperparams["C_END"], dtype=torch.float32))
            self.register_buffer("c_mean", torch.tensor(training_hyperparams["C_MEAN"], dtype=torch.float32))
            self.register_buffer("c_width", torch.tensor(training_hyperparams["C_WIDTH"], dtype=torch.float32))
        else:
            assert obj_num is not None
            self.register_buffer("c_start", torch.zeros((obj_num,), dtype=torch.float32))
            self.register_buffer("c_end", torch.zeros((obj_num,), dtype=torch.float32))
            self.register_buffer("c_mean", torch.tensor(1., dtype=torch.float32))
            self.register_buffer("c_width", torch.tensor(1., dtype=torch.float32))
            self.c_start[0] = 1.
            self.c_end[0] = 1.

        #save their cpu version
        self.c_start_cpu = self.c_start.cpu().numpy()
        self.c_end_cpu = self.c_end.cpu().numpy()
        self.c_mean_cpu = self.c_mean.cpu().numpy()
        self.c_width_cpu = self.c_width.cpu().numpy()
        
    def forward(self, step, return_numpy_cpu = False):
        """
        Return the weights for the multi-objectives. 
        Args:
            step(int): current training step number
            return_numpy_cpu(bool): whether the output should by a np.array or torch.tensor 
        """
        if return_numpy_cpu:
            return self.return_schedule(step, self.c_start_cpu, self.c_end_cpu, self.c_mean_cpu, self.c_width_cpu, np)
        else:
            return self.return_schedule(step, self.c_start, self.c_end, self.c_mean, self.c_width, torch)

    def return_schedule(self, step, start, end, mean, width, module):
        """return the Fermi distribution computed in the current step"""
        return end + (start - end)/(1. + module.exp( (step-mean)/width ))

            
        
        
        

