from __future__ import print_function
import gym
import numpy as np
import dataclasses
import qutip as qt

"""
This module contains gym.Env environments that can be trained using sac_epi.SacTrain. 
Additionally to training based on rewards,  sac_epi.SacTrain can also train on multi objectives.
If "multi_obj" is returned in the info_dict, then the reward will be computed as a weighted average
of the multi objectives. The weights change during training (see sac_epi.SacTrain). For instance, we
use this to shift, during training, the optimization goal from the energy to the ergotropy. This stabilizes
training, since learning to optimize the energy is easier.

These environments, besides being proper gym.Env, must satisfy these additional requirements:
    1) __init__ must accept a single dict with all the parameters necessary to define the environment.
    2) implement set_current_state(state). Functions that takes a state as input, and sets the environment to that state
    3) if the multi-objective training is required, then the info_dict must contain a key "multi_obj" whose
        value if a 1D np.array with the value of each objective
    
They must satisfy these additional properties for the functions that export physical quantities to work
    1) implement env.state.psi as a qutip quantum object
    2) provide env.nc (cutoff value of Fock space of the cavity), env.nq (number of qubits/TLS), env.wc (frequency
        of the cavity)
    3) provide env.hc and env.hq as two qutip objects representing the cavity and the qubits Hamiltonian respectively
"""

#shared functions to compute ergotropy 

def rho_b1(rho_b):
    """
    computes the reduced density matrix of a single TLS. 

    Args:
        rho_b(qutip.Qobj): denisty matrix of all qubits together

    Returns:
        rho_b1(np.array): 2x2 matrix representing the single qubit state
    """

    N=np.array(rho_b.dims)[0,0]-1
    rho=np.zeros((2,2),dtype=complex)

    for e in range(N):
        rho[1,1]+=(N-e)/N *rho_b[N-e,N-e]
        rho[1,0]+=np.sqrt((e+1)*(N-e))/N *rho_b[N-e-1,N-e]
        rho[0,0]+=(e+1)/N*rho_b[N-e-1,N-e-1]

    rho[0,1]=np.conjugate(rho[1,0])

    return rho        

def ergotropy_1tls(rho_b, omega0):  
    """
    Computes the ergotropy of a single qubit, given the density matrix of all qubits
    
    Args:
        rho_b(qutip.Qobj): denisty matrix of all qubits together
        omega0(float): energy gap of the battery qubits (hbar=1)

    Returns:
        erg(float): value of the ergotropy
    """

    #compute single qubit density matrix
    rho = rho_b1(rho_b) 
    #get its energy
    en = omega0*rho[0,0] 
    #compute the smallest eigenvalue of rho
    p1 = np.linalg.eigvalsh(rho)[0]
    #compute the energy of the passive state
    en_passive = omega0*p1 
    #compute ergotropy
    erg=en-en_passive
    return  np.real(erg)

def return_physical_quantities(env):
    """
    Compute various quantities characterizing the state of the system (see Returns)

    Args:
        env(gym.Env): environment representing the battery. It is used to get the state
            of the quantum system, from which various quantities are computed

    Returns:
        see last line, the return
    """
    #get the global state, and the partial traces
    tot_state = env.state.psi
    qubits_state = tot_state.ptrace(1)
    qubit_state = qt.Qobj(rho_b1(qubits_state) , isherm=True)
    cavity_state = tot_state.ptrace(0)
    
    #compute useful operators
    a = qt.destroy(env.nc + 1)
    a_both = qt.tensor(a, qt.qeye(env.nq + 1))
    jz_both= qt.tensor(qt.qeye(env.nc + 1), qt.jmat(env.nq/2,'z'))      
    h0_qubit = qt.jmat(1/2,'z') + 1/2
    h0_qubits = qt.jmat(env.nq/2,'z') + env.nq/2
    h0_cavity = env.wc* a.dag()*a
    h0_tot = (jz_both + env.nq/2) + env.wc* a_both.dag()*a_both

    #compute single qubit quantities
    qubit_energy = qt.expect(h0_qubit, qubit_state)
    qubit_entropy = qt.entropy_vn(qubit_state)
    qubit_variance = qt.expect(h0_qubit*h0_qubit, qubit_state) - qt.expect(h0_qubit, qubit_state)**2
    qubit_purity = (qubit_state*qubit_state).tr()

    #compute all qubits quantities
    qubits_energy = qt.expect(h0_qubits, qubits_state)
    qubits_entropy = qt.entropy_vn(qubits_state)
    qubits_variance = qt.expect(h0_qubits*h0_qubits, qubits_state) - qt.expect(h0_qubits, qubits_state)**2

    #compute all cavity quantities
    cavity_energy = qt.expect(h0_cavity, cavity_state)
    cavity_entropy = qt.entropy_vn(cavity_state)
    cavity_variance = qt.expect(h0_cavity*h0_cavity, cavity_state) - qt.expect(h0_cavity, cavity_state)**2

    #compute all total quantities
    tot_energy = qt.expect(h0_tot, tot_state)
    tot_variance = qt.expect(h0_tot*h0_tot, tot_state) - qt.expect(h0_tot, tot_state)**2

    return [qubit_energy, qubit_entropy, qubit_variance, qubit_purity,
            qubits_energy, qubits_entropy, qubits_variance,
            cavity_energy, cavity_entropy, cavity_variance,
            tot_energy, tot_variance]

#these variables are used in return_spectrum_weights to avoid diagonalizing multiple times if h doesnt change
h_buf = [None,None,None]
energies_buf = [None,None,None]
eig_states_buf = [None,None,None]

def return_spectrum_weights(env):
    """
    The output is an array with shape (3, eig_vals, 2).
    - First index is for h_tot, hc, hq.
    - Second index is one for each eigen value
    - Third index. 0: energies, 1: weights.
    
    """
    #get hamiltonian and state
    #h_tot = env.hamiltonian(env.state.u)
    hc = env.hc
    hq = env.hq
    htot = hc+hq
    psi = env.state.psi

    #put the 3 hamiltonians together in a list
    h = [htot, hc, hq]

    #the output
    output = []
    
    #loop over the 3 hamiltonians
    for i in range(len(h)):
        
        #if the hamiltonian hasnt been diagonalized, i do it
        if h[i] != h_buf[i]:
            #get the spectrum 
            energies_buf[i], eig_states_buf[i] = h[i].eigenstates()
            #update h_buf
            h_buf[i]= h[i]         
        
        #project and get the square modulus
        weights = np.array([np.abs(eig_state.overlap(psi))**2 for eig_state in eig_states_buf[i]])
        
        #concatenate energy and weights
        output.append(np.stack([energies_buf[i], weights], axis=1))
        
    #return it as a numpy array
    return np.array(output)

class DickeBatteryOneControlCoupling(gym.Env):
    """
    Gym.Env representing a Dicke battery where the coupling between the cavity and the
    qubits is the single continuous control.

    Args:
        env_params is a dictionary that must contain the following: 
            "wc" (float): frequency of the cavity
            "nc" (int): maximum number of photons in cavity (Fock space truncation)
            "nc_i" (int): initial number of photons in cavity   
            "wq" (float): frequency of the qubits / 2 level atoms
            "nq" (int): number of qubits  
            "min_u" (float): minimum value of the control (the light-matter 
                coupling strenght)
            "max_u" (float): maximumm value of the control (the light-matter 
                coupling strenght)
            "dt" (float): timestep length 
            "tau" (float): the charging time
            "reward_coeff" (float): the reward is multiplied by this    
            "quantity"(str): either "energy" or "ergotropy". Determines what is returned
                as the reward. Since this environment returns both quantities in multi_obj,
                this is not really important. sac_epi.SacTrain then determines the weight between
                the two.
    """

    @dataclasses.dataclass
    class State:
        """
        Data object representing the state of the environment. It consists of a qutip 
        vector psi, the last chosen action u, and the current time
        """
        psi = 0.
        u  = 0.
        t = 0.
        
    def __init__(self, env_params):
        super().__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.wc = env_params["wc"]
        self.nc = env_params["nc"]
        self.nc_i = env_params["nc_i"]
        self.wq = env_params["wq"]
        self.nq = env_params["nq"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.dt = env_params["dt"]
        self.tau = env_params["tau"]
        self.reward_coeff = env_params["reward_coeff"]
        if "quantity" in env_params:
            self.quantity = env_params["quantity"]
        else:
            #for backward compatibility
            self.quantity = "energy"
            
        self.state = self.State()

        #call reset to setup the initial state
        init_state = self.reset()

        #prepare the observation space (each coefficient is between -1 and +1)
        obs_low = np.zeros(len(init_state), dtype=np.float32) - 1. 
        obs_low[-2] = self.min_u
        obs_low[-1] = 0.
        obs_high = obs_low + 2.
        obs_high[-2] = self.max_u
        obs_high[-1] = self.tau 

        #set the observation and action spaces
        self.observation_space = gym.spaces.Box( low=obs_low, high= obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                              high=np.array([self.max_u],dtype=np.float32), dtype=np.float32)

        #initialize the Hamiltonian
        self.init_hamiltonian()
        
    def reset(self):
        """ resets the state of the environment """
        
        psi_list = []

        #initialize the cavity state
        psi_list.append(qt.basis(self.nc + 1, self.nc_i))
        
        #initialize the qubit state
        psi_list.append(qt.basis(self.nq + 1, self.nq))  
        
        #set the psi state of this environment
        self.state.psi = qt.tensor(psi_list)
        
        #set the initial value of the controls (by default I choose the lowest value)
        self.state.u = self.min_u

        #reset the time
        self.state.t = 0.

        return self.current_state()
    
    def step(self, action):
        """
        Evolves the state for a timestep depending on the chosen action and returs standard RL env quantities

        Args:
            action (type specificed by self.action_space): the action to perform on the environment.
                Here it's a 1D np.float32 np.array with a single quantity representing the cavity-qubits coupling

        Raises:
            Exception: action out of bound

        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, as specified by env_params["quantity"] (either the energy or ergotropy
                difference upon taking the action)
            terminal_state(bool): whether the episode ends (the charging time is up)
            info_dict(dict): dict with "multi_obj" that is an array with the multiobjectives (first element
                is the energy difference of all qubits, second one is the single-qubit ergotropy difference)
        """
   
        #check if action in bound
        if not self.action_space.contains(action):
            raise Exception(f"Action {action} out of bound")
        
        #load action 
        u = action[0]
        
        #save the energy and single-qubit ergotropy difference, measured with respect to h0, before the state evolution
        e_i = self.current_energy()
        erg_i = self.current_ergotropy()

        #evolve the wavefunction and update the state
        self.state.psi = qt.mesolve(self.hamiltonian(u), self.state.psi, [0,self.dt]).states[1]
        self.state.u = u
        self.state.t += self.dt

        #compute new energy and ergotropy
        e_f = self.current_energy()
        erg_f = self.current_ergotropy()

        #compute energy and ergotropy change
        de = self.reward_coeff * (e_f - e_i)
        derg = self.reward_coeff * (erg_f - erg_i)

        #choose the correct reward
        if self.quantity == "energy":
            reward = de
        elif self.quantity == "ergotropy":
            reward = derg
        else:
            print(f"Quantity {self.quantity} is not a valid choice. It must be 'energy' or 'ergotropy'")

        #check if state is terminal
        terminal_state = (self.state.t >= self.tau)

        return self.current_state(), reward, terminal_state, {"multi_obj": 
            np.array([de, derg], dtype=np.float32)}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.current_state())
    
    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_epi.SacTrain.load_full_state() to properly load a saved training session.

        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        psi_len = state.shape[0] - 2
        psi_re_np = state[:psi_len//2]
        psi_im_np = state[psi_len//2:-2]
        self.state.u = state[-2]
        self.state.t = state[-1]
        self.state.psi = qt.Qobj(psi_re_np + 1.j*psi_im_np, dims=[[self.nc+1, self.nq+1],[1,1]])

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        psi_np = self.state.psi.full(squeeze=True)
        psi_re_np = np.real(psi_np)
        psi_im_np = np.imag(psi_np)
        u_t_vec = np.array([self.state.u, self.state.t])
        return np.concatenate([psi_re_np, psi_im_np, u_t_vec])
           
    def init_hamiltonian(self):
        """ initialize qutip object to speed up the construction of the Hamiltonian during step """

        a = qt.tensor(qt.destroy(self.nc + 1), qt.qeye(self.nq + 1))
        jz_b= qt.tensor(qt.qeye(self.nc + 1), qt.jmat(self.nq/2,'z')) 
        
        #set up the hamiltonian of the qubits (battery) 
        self.hq = self.wq*(jz_b + self.nq/2)

        #set up the battery of the cavity
        self.hc = self.wc* a.dag()*a

        # construct the hamiltonian of the cavity 
        self.h0 = self.hc + self.hq

        #construct the interaction term (with interaction coefficient = 1)
        self.h1 = 2.*qt.tensor(qt.destroy(self.nc+1).dag() + qt.destroy(self.nc+1), qt.jmat(self.nq/2,'x'))

    def hamiltonian(self, u):
        """
        returns the Hamiltonian given the control u

        Args:
            u(float): value of the control

        Returns
            h(qutip.Qobj): total qubits and cavity Hamiltonian given the control u
        """
        return self.h0 + u*self.h1

    def current_energy(self):
        """ energy of the current state, measured with respect to the bare qubit Hamiltonian """
        return qt.expect(self.hq, self.state.psi)
    
    def current_ergotropy(self):
        """ ergotropy of the current state, measured respect to the bare qubit Hamiltonian """
        #the omega in the second argument is the batteries proper omerga
        return ergotropy_1tls(self.state.psi.ptrace(1), self.wq)

class DickeBatteryOneControlDetuning(gym.Env):
    """
    Gym.Env representing a Dicke battery    

    Args:
        env_params is a dictionary that must contain the following: 
            "wc" (float): frequency of the cavity
            "nc" (int): maximum number of photons in cavity
            "nc_i" (int): initial number of photons in cavity   
            "wq" (float): frequency of the qubits / 2 level atoms
            "nq" (int): number of qubits  
            "g" (float): the coupling strength \tilde{g} between the qubits and the cavity
            "min_u" (float): minimum value of the control (the detuning of the qubits)
            "max_u" (float): maximumm value of the control (the detuning of the qubits)
            "dt" (float): timestep \Delta t
            "tau" (float): the charging time
            "reward_coeff" (float): the reward is multiplied by this  
            "quantity"(str): either "energy" or "ergotropy". Determines what is returned
                as the reward. Since this environment returns both quantities in multi_obj,
                this is not really important. sac_epi.SacTrain then determines the weight between
                the two.  
    """
    
    @dataclasses.dataclass
    class State:
        """
        Data object representing the state of the environment. It consists of a qutip 
        vector psi and the last chosen action u
        """
        psi = 0.
        u  = 0.
        t = 0. 

    def __init__(self, env_params):
        super().__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.wc = env_params["wc"]
        self.nc = env_params["nc"]
        self.nc_i = env_params["nc_i"]
        self.wq = env_params["wq"]
        self.nq = env_params["nq"]
        self.g = env_params["g"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.dt = env_params["dt"]
        self.tau = env_params["tau"]
        self.reward_coeff = env_params["reward_coeff"]
        if "quantity" in env_params:
            self.quantity = env_params["quantity"]
        else:
            #for backward compatibility
            self.quantity = "energy"
            
        self.state = self.State()

        #call reset to setup the initial state
        init_state = self.reset()

        #prepare the observation space (each coefficient is between -1 and +1)
        obs_low = np.zeros(len(init_state), dtype=np.float32) - 1. 
        obs_low[-2] = self.min_u
        obs_low[-1] = 0.
        obs_high = obs_low + 2.
        obs_high[-2] = self.max_u
        obs_high[-1] = self.tau 

        #set the observation and action spaces
        self.observation_space = gym.spaces.Box( low=obs_low, high= obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                              high=np.array([self.max_u],dtype=np.float32), dtype=np.float32)

        #initialize the Hamiltonian
        self.init_hamiltonian()
        
    def reset(self):
        """ resets the state of the environment """
        
        psi_list = []

        #initialize the cavity state
        psi_list.append(qt.basis(self.nc + 1, self.nc_i))
        
        #initialize the qubit state
        psi_list.append(qt.basis(self.nq + 1, self.nq))  
        
        #set the psi state of this environment
        self.state.psi = qt.tensor(psi_list)
        
        #set the initial value of the controls (by default i choose the lowest value)
        self.state.u = self.min_u

        #reset the time
        self.state.t = 0.

        return self.current_state()
    
    def step(self, action):
        """
        Evolves the state for a timestep depending on the chosen action and returs standard RL env quantities

        Args:
            action (type specificed by self.action_space): the action to perform on the environment.
                Here it's a 1D np.float32 np.array with a single quantity representing the detuning of the qubits

        Raises:
            Exception: action out of bound

        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, as specified by env_params["quantity"] (either the energy or ergotropy
                difference upon taking the action)
            terminal_state(bool): whether the episode ends (the charging time is up)
            info_dict(dict): dict with "multi_obj" that is an array with the multiobjectives (first element
                is the energy difference of all qubits, second one is the single-qubit ergotropy difference)
        """
   
        #check if action in bound
        if not self.action_space.contains(action):
            raise Exception(f"Action {action} out of bound")
        
        #load action 
        u = action[0]
        
        #save the energy, measured respect to h0, before the state evolution
        e_i = self.current_energy()
        erg_i = self.current_ergotropy()

        #evolve the wavefunction and update the state
        self.state.psi = qt.mesolve(self.hamiltonian(u), self.state.psi, [0,self.dt]).states[1]
        self.state.u = u
        self.state.t += self.dt

        #compute new energy
        e_f = self.current_energy()
        erg_f = self.current_ergotropy()

        #compute energy and ergotropy change
        de = self.reward_coeff * (e_f - e_i)
        derg = self.reward_coeff * (erg_f - erg_i)

        #choose the correct reward
        if self.quantity == "energy":
            reward = de
        elif self.quantity == "ergotropy":
            reward = derg
        else:
            print(f"Quantity {self.quantity} is not a valid choice. It must be 'energy' or 'ergotropy'")

        #check if state is terminal
        terminal_state = (self.state.t >= self.tau)

        return self.current_state(), reward, terminal_state, {"multi_obj": 
            np.array([de, derg], dtype=np.float32)}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.current_state())
    
    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_tri.SacTrain.load_full_state() to properly load a saved training session.

        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        psi_len = state.shape[0] - 2
        psi_re_np = state[:psi_len//2]
        psi_im_np = state[psi_len//2:-2]
        self.state.u = state[-2]
        self.state.t = state[-1]
        self.state.psi = qt.Qobj(psi_re_np + 1.j*psi_im_np, dims=[[self.nc+1, self.nq+1],[1,1]])

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        psi_np = self.state.psi.full(squeeze=True)
        psi_re_np = np.real(psi_np)
        psi_im_np = np.imag(psi_np)
        u_t_vec = np.array([self.state.u, self.state.t])
        return np.concatenate([psi_re_np, psi_im_np, u_t_vec])
           
    def init_hamiltonian(self):
        """ initialize qutip object to speed up the construction of the hamiltonian """

        a = qt.tensor(qt.destroy(self.nc + 1), qt.qeye(self.nq + 1))
        jz_b= qt.tensor(qt.qeye(self.nc + 1), qt.jmat(self.nq/2,'z')) 
        
        #set up the hamiltonian of the qubits (battery) alone 
        self.hq_bare = (jz_b + self.nq/2)
        self.hq = self.wq*self.hq_bare

        #set up the hamiltonian of the cavity
        self.hc = self.wc* a.dag()*a

        # construct the hamiltonian of the cavity + the interaction
        self.hc_hint = self.hc + self.g*2.*qt.tensor(qt.destroy(self.nc+1).dag() + qt.destroy(self.nc+1), qt.jmat(self.nq/2,'x'))

    def hamiltonian(self, u):
        """
        returns the Hamiltonian given the control u

        Args:
            u (float): value of the control
        """
        return self.hc_hint + (self.wq + u)*self.hq_bare

    def current_energy(self):
        #energy of the current state, measured respect to bare qubit hamiltonian
        return qt.expect(self.hq, self.state.psi)
    
    def current_ergotropy(self):
        #the omega in the second argument is the batteries proper omerga
        return ergotropy_1tls(self.state.psi.ptrace(1), self.wq)
