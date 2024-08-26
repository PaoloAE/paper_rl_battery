# paper_rl_battery
Code used to produce the results presented in the manuscript:

P. A. Erdman, G. M. Andolina, V. Giovannetti, and F. Noé, *Reinforcement learning optimization of the charging of a Dicke quantum battery*, [arXiv:2212.12397](https://doi.org/10.48550/arXiv.2212.12397). 

## Getting started
To get started, open the [```jupyter```](jupyter) folder which contains the following Jupyter Notebooks:
* [```0_train_coupling_case.ipynb```](jupyter/0_train_coupling_case.ipynb): this Notebook allows to optimize the Dicke mattery in the coupling scheme, and contains details on the physical problem;
* [```0_train_detuning_case.ipynb```](jupyter/0_train_detuning_case.ipynb): this Notebook allows to optimize the Dicke mattery in the detuning scheme, and contains details on the physical problem;
* [```1_evaluate_and_export_performance.ipynb```](jupyter/1_evaluate_and_export_performance.ipynb): this Notebook loads the trained RL agent data previously produced with either  ```0_train_coupling_case.ipynb``` or ```0_train_detuning_case.ipynb```, evaluates their performance, and exports all relevant physical quantities and plots (see the Jupyter Notebook for details on the exported quantity). 
* [```2_plot_results.ipynb```](jupyter/2_plot_results.ipynb): this Notebook uses the data exported from ```1_evaluate_and_export_performance.ipynb``` to plot the results of the optimization.

## Acknowledgement
Implementation of the soft actor-critic method based on extensive modifications and generalizations of the code provided at:

J. Achiam, Spinning Up in Deep Reinforcement Learning, https://github.com/openai/spinningup (2018).
