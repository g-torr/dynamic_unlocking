# Overcoming the complexity barrier of the dynamic message-passing method in networks with fat-tailed degree distributions 
## :key: Key points 
 Dynamics of agents on network is characterised by the interplay between (thermal) noise  and interaction, the balance between the two has deep effects on the states of the agents. We study the linear threshold model on a random network with direct coupling:  <img src="https://render.githubusercontent.com/render/math?math=n_{i}(t)=  \Theta \big[\sum_j J_{ij}n_j(t-1)\big) -\vartheta_i -z_i(t)\big]">, where  <img src="https://render.githubusercontent.com/render/math?math=z_i(t)"> and  <img src="https://render.githubusercontent.com/render/math?math=J_{ij}"> are the noise and interaction term respectively. 
 
We develop an analytical method that performes the average over the noise, so you get the activation probability of each node of the network. However, performing this  average requires sampling from a huge space. This contribution provides an efficient way to solve this problem using dynamic programming.   Check out our [article](https://doi.org/10.1103/PhysRevE.104.045313)  or the [preprint](https://arxiv.org/abs/2105.04197)  for more.
 This repository provides the code to reproduce the results of the manuscrip. 

## :computer: How to run 
You need python and jupyter notebook installed in your system. 
Important  files are:
- power.py: python script 
- analyse_results.ipynb

*power.py*  is a Python script that runs the cavity method on a network with power law degree distribution and stores the output files inside a folder  /data. Run it first from a terminal by typing `python power.py`

*analyse_results.ipynb* is a  Jupyter notebook that loads the output files from the script and create the heatmap of Fig. 2

The figure 2 in the letter explores different values of the noise level T. To reproduce the results,  the evaluation associated to each value of T is run in parallel. You can control how many processes to be run in parallel, run `python power.py -h`  in a terminal to know more.

N.B.
Default parameter of *power.py* are chosen to complete the evaluation in a reasonable time. However,  If you want to reproduce the figure of the manuscript,  run 
`python power.py -N 200000 --Ts 0.01 1.1 0.01`

[![DOI](https://zenodo.org/badge/355104547.svg)](https://zenodo.org/badge/latestdoi/355104547)
