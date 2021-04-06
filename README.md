# Unlocking heterogeneinity of node activation in Boolean networks through dynamical programming
In the letter we evaluate the activation probability for  states associated to  nodes of a network. This repository provides the code to reproduce the results of the letter. 
Important  files:
- power.py: python script 
- analyse_results.ipynb

*power.py*  is a Python script that runs the cavity method on a network with power law degree distribution and stores the output files inside /data

*analyse_results.ipynb* is a  Jupyter notebook that loads the output files from the script and create the image
Note that *power.py* depends on the  module /lib/dynamical_cavity.py which contains the code to run the recursive calls of the cavity equations. Figure in the letter explores different values of the noise level T. To reproduce the results,  the evaluation associated to each value of T is run in parallel. Open a terminal and type `python power.py -h' to know more.
Default parameter of ‘‘power.py'' are chosen to complete the evaluation in a reasonable time. However  If you want to reproduce the figure run ‘‘‘python -N 200000 --Ts 0.01 1.1 0.01'''
