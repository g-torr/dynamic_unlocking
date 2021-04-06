'''Compute the stationary node activation probability for a graph with fat tailed distribution using dynamical cavity'''
import numpy as np
import pickle
import networkx as nx
import itertools
import sys
import argparse
from argparse import RawTextHelpFormatter
import os
import time
sys.path.insert(0, "./lib")  # add the library folder to the path I look for modules
import dynamical_cavity as cavity # this script comutes the dynamical cavity


def save_obj(obj,theta):
    name='theta:'+str(theta)+'.pkl'
    if not os.path.exists("./data"):
          os.makedirs("./data")
    if os.path.isfile('./data/dic-' + name ):
        name = name[:-4]+'_'+ str(time.time())+'.pkl'
    with open('./data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def generate_degree_seq(gamma, N):
    kseq = np.ceil( np.random.pareto(gamma, N))
    cond = kseq > N
    while any(cond):
        temp_seq = np.ceil( np.random.pareto(gamma, np.count_nonzero(cond)))
        kseq[cond] = temp_seq
        cond = kseq > N
    return np.array(kseq, dtype=int)
def make_network(N, gamma,bias):
    def neighbouring(A):
        interaction = []
        js = []
        Ks = []
        for l, u in zip(A.indptr[:-1], A.indptr[1:]):
            js += [A.indices[l:u]]
            interaction += [A.data[l:u] / np.sqrt(u - l)]
            Ks += [u - l]
        return js, interaction, Ks

    seq = generate_degree_seq(gamma, N)
    G = nx.generators.degree_seq.directed_configuration_model(seq, np.random.permutation(seq))
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    J = nx.adjacency_matrix(G)
    sign_interaction = np.where(np.random.rand(J.nnz) > bias, 1, -1)  # bias in positive regulation
    J.data = np.ravel(sign_interaction)  # add negative element
    return J


def main():
    parser = argparse.ArgumentParser(
        description='Compute the stationary  node activation probability for a graph with fat tailed distribution using dynamical programming.\n'
                    '\n'
                    'Returns:\n'
                    'a dictionary containing the topology "J", and the activation probability "data". \n'
                    '"J" is a scipy.sparse matrix.\n'
                    '"data" is a 2d list containing single node activation probabilities at different noise parameters T.\n'
                    'Output is saved in /data/ folder with unique identifier. Simulation for different values of T are run in parallel. By default, code runs on  all cores available on your machine.',formatter_class=RawTextHelpFormatter)
    parser.add_argument("-N", help="Number of nodes", type=int, const=20000, default=20000, nargs='?')
    parser.add_argument('--theta', type=float, default=0., help="theta. Default set to 0")
    parser.add_argument('--nprocess', type=int, const=-1,default=-1,nargs='?', help="number of processes run in parallel, i.e. number of cores to be used in your local machine. Default all cores available")
    parser.add_argument('--Ts', type = float, nargs = '*', default = [0.05, 1.1, 0.05],help = "[Tmin,Tmax,dT]. Simulation investigates noise parameter values: np.arange(Tmin,Tmax,dt). Default [0.05,1.1,0.05] ")
    args = parser.parse_args()
    N = args.N
    theta = args.theta
    threads = args.nprocess
    gamma = 1.81
    bias = 0.379
    Ts = np.arange((args.Ts)[0], (args.Ts)[1], (args.Ts)[2])

    J = make_network(N,gamma,bias)
    max_outdegree = max(np.diff(J.indptr))
    max_recursions = int((max_outdegree+1)*(max_outdegree+2)/2)
    if max_recursions> sys.getrecursionlimit():
        print("Warning! maximum degree larger than default recursion limit, I 'll update recursion limit to", max_recursions )
        sys.setrecursionlimit(max_recursions)
    print('Network done')

    data =cavity.cavity_parallel([0.5]*N,Ts,J,theta,threads)# run in parallel at different temperatures
    dic = {'data': data, 'J': J, 'Ts': Ts, 'theta': theta}
    save_obj(dic,theta)


if __name__ == '__main__':
    main()
