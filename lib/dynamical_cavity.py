from functools import lru_cache
import numpy as np
import itertools
import warnings
import scipy
import numpy.random
import argparse
from functools import lru_cache
import sys
from numba import jit
import os
from multiprocessing import Pool
@jit(nopython=True)
def cavity_single_numba(P_loc,inter,T,theta,K,J0):
    '''
    Dynamic programming is run using iterative calls. It creates a matrix for this task
    '''
    pos = inter.copy()
    neg = inter.copy()
    pos[pos<0]=0#apply theta function
    neg[neg>0]=0
    m = np.zeros((K+1,K+1))

    h_tilde =np.arange(np.sum(neg),np.sum(pos)+1)
    for ind in range(len(h_tilde)):
        m[K,ind]=1/2*(1+np.tanh((h_tilde[ind]*J0-theta)/2/T))
    offset = np.sum(neg)#this is the offset to map \tilde{h} to index of m matrix
    for l,low,top in list(zip(np.arange(1,K),np.cumsum(neg)[:-1],np.cumsum(pos)[:-1]+1))[::-1]:
        #print(l)
        for h in np.arange(low-offset,top-offset):
            m[l,h]= P_loc[l]*m[l+1,h+inter[l]]+(1-P_loc[l])*m[l+1,h]
        #print(top)
    return P_loc[0]*m[1,inter[0]-offset]+(1-P_loc[0])*m[1,-offset]
def cavity_parallel(P_init, Ts, J, theta, threads,J0 = 'auto', precision=1e-4, max_iter=50):
    """
    Run the dynamical cavity with recursive calls.
    :param P_init: list of floats of length N
    :param T: float
    :param J: sparse.csr_matrix
    :param theta: float (in units of 1/sqrt(<K>))
    :param max_iter: int
    :param precision: float
    :return: P_new it is a  list of dimensions N which contains the probability of active state for each gene.
    In order to help storing, couplings are taken to be +-1, bias is then rescaled by 1/sqrt(<|J_{ij}|>)
    """
    J = J.copy()
    J.data = np.where(J.data > 0, 1, -1)
    N = J.shape[0]
    J_transpose = J.transpose().tolil()
    js = J_transpose.rows  # list of list, structure is [el[i]] where el[i]
    # is the list of  predecessors of gene i ( the index)
    interaction = J_transpose.data  # list of list, structure is [el[i]]
    # where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    Ks = np.array([len(neigh) for neigh in js])  # in degree of each gene
    print(' the compuational cost is equivalent to evaluate a random reg. graph with degree = ',
          (-1 + 8 * np.sqrt(1 + np.mean((Ks + 1) * (Ks + 2)))) / 2)
    if threads < 0:
        pool = Pool()
    else:
        pool = Pool(int(threads))
    if type(J0)==str:
        if J0!='auto':
            raise ValueError('uncomprensible argument')
        if J0 =='auto':
            avg_degree = np.mean(Ks)
            J0 = 1/ np.sqrt(avg_degree)

    data = pool.starmap(cavity, itertools.product([P_init], [js], Ts, [interaction], [N], [Ks], [theta],[J0], [precision],
                                                  [max_iter]))  # run in parallel at different temperatures
    pool.close()
    return data

def cavity_caller(J,T,theta,precision=1e-4,J0 = 'auto'):
    J = J.copy()
    J.data = np.where(J.data > 0, 1, -1)
    J_transpose = J.transpose().tolil()
    js = J_transpose.rows  # list of list, structure is [el[i]] where el[i]
    # is the list of  predecessors of gene i ( the index)
    interaction = J_transpose.data  # list of list, structure is [el[i]]
    # where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    Ks = np.array([len(neigh) for neigh in js])  # in degree of each gene
    max_outdegree = max(Ks)
    max_recursions = int((max_outdegree + 1) * (max_outdegree + 2) / 2)
    '''
    if max_recursions > sys.getrecursionlimit():
        print("Warning! maximum degree larger than default recursion limit, I 'll update recursion limit to",
              max_recursions)
        sys.setrecursionlimit(max_recursions)
    '''
    N = J.shape[0]
    if J0 =='auto':
        avg_degree = np.mean(Ks)
        J0 = 1/ np.sqrt(avg_degree)
    return cavity(np.random.rand(N), js, T, interaction, N, Ks, theta,J0,precision)

def cavity(P, js, T, interaction, N, Ks, theta,J0, precision=1e-4, max_iter=50):
    """
    This runs the dynamical cavity without recursive calls. It creates instead a matrix. This works only if couplings are in the form  \pm J.
    If couplings are in a different form, use it cavity_general
     It computes the node activation probability for a  directed network.
    :param P_init: list of floats of length N
    :param T: float
    :param js: list of list, structure is [el[i]] where el[i] is the list of  predecessors of gene i ( the index)
    :param interaction:  list of list, structure is [el[i] for i in range(N)]
            where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    :param theta: float (in units of 1/sqrt(<K>))
    :param max_iter: int
    :param precision: float
    :return: P_new it is a  list of dimensions N which contains the probability of active state for each gene.
    ----NOTES------
    In order to help storing, couplings are taken to be +-1, at the end the local field is rescaled by 1/sqrt(<|J_{ij}|>)
    Even though code runs for any directed network, results  are exact for fully asymmetric networks only.
    """

    if T == 0:
        return cavity_zero_T(P, js, interaction, N, Ks, theta)

    avg_degree = np.mean(Ks)
    for count in range(max_iter):
        P_new = np.zeros(N)
        for i in range(N):
            j = js[i]
            bias = 0
            K = Ks[i]
            if K ==0:
                P_new[i]=0.5
            else:
                inter = interaction[i]
                P_new[i]=cavity_single_numba(P[j],np.array(inter),T,theta,K,J0)
        if max(np.abs(np.array(P) - np.array(P_new))) < precision:
            P = P_new
            print('finishing after', count, 'iterations')
            break
        if count == max_iter:
            print("Maximum number of repetition reached, but target  precision has not been reached. Precision reached is "+str(max(np.abs(np.array(P) - np.array(P_new)))))

        P = np.array(P_new)
    P = np.array(P)
    return P



def cavity_general(P, js, T, interaction, N, Ks, theta, J0,precision=1e-4, max_iter=50):
    """
    This runs the dynamical cavity with recursive calls. It computes the node activation probability for a  directed network.
    This can be called by cavity_parallel if you want poarallel verion over many Ts, or from cavity_caller for standard use
    :param P_init: list of floats of length N
    :param T: float
    :param js: list of list, structure is [el[i]] where el[i] is the list of  predecessors of gene i ( the index)
    :param interaction:  list of list, structure is [el[i] for i in range(N)]
            where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    :param theta: float (in units of 1/sqrt(<K>))
    :param max_iter: int
    :param precision: float
    :return: P_new it is a  list of dimensions N which contains the probability of active state for each gene.
    ----NOTES------
    In order to help storing, couplings are taken to be +-1, at the end the local field is rescaled by 1/sqrt(<|J_{ij}|>)
    Even though code runs for any directed network, results  are exact for fully asymmetric networks only.
    """

    if T == 0:
        return cavity_zero_T(P, js, interaction, N, Ks, theta)


    @lru_cache(maxsize=None)
    def recursion(bias, l):
        if (l == K):
            bias = (bias - theta) *J0
            return np.tanh(bias / 2 / T)

        include = P[j[l]] * recursion(bias + inter[l], l + 1)  # include node l with prob. P[j[l]]
        exclude = (1 - P[j[l]]) * recursion(bias, l + 1)  # ignore node l
        return include + exclude

    for count in range(max_iter):
        P_new = []
        for i in range(N):
            j = js[i]
            bias = 0
            K = Ks[i]
            inter = interaction[i]
            P_new += [0.5 + 0.5 * recursion(bias, 0)]
            if count == 0:
                if recursion.cache_info().currsize > (K + 1) * (K + 2) / 2:
                    print('caching is storing more than one expects at i = ', i)
            recursion.cache_clear()
        if max(np.abs(np.array(P) - np.array(P_new))) < precision:
            P = P_new
            print('finishing after', count, 'iterations')
            break
        if count == max_iter:
            print("Maximum number of repetition reached, but target  precision has not been reached. ")

        P = P_new
    P = np.array(P)
    return P


def cavity_zero_T(P, js, interaction, N, Ks, theta, J0,precision=1e-4, max_iter=50):
    """
    Dynamical cavity at T = 0. It replaces tanh with sign.
    :param P: list of floats of length N
    :param theta: float (in units of 1/sqrt(<K>))
    :param max_iter: int
    :param precision: float
    :return: P_new it is a  list of dimensions N which contains the probability of active state for each gene.
    In order to help storing, couplings are taken to be +-1, bias is then rescaled by 1/|J_{ij}|
    """
    avg_degree = np.mean(Ks)

    @lru_cache(maxsize=None)
    def recursion(bias, l):
        if (l == K):
            bias = (bias - theta) *J0
            if bias > 0:
                return 1
            elif bias == 0:
                return 0
            else:
                return -1

        include = P[j[l]] * recursion(bias + inter[l], l + 1)  # include node l with prob. P[j[l]]
        exclude = (1 - P[j[l]]) * recursion(bias, l + 1)  # ignore node l
        return include + exclude

    for count in range(max_iter):
        P_new = []
        for i in range(N):
            j = js[i]
            bias = 0
            K = Ks[i]
            inter = interaction[i]
            P_new += [0.5 + 0.5 * recursion(bias, 0)]
            if count == 0:
                if recursion.cache_info().currsize > (K + 1) * (K + 2) / 2:
                    print('caching is storing more than one expects at i = ', i)
            recursion.cache_clear()
        if max(np.abs(np.array(P) - np.array(P_new))) < precision:
            P = P_new
            print('finishing after', count, 'iterations')
            break
        if count == max_iter:
            print("Maximum number of repetition reached, but target  precision has not been reached. ")

        P = P_new
    P = np.array(P)
    return P

def cavity_AND_parallel(P_g,T,mu_s,Ks,interaction,j_s,theta,J0, precision = 1e-4,max_iter = 50):
    avg_degree = np.mean(Ks)
    @lru_cache(maxsize = None)
    def recursion(bias, l):
        if (l == K):
            bias = (bias-theta)*J0
            return np.tanh(bias/2/T)

        include = P_t[mu[l]] * recursion(bias+inter[l], l+1 )# include node l with prob. P[j[l]]
        exclude = (1-P_t[mu[l]]) *  recursion(bias, l+1 )# ignore node l
        return  include + exclude
    N1 = len(Ks)
    N2 = len(j_s)
    for count in range(max_iter):
        P_g_new =np.zeros(N1)
        P_t = [np.prod(P_g[j_s[mu]]) for mu in range(N2)]
        for i in range(N1):
            mu = mu_s[i]# list of predecessors of node i
            bias = 0
            K=Ks[i]
            inter = interaction[i]
            #theta = interaction_sum[i]/2
            P_g_new[i] = 0.5+0.5*recursion(bias,0)
            if count == 0:
                if recursion.cache_info().currsize>(K+1)*(K+2)/2:
                    print('caching is storing more than one expects at i = ',i)
            recursion.cache_clear()
        if max(np.abs(np.array(P_g)-np.array(P_g_new)))<precision:
            P_g = P_g_new
            print('finishing after', count,'iterations')
            break
        if count ==max_iter :
            print("Maximum number of repetition reached, but target  precision has not been reached. ")

        P_g = P_g_new
    return P_g

def cavity_AND(P_g,T,R,M,theta,J0 = 'auto', precision = 1e-4,max_iter = 50):
    """
    :param P_init: list of floats of length N
    :param T: float
    :param J: sparse.csr_matrix
    :param theta: float (in units of 1/sqrt(<K>))
    :param max_iter: int
    :param precision: float
    :return: P_new it is a  list of dimensions N which contains the probability of active state for each gene.
    In order to help storing, couplings are taken to be +-1, bias is then rescaled by 1/|J_{ij}|
    """

    N1,N2 = M.shape
    R = R.tocsc()
    M = M.tocsc()
    mu_s = [R.indices[R.indptr[i]:R.indptr[i + 1]] for i in range(N1)]  # list of list, structure is [el[i]] where el[i]
    # is the list of  predecessors of gene  i ( the index)
    interaction = [R.data[R.indptr[i]:R.indptr[i + 1]] for i in range(N1)]  # list of list, structure is [el[i]]
    # where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    Ks = np.diff(R.indptr)  # in degree of each gene
    print(' the compuational cost is equivalent to evaluate a random reg. graph with degree = ',\
           np.mean((Ks+1) * (Ks + 2)) / 2)
    interaction = [np.where(inter>0,1,-1) for inter in interaction]#make interactions as +=1 rather than floats
    avg_degree = len(R.data) / N1
    j_s = [M.indices[M.indptr[mu]:M.indptr[mu + 1]] for mu in range(N2)]  # list of list, structure is [el[mu]] where el[mu]
    if J0 =='auto':
        avg_degree = np.mean(Ks)
        J0 = 1/ np.sqrt(avg_degree)

    cavity_AND_parallel(P_g,T,mu_s,Ks,interaction,j_s,theta,J0)
    # is the predecessor of TF mu
    #interaction_sum = np.sum(interaction,axis = 1)
#=----- END of cavity-----

def cavity_faster(P,T,J,theta = 0, precision = 1e-4,max_iter = 50):
    """
    This is the old version, here for backward compatibility
    :param P_init: list of floats of length N
    :param T: float
    :param J: sparse.csr_matrix
    :param theta: float (in units of 1/sqrt(<K>))
    :param max_iter: int
    :param precision: float
    :return: P_new it is a  list of dimensions N which contains the probability of active state for each gene.
    In order to help storing, couplings are taken to be +-1, bias is then rescaled by 1/|J_{ij}|
    """
    N = J.shape[0]
    A = J.tocsc()
    js = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(N)]  # list of list, structure is [el[i]] where el[i]
    # is the list of  predecessors of gene i ( the index)
    interaction = [A.data[A.indptr[i]:A.indptr[i + 1]] for i in range(N)]  # list of list, structure is [el[i]]
    # where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    Ks = np.diff(A.indptr)  # in degree of each gene
    print(' the compuational cost is equivalent to evaluate a random reg. graph with degree = ',\
           np.mean((Ks+1) * (Ks + 2)) / 2)
    interaction = [np.where(inter>0,1,-1) for inter in interaction]#make interactions as +=1 rather than floats
    avg_degree = len(J.data) / N
    @lru_cache(maxsize = None)
    def recursion(bias, l):
        if (l == K):
            bias = (bias-theta)/ np.sqrt(avg_degree)
            return np.tanh(bias/2/T)

        include = P[j[l]] * recursion(bias+inter[l], l+1 )# include node l with prob. P[j[l]]
        exclude = (1-P[j[l]]) *  recursion(bias, l+1 )# ignore node l
        return  include + exclude
    for count in range(max_iter):
        P_new =[]
        for i in range(N):
            j = js[i]
            bias = 0
            K=Ks[i]
            inter = interaction[i]
            P_new +=[0.5+0.5*recursion(bias,0)]
            if count == 0:
                if recursion.cache_info().currsize>(K+1)*(K+2)/2:
                    print('caching is storing more than one expects at i = ',i)
            recursion.cache_clear()
        if max(np.abs(np.array(P)-np.array(P_new)))<precision:
            P = P_new
            print('finishing after', count,'iterations')
            break
        if count ==max_iter :
            print("Maximum number of repetition reached, but target  precision has not been reached. ")

        P = P_new
    P = np.array(P)
    return P

