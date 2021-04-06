'''Run the dynamical cavity with recursive calls.'''
import numpy as np
import itertools
import warnings
import scipy
import numpy.random
from multiprocessing import Pool
import argparse
from functools import lru_cache
import sys
import os


def cavity_parallel(P_init, Ts, J, theta, threads, precision=1e-4, max_iter=50):
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
    data = pool.starmap(cavity, itertools.product([P_init], [js], Ts, [interaction], [N], [Ks], [theta], [precision],
                                                  [max_iter]))  # run in parallel at different temperatures
    pool.close()
    return data


def cavity(P, js, T, interaction, N, Ks, theta, precision=1e-4, max_iter=50):
    """
    This runs the dynamical cavity with recursive calls. It computes the node activation probability for a  directed network.

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

    @lru_cache(maxsize=None)
    def recursion(bias, l):
        if (l == K):
            bias = (bias - theta) / np.sqrt(avg_degree)
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


def cavity_zero_T(P, js, interaction, N, Ks, theta, precision=1e-4, max_iter=50):
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
            bias = (bias - theta) / np.sqrt(avg_degree)
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
