#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the empirical comparison of the lower sample complexity bounds
stated in Prop. 4.1 and Thm. 5.2 from our paper.
"""

import numpy as np


def LB_1(p,gamma,k):
    """
Calculates the LB stated in Prop. 4.1.

EXAMPLE
p = np.array([0.2,0.2,0.35,0.25])
print(LB_1(p,gamma=0.05,k=4))
    """
    assert len(list(p)) == k, "len(p) has to be k."
    p = np.array(p)
    i = np.argmax(p)
    result = 0
    for j in range(0,k):
        if j!= i:
            buf= 0
            z = (p[i] - p[j])/(2*( p[i]+p[j] ) )
            buf = (1-2*gamma)/(2*z) * np.ceil( np.log((1-gamma) / gamma) / np.log( (0.5+z)/(0.5-z)  ))            
            buf = buf / (p[i]+p[j])
            result = max(result,buf)
    return(result)



def KL(p,q):
    """ 
Calculates the Kullback-Leibler divergence of two categ.
random variables with distributions p and q.

EXAMPLE
p = [0.1,0.2,0.3,0.4]
q = [0.4,0.3,0.2,0.1]
print(KL(p,q))
    """
    k = len(list(p))
    result = 0
    for y in range(0,k):
        if p[y]==0 and q[y] != 0:
            return np.inf 
        if p[y] > 0:
            result += p[y] * np.log(p[y] / q[y])
    return(result)

def flip(p,i,j):
    q = np.array(p).copy()
    q[i] = p[j]
    q[j] = p[i]
    return(q)


def LB_2(p,gamma,k):
    """
This calculates the Lower Bound stated in Thm. 5.2.
p = np.array([0.2,0.2,0.35,0.25])
print(LB_2(p,gamma=0.05,k=4))
    """
    assert len(list(p)) == k, "len(p) has to be k"
    p = np.array(p)
    i = np.argmax(p)
    result = 0
    for l in range(0,k):
        if l != i:
            result += 1 / KL( p, flip(p,i,l) )
    result = result * np.log((2.4*gamma)**(-1)) / (k-1)
    return(result)



def compare_both_bounds(nr_runs=10,k=10,gamma=0.05,show_progress = False):
    p = np.zeros(k)
    results_LB1 = list()
    results_LB2 = list()
    for run in range(0,nr_runs):
        for i in range(0,k):
            p[i] = np.random.uniform()
        p = p/np.sum(p)
        res1 = LB_1(p,gamma,k)
        res2 = LB_2(p,gamma,k)
        if show_progress is True:
            print("Run:",run,"  LB1:",res1,"  LB2:",res2)
            print("   (p for previous run:",p)
            # print("LB_2v2:",LB_2v2(p,gamma,k))
            # assert LB_2v2(p,gamma,k) == res2, "ERROR"
        results_LB1.append(res1)
        results_LB2.append(res2)
    results_LB1 = np.array(results_LB1)
    results_LB2 = np.array(results_LB2)
    print("LB1>LB2 in "+str(np.sum(results_LB1 > results_LB2))+" runs.")
    print("LB1<=LB2 in "+str(np.sum(results_LB1 <= results_LB2))+" runs.")
    print("Mean(LB1):",np.mean(results_LB1)," Std-error(LB1):",np.std(results_LB1)/np.sqrt(nr_runs))
    print("Mean(LB2):",np.mean(results_LB2)," Std-error(LB2):",np.std(results_LB2)/np.sqrt(nr_runs))
    return(p / sum(p))


p = np.array([0.2,0.2,0.15,0.2,0.25])
print(LB_1(p,gamma=0.05,k=5))
print(LB_2(p,gamma=0.05,k=5))

gammas = [0.01,0.05,0.10]
ks = [5,10,15]
for gamma in gammas:
    for k in ks:
        print("\n \n k="+str(k)+" gamma="+str(gamma))
        compare_both_bounds(nr_runs=1000,k=k,gamma=gamma,show_progress = False)
