#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains an implementation of the Explore-then-Verify algorithm from
[Karnin2016].

[Karnin2016]: Zohar Karnin. Verification Based Solution for Structured MAB
Problems, In Proceedings of Conference on Neural Information Processing Systems (NIPS), 2016
"""

import numpy as np
import PM as pm

def ExploreThenVerify(P,gamma,show_progress = False):
    """ 
This is Alg. 1 in [Karnin2016].

REMARKS:
(1) In l.6 of the pseudocode in [Karnin2016] the authors write "delta/2r^{2}".
    Regarding the proof of Thm. 3 on p.4, they mean "delta / (2r^{2})".
(2) As suggested after Cor. 4 in [Karnin2016] we choose "kappa=min(1/3, 1/log(1/delta))" 
    as failure prob. for the exploration phase.
(3) Due to its necessary modification, Explorer may return "FAIL". If this happens,
    we simply continue with the next iteration.

[Karnin2016]: Zohar Karnin. Verification Based Solution for Structured MAB
Problems, In Proceedings of Conference on Neural Information Processing Systems (NIPS), 2016
    
EXAMPLE
P = pm.PLPM([1,2,3,4],2)
gamma = 0.05
print("ExploreThenVerify:",ExploreThenVerify(P,0.05,show_progress=True), "t :", P.get_time() )
    """
    
    r = 1
    kappa = min(1/3, 1 / np.log(1/gamma))       # Cf. the remark after Cor. 4 in [Karnin2016]
    while True:
        result = Explorer(P,kappa)
        if result == "FAIL":                    # Cf. REMARK (3)
            if show_progress is True:
                print("r="+str(r))
                print("Explorer ended t="+str(P.get_time())+\
                      " with 'FAIL'")
        else:                                   # Cf. REMARK (3)
            candidate, witnesses = result
            if show_progress is True:
                print("r="+str(r))
                print("Explorer ended at t="+str(P.get_time())+\
                      " with candidate="+str(candidate))
                print("The 'witnesses' for "+str(list(range(0,P.m)))+" are "+str(witnesses)+" (-1 indicates the candidate)")
            result = Verifier(P, gamma= gamma / (2 * (r**2)), candidate = candidate,\
                              witnesses = witnesses)
            if show_progress is True:
                print("Verifier ended with result "+str(result)+" at t="+str(P.get_time())+"\n")
            if result == "SUCCESS":
                return(candidate)
        r = r+1

def Explorer(P,kappa):
    """
This is Alg. 3 in [Karnin2016].

REMARKS:
(1) In the last line of Alg. 3 in [Karnin2016] it is not specified what happens
    if the candidate "\hat{x}" does not exist. In this case, we simply return 
    "FAIL". This modification does not lead to violations of the guarantees 
    (error estimate and sample complexity) of ExploreThenVerify.

[Karnin2016]: Zohar Karnin. Verification Based Solution for Structured MAB
Problems, In Proceedings of Conference on Neural Information Processing Systems (NIPS), 2016

For an EXAMPLE see ExploreThenVerify above.
    """
    assert P.k == 2, "k=2 required."
    m = P.m
    Q = list([])
    for i in range(0,m):
        for j in range(i+1,m):
            Q.append((i,j))
    t = 1
    win_matrix = np.zeros((m,m))    # if i wins against j, win_matrix[i][j] resp. win_matrix[j][i] is in- resp. de-creased by 1
    num_pulls = np.zeros((m,m))     # num_pulls[i][j] is the number of times i have been compared to j  so far.
    
    L = np.zeros((m,m))     # L[x][y] is l_xy
    U = np.zeros((m,m))     # U[x][y] is u_xy
    while len(Q) > 0:
        for S in Q:
            S_buf, result = P.pull_arm(S,size=1)
            if result[0]==1:
                win_matrix[S_buf[0],S_buf[1]] += 1  
                win_matrix[S_buf[1],S_buf[0]] -= 1
            if result[1]==1:
                win_matrix[S_buf[0],S_buf[1]] -= 1  
                win_matrix[S_buf[1],S_buf[0]] += 1
            num_pulls[S_buf[0],S_buf[1]] += 1
            num_pulls[S_buf[1],S_buf[0]] += 1
       
        gamma_t = np.sqrt( 2 * np.log( 2*t*t*m*m /kappa ) / t)
        for i in range(0,m):
            for j in range(0,m):
                if i!= j:
                    L[i][j] = win_matrix[i][j] / num_pulls[i][j] - gamma_t      
                    U[i][j] = win_matrix[i][j] / num_pulls[i][j] + gamma_t 
        for (x,y) in Q:
            if L[x][y] > 0:
                Q.remove((x,y))
            elif U[x][y] < 0 and 2*U[x][y] < L[x][y]:
                Q.remove((x,y)) 
            elif L[x][y] < 0:
                buf = False
                for j in range(0,m):
                    if j != y and j!=x and L[x][y] > U[x][j]:
                        buf = True  
                if buf is True:
                    Q.remove((x,y))
        t = t+1
        
    candidates = np.argwhere(np.sum(L < 0, axis=1) == 0)    
    if len(candidates) == 0:                            # Confer REMARK (1).
        return("FAIL")                                  
    candidates = candidates[0]
    candidate = np.random.choice(candidates,1)[0]
    witnesses = np.zeros(m,dtype=int) - 1
    for x in range(0,m):
        if x != candidate:
            minimum = np.inf 
            for y in range(0,m):
                if x!= y and U[x][y] < minimum:
                    witnesses[x] = y
                    minimum = U[x][y]
    return(candidate, witnesses)
    
def Verifier(P,gamma,candidate,witnesses):
    """ 
This is Alg. 4 in [Karnin2016].

[Karnin2016]: Zohar Karnin. Verification Based Solution for Structured MAB
Problems, In Proceedings of Conference on Neural Information Processing Systems (NIPS), 2016

For an EXAMPLE see ExploreThenVerify above.
    """
    assert P.k == 2, "k=2 required."
    m = P.m
    Q = list(range(0,m))
    Q.remove(candidate)
    t = 1
    win_matrix = np.zeros((m,m))    # if i wins against j, win_matrix[i][j] resp. win_matrix[j][i] is in- resp. de-creased by 1
    num_pulls = np.zeros((m,m))     # num_pulls[i][j] is the number of times i have been compared to j  so far.
    L = np.zeros((m,m))     # L[x][y] is l_xy
    U = np.zeros((m,m))     # U[x][y] is u_xy
    while len(Q)>0:
        for x in Q:
            y = witnesses[x]
            S_buf, result = P.pull_arm([x,y],size=1)
            if result[0]==1:
                win_matrix[S_buf[0],S_buf[1]] += 1  
                win_matrix[S_buf[1],S_buf[0]] -= 1
            if result[1]==1:
                win_matrix[S_buf[0],S_buf[1]] -= 1  
                win_matrix[S_buf[1],S_buf[0]] += 1
            num_pulls[S_buf[0],S_buf[1]] += 1
            num_pulls[S_buf[1],S_buf[0]] += 1 
            
        gamma_t = np.sqrt( 2 * np.log( 2*t*t*m*m /gamma ) / t)
        for i in range(0,m):
            for j in range(0,m):
                if num_pulls[i][j] != 0:
                    L[i][j] = win_matrix[i][j] / num_pulls[i][j] - gamma_t      
                    U[i][j] = win_matrix[i][j] / num_pulls[i][j] + gamma_t
        for x in Q:
            y = witnesses[x]
            if U[x][y] < 0:
                Q.remove(x)
            if L[x][y] > 0:
                return("FAIL")  
        t = t+1
    return("SUCCESS")