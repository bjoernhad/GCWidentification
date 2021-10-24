#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains implementations from the algorithms in  the paper
"Identification of the Generalized Condorcet Winner
in Multi-dueling Bandits", which is referred to as "our paper".
"""
import numpy as np
import PM as pm
import random 

###############################################################################
#       ALGORITHM 1 (Non-sequential solution for the case m=k)
###############################################################################
def algo_4(gamma,h,P):
    """ 
This is Algorithm 1 from our GCW paper.

EXAMPLE
P = pm.create_PL_PM([0.3,1,0.9,0.3,0.2],5)
k, gamma, h = 5, 0.05, 0.1
print("Output of algo_4:",algo_4(k,gamma,h,P))
P.show()
    """
    assert type(P) is pm.PM or type(P) is pm.PLPM, "P has to be a PM of size (k,k)"
    k = P.get_size()[0]
    assert P.get_size()==(k,k), "P has to be a PM of size (k,k)"
    # assert P.is_filled(), "P has to be completely filled."
    assert 0<gamma and gamma <1 and 0<h and h<0.5, "algo_4 requires 0<gamma<1 and 0<h<0.5"
    T = int( np.ceil( 8* np.log(4 / gamma ) /  (h**2)) )
    S = tuple(range(0,k))
    S, observations = P.pull_arm(S=S, size=T)
    observations = np.array(observations)
    candidates = np.where(observations == max(observations))[0]
    result = random.sample(list(candidates),1)
    return(result,sum(observations))
    

###############################################################################
#       ALGORITHM 2 (Main Component for Algorithms ..., also for m=k)
###############################################################################
def algo_1(gamma,h,P,S):
    """ 
This is Algorithm 1 from our GCW paper started on P(.|S)

EXAMPLE
k = 3
P = pm.PLPM([0.3,0.88,0.9,0.3,0.2],k)
gamma, h = 0.05, 0.2
S = tuple(range(0,k))
print("Output of algo_1:",algo_1(gamma,h,P,S),"\n")
P.show()
    """
    assert type(P) is pm.PM or type(P) is pm.PLPM, "P has to be a PM"
    m,k = P.get_size()
    assert 0<gamma and gamma <1 and 0<h and h<0.5, "algo_1 requires 0<gamma<1 and 0<h<0.5"
    # T = int( np.ceil( np.log(4 / gamma ) / (2 * (h**2)) ) )
    T = int( np.ceil( 8* np.log(4 / gamma ) /  (h**2) ) )
    S, observations = P.pull_arm(S,size=T)  #Afterwards, S is sorted
    observations = np.array(observations)
    
    #Position of the sample mode of S (based on the just made observations) (in S):
    candidate_pos = list( np.where(observations == max(observations)) )[0]  
    if len(candidate_pos) > 1:
        return("UNSURE")
    candidate_pos = int( candidate_pos[0] )
    hat_p = observations / T 
    # print(hat_p,candidate_pos)  #For Debugging
    for j in range(0,len(S)):
        if j!= candidate_pos and hat_p[candidate_pos] <= hat_p[j] + h :
            return("UNSURE")
    return(S[candidate_pos])
    
    # S = tuple(sorted(S))    
    # old_observations = np.array(P.arms[S][1])
    # # P.show()      # For Debugging
    # P.pull_arm(S=S, size=T)
    # # P.show()      # For Debugging
    # new_observations = np.array(P.arms[S][1]) - old_observations
    # # print(new_observations)       # For Debugging
    
    # #Position of the sample mode of S (based on the NEW observations) (in S):
    # candidate_pos = list( np.where(new_observations == max(new_observations)) )   
    
    # if len(candidate_pos) > 1:
    #     return("UNSURE")
    # candidate_pos = int( candidate_pos[0][0] )
    # hat_p = new_observations / T 
    # # print(hat_p,candidate_pos)  #For Debugging
    # for j in range(0,len(S)):
    #     if j!= candidate_pos and hat_p[candidate_pos] <= hat_p[j] + h :
    #         return("UNSURE")
    # return(S[candidate_pos])


###############################################################################
#       ALGORITHM 3 (Solution for the case m=k and \Delta^{0})
###############################################################################
def algo_2(gamma,P,S):
    """ 
This is Algorithm 1 from our GCW paper started on P(.|S)

EXAMPLE
k, gamma, h = 4, 0.03, 0.2
P = pm.PLPM([0.3,0.75,0.9,0.3,0.2],k)
print("Output of algo3:",algo_2(gamma,P,[4,1,2,3]),"\n")
P.show()
    """
    assert type(P) is pm.PM or type(P) is pm.PLPM, "P has to be a PM"
    m,k = P.get_size()
    assert 0<gamma and gamma <1, "algo_2 requires 0<gamma<1"
    feedback = "UNSURE"
    s = 1
    while feedback == "UNSURE":
        feedback = algo_1(6*gamma / ((np.pi**2) * (s**2) ), 2**(-s-1),P,S)
        s = s+1
    return(feedback)


###############################################################################
#       DVORETZKY-KIEFER-WOLFOWITZ TOURNAMENT (DKWT)
#       (Solution for the case m>k and under assumptions 
#                    \existsGCW and \Delta^{0})
#
#   REMARKS:
#   (i) Whenever picking some elements from a set A, we pick them UNIFORMLY AT
#       RANDOM.
###############################################################################
    
def DKWT(gamma,P,debugging=False):
    """ 
This is Algorithm 1 from our GCW paper.

EXAMPLE
k, gamma, h = 3, 0.03, 0.2
P = pm.PLPM([0.3,0.98,0.9,0.94,0.2],k)
print("Output of algo4:",DKWT(gamma, P),"\n")
P.show()
    """
    assert type(P) is pm.PM or type(P) is pm.PLPM, "P has to be a PM"
    m,k = P.get_size()
    assert 0<gamma and gamma <1, "DKWT requires 0<gamma<1"
    # S = tuple(range(0,k))
    S = tuple(random.sample(range(0,m),k))              # cf. REMARK (i)
    F = list(range(0,m)) 
    gamma_prime = gamma / np.ceil(m/(k-1))
    s = 1
    while s <= int(np.ceil(m/(k-1))-1):
        S = tuple(S)
        i_s = algo_2(gamma_prime,P,S)       #i_s
        if debugging is True:
            print("s",s,"S: ",S,"F: ",F,"i_s: ",i_s)   #For Debugging
        F = list(set(F) - set(S))   # Removes all elements of S from F
        random.shuffle(F)           # randomly shuffle, cf. REMARK (i)
        if len(F) >= k-1:
            S = list([i_s]) + F[:(k-1)] 
        elif len(F)>0:
            S = list([int(i_s)]) + F
            buf = list(set(range(0,m)) - set(S))
            random.shuffle(buf)             # randomly shuffle, cf. REMARK (i)
            for l in range(0,int(k-len(S))):
                S = S + list([buf[l]])
        assert len(S)==k, "An error occured in algo_5, len(S)=k is violated"
        s = s + 1
    i_s = algo_2(gamma_prime,P,S)
    if debugging is True:
        print("s",s,"S: ",S,"F: ",F,"i_s: ",i_s)   #For Debugging
    return(i_s)   

# OLD VERSION:
# def DKWT(gamma,P,debugging=False):
#     """ 
# This is Algorithm 1 from our GCW paper.

# EXAMPLE
# k, gamma, h = 3, 0.03, 0.2
# P = pm.PLPM([0.3,0.98,0.9,0.94,0.2],k)
# print("Output of algo4:",DKWT(gamma, P),"\n")
# P.show()
#     """
#     assert type(P) is pm.PM or type(P) is pm.PLPM, "P has to be a PM"
#     m,k = P.get_size()
#     assert 0<gamma and gamma <1, "DKWT requires 0<gamma<1"
#     S = tuple(range(0,k))
#     F = list(range(0,m)) 
#     gamma_prime = gamma / np.ceil(m/(k-1))
#     for s in range(0,int(np.ceil(m/(k-1)))):        # CEIL OR FLOOR? --> TO DO!
#         S = tuple(S)
#         i = algo_2(gamma_prime,P,S)
#         if debugging is True:
#             print("S: ",S,"F: ",F,"i: ",i)   #For Debugging
#         F = list(set(F) - set(S))   # Removes all elements of S from F
#         if len(F) >= k-1:
#             S = list([i]) + F[:(k-1)]
#         elif len(F)>0:
#             S = list([int(i)]) + F
#             buf = list(set(range(0,m)) - set(S))
#             for l in range(0,int(k-len(S))):
#                 S = S + list([buf[l]])
#         assert len(S)==k, "An error occured in algo_5, len(S)=k is violated"
#     i = algo_2(gamma_prime,P,S)
#     if debugging is True:
#         print("S: ",S,"F: ",F,"i: ",i)   #For Debugging
#     return(i)   

###############################################################################
#       ALGORITHM 5 (Solution for the case m>k and under assumption
#                    \existshGCW )
#
#   REMARKS:
#   (i) Whenever picking some elements from a set A, we pick them UNIFORMLY AT
#       RANDOM.
############################################################################### 
def algo_5(gamma,h,P,debugging=False):
    """ 
    
EXAMPLE
k, gamma, h = 3, 0.25, 0.02
P = pm.PLPM([0.3,0.7,0.94,0.99,0.2],k)
print("Output of algo_5:",algo_5(gamma, h, P),"\n")
P.show()
    """
    assert type(P) is pm.PM or type(P) is pm.PLPM, "P has to be a PM"
    m,k = P.get_size()
    assert 0<gamma and gamma <1, "algo_5 requires 0<gamma<1"
    assert 0<h and h<1, "algo_5 requires 0<h<1"
    i_s = "UNSURE"
    h_prime = h/3
    gamma_prime = gamma / np.ceil( m / (k-1))
    # S = tuple(range(0,k))
    S = tuple(random.sample(range(0,m),k))              # cf. REMARK (i)
    F = list(range(0,m))
    s = 1
    while len(F) > 0:
        i_s = algo_1(gamma_prime,h_prime,P,S)
        if debugging is True:
            print("s:",s,"S: ",S,"F: ",F,"i_s: ",i_s)      #For Debugging
        F = list(set(F) - set(S))   # Removes all elements of S from F
        buf = list(set(range(0,m))-set(F))
        random.shuffle(buf)         # randomly shuffle, cf. REMARK (i)
        if i_s == "UNSURE":
            F_buf = list(F)         # CHECK THIS!
            random.shuffle(F_buf)   # CHECK THIS!
            buf = F_buf + buf             # CHECK THIS!
            S = buf[:k]
        else: 
            if len(F) >= k-1:
                random.shuffle(F)       # randomly shuffle, cf. REMARK (i)
                S = list([i_s]) + F[:(k-1)]
            elif len(F)>0:
                S = list([int(i_s)]) + F
                buf = list(set(range(0,m)) - set(S))
                random.shuffle(buf)       # randomly shuffle, cf. REMARK (i)
                for l in range(0,int(k-len(S))):
                    S = S + list([buf[l]])
        assert len(S)==k, "An error occured in algo_5, len(S)=k is violated"
        s = s+1
    i_s = algo_1(gamma_prime,h_prime,P,S)
    if debugging is True:
        print("s:",s,"S: ",S,"F: ",F,"i_s: ",i_s)      #For Debugging
    if i_s == "UNSURE":
        return(1)
    return(i_s)
