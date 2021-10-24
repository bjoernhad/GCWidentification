#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the following algorithms:
    - Partition
    - Score_Estimate
    - RankBreaking
    - Divide_and_Battle (Alg. 2 in [Saha2019])
    - PAC_Best_Item (Alg. 5 in [Saha2020])
    - PAC_Wrapper (Alg. 1 in [Saha2020])   

SOURCES:
[Saha2020]: Aadirupa Saha, Aditya Gopalan
From PAC to Instance-Optimal Sample Complexity in the Plackett-Luce Model, 
Proceedings of ICML (2020)
URL: http://proceedings.mlr.press/v119/saha20b.html

[Saha2019]: Aadirupa Saha, Aditya Gopalan
PAC Battling Bandits in the Plackett-Luce Model
Proceedings of ALT (2019)
URL: http://proceedings.mlr.press/v98/saha19a.html
"""
import random
import numpy as np 
import PM as pm

###############################################################################
#   PREREQUISITES 
###############################################################################

###############################################################################
#      Algorithm: Partition
#   Alg. 2 on p.11 in [Saha2020]
###############################################################################
def Partition(A,k):
    """ 
Cf. Algorithm 2 on p.19 in [Saha2020]

EXAMPLE
print(Partition(range(0,22),5))
    """
    B = int(np.ceil(len(A) / k))
    batches = list()
    for i in range(0,B):
        batches.append(list(A[i*k:(i+1)*k]))
    return(batches)

###############################################################################
#      Algorithm: Score_Estimate
#   Alg. 3 on p.12 in [Saha2020]
###############################################################################
def Score_Estimate(S,b,delta,P):
    """
Cf. Algorithm 3 on p.19 in [Saha2020].

EXAMPLE
P = pm.PLPM([0.8,1,0.9,0.3,0.2],4)
P.show()
print(Score_Estimate([2,3,1],0,0.03,P))
    """
    assert b not in set(S), "b MUST NOT be an element of S!"
    S_prime =  tuple(sorted(list(S)+list([b]))) #The arm to pull
    d = 10 * np.log(4/delta)
    pos_b = np.where(np.array(S_prime) == b)[0][0]
    observations = np.zeros(len(S_prime))
    # buf = P.time    #For Debugging
    while observations[pos_b] < d:
        observations += P.pull_arm(S_prime, size=1)[1]
    # print("...Time spent for Score_Estimate",P.time - buf)  #For Debugging
    return( (sum(observations)-d) / d)

###############################################################################
#      Algorithm: RankBreaking
#   Alg. 4 on p.13 in [Saha2020]
###############################################################################
def RankBreaking(S,top_m_ranking,W):
    """
Cf. Algorithm 4 on p.21 in [Saha2020].

EXAMPLE
W = np.ones((4,4))
W = RankBreaking((1,2,3,8),[3],W)
print(W)
    """
    S = tuple(S)
    k = len(S)
    assert type(top_m_ranking) is list and len(top_m_ranking) <= k-1, "top_m_ranking has to be a tuple."
    m = len(top_m_ranking)
    assert set(top_m_ranking).issubset(set(S)), "top_m_ranking has to be a subset of S"
    assert type(W) is np.ndarray and W.shape == (k,k), "W has to be an np.array of size |S| times |S|"
    S_prime = list(S)
    for l in range(0,m):
        s = top_m_ranking[l]
        pos_s = np.where(np.array(S) == s)[0][0]
        S_prime = list( set(S) - set([s]))
        for i in S_prime:
            pos_i = np.where(np.array(S) == i)[0][0]
            print(S[pos_s],S[pos_i])
            W[pos_s][pos_i] += 1
    return(W)


###############################################################################
#      Algorithm: Divide_and_Battle 
#   Alg. 2 on p.24 in [Saha2019]
# 
#   MAJOR ISSUES:
#   (I) Lines 21--22 of Alg.2 in [Saha2019] are only executed if none of the 
#       conditions in l.16 and l.18 are fulfilled. In case |S|<= k holds, 
#       ONLY l.19 is executed; this way, the definition of S in l.19 is obsolete,
#       the next iteration of the while loop would be executed 
#       with the exactly the same values of \eps_{l}, \delta_{l}, S and \mathcal{G}_{g}
#       again, which causes an endless iteration through the while loop.
#   -> Fix: Simply delete "else" in l.20. Note that this is consistent with
#       the procedure described in Algo5 of [Saha2020]
#
#   MINOR ISSUES:
#   (a) R_{1} resp. R_{l} may not be defined; they are only defined in l.7 or 21 
#       in case |G_{\mathcal{G}}| < k is fulfilled
#    -> Fix: As e.g. in Alg. 5 of [Saha2020] we initialize R_{l} = \emptyset.
#
#   REMARKS:
#   (i) Before using Partition on a set A, we randomly shuffle the elements in
#        it.
###############################################################################
def Divide_and_Battle(A,eps,delta,P):
    """ 

EXAMPLE
A = [0,1,2,3,4,5,6,8]
k, m_top_param, eps, delta = 3, 1, 0.1, 0.2
P = pm.PLPM([0.3,0.8,0.9,0.3,0.2,0.6,0.8,0.2,0.3],k)
result = Divide_and_Battle(A,eps,delta,P)
print(result)
P.show()
    """
    k = P.k                                 # k
    S = A                                   # \mathcal{A}
    eps_l = eps / 8                         # \epsilon_{l} 
    delta_l  = delta / 2                    # \delta_{l}
    G = int(np.ceil( len(A) / k))           # G
    random.shuffle(S)
    G_batches = Partition(S,k)              # \mathcal{G}_{1},...,\mathcal{G}_{G}
    R_l = []                                # \mathcal{R}_{l}    # FIX OF ISSUE (a)
    if len(G_batches[-1]) < k:
        R_l = G_batches[-1]
        G = G - 1
    while True:
        S = set([])
        delta_l = delta_l / 2
        eps_l = 3 * eps_l / 4
        
        t = int( np.ceil( k / (2 * (eps_l **2)) ) * np.log( k / delta_l ))
        for g in range(0,G):
            Query, w = P.pull_arm(S=G_batches[g], size=t)
            c_g = Query[np.argmax(w)]
            S = list(set(S).union(set([c_g])))
        
        # print("Batches:",G_batches,"Batch-Winners(S):",S,"R_l:",R_l,"eps_l:",eps_l,"delta_l:",delta_l) #For Debugging
        S = list(set(S).union(set(R_l)))
        if len(S) == 1:
            break   #breaks out of the while loop
        elif len(S) <= k:
            buf = list( set(A) - set(S) )
            S = list(S) + list(random.sample(buf,k-len(S)))
            eps_l = 2*eps / 3
            delta_l = delta

        ######################################
        # FIX OF ISSUE (I)
        # else:
        ######################################

        G = int(np.ceil( len(S) / k))
        random.shuffle(S)
        G_batches = Partition(S,k)
        R_l = []                                    #FIX OF ISSUE (a)
        if len(G_batches[-1]) < k:
            R_l = G_batches[-1]
            G = G-1       
    assert len(S) == 1, "An error occured: len(S)==1 is violated in Divide_and_Battle"
    return(S[0])


###############################################################################
#      Algorithm: PAC_Best_Item
#     Alg. 5 on p.14 in [Saha2020]
# 
#   MAJOR ISSUES: 
#   (I) In line 11 (of Alg. 5 in [Saha2020]), Score-Estimate(G_g,b,delta_l) can 
#       not be executed properly:
#       It is required that |G_g| = k-1 and also b \not\in G_g are fulfilled.
#   -> Fix: To calculate the score estimate, let G_fixed = G_g \setminus \{b\}. 
#            In case |G_fixed|=k, remove one random element from G_fixed.
#            Then, calculate Score-estimate(G_fixed,b,delta_l).
#            
#   REMARKS:
#   (i) We only implemented the algorithm for the case of top-1-ranking feedback 
#       (i.e., winner feedback); note that this is sufficient for implementing 
#       PAC_Wrapper.
#   (ii) For the case of top-1-ranking (which is the only one we consider here)
#         RANK-BREAKING in l.16 is easy: If i wins, then the win-count w_{i,j}'s
#         are simply updated via "w_{i,j} = w_{i,j} + 1 for all j!=i".
#         Consequently, w_{i,j} in l.18 is just the number w[i] of times $i$ has won
#         any multi-duel, and we have in l.18 
#           \hat{p}_{i,j} = w_{i,j} / (w_{i,j} + w_{j,i})  = w[i] / (w[i] + w[j])
#   (iii) L.19 of Alg.5 in [Saha2020] says to choose (if existent) SOME i \in 
#       \mathcal{G}_{g} with \hat{p}_{i,j} + \epsilon_{l} /2 >= 0.5, it is not
#       specified how to proceed if there are multiple ones. Here, we choose
#       such i uniformly at random from the set of all possible i's
#   (iv) Before using Partition on a set A, we randomly shuffle the elements in
#        it.       
###############################################################################
def PAC_Best_Item(A, eps, delta, P):
    """
Cf. Algorithm 5 on p.22 in [Saha2020]. Here we consider top-1 feedback only, i.e. "m=1"

A = [0,1,2,3,4,5,6,8]
k, eps, delta = 3, 0.3, 0.2
P = pm.PLPM([0.3,0.8,0.9,0.3,0.2,0.99,0.8,0.2,0.3],k)
P.show()
# print(PAC_Best_Item(A,eps,delta,P))
print("Outcomes of 10 runs of PAC_Best_Item")
for run in range(0,10):
    print(PAC_Best_Item(A,eps,delta,P),end=", ")
    """
    k = P.k
    # At first, obtain a (1/2,delta)-optmal item b from Divide_and_Battle, cf. the
    # last paragraph on p.21 of [Saha2020]
    b = Divide_and_Battle(A,0.5,delta,P)  
    
    S = A
    eps_l = eps / 8                     # \epsilon_{l}
    delta_l = delta / 2                 # \delta_{l}
    G = int(np.ceil( len(A) / k))       # G
    random.shuffle(S)
    G_batches = Partition(S,k)          # \mathcal{G}_{1}, ..., \mathcal{G}_{G}
    R_l = []                            # \mathcal{R}_{l}
    if len(G_batches[-1]) < k:
        R_l = G_batches[-1]
        G = G - 1
    while True:
        delta_l = delta_l / 2   
        eps_l = 3 * eps_l / 4   
        for g in range(0,G):
            
            ###################################
            # FIX OF ISSUE (I)
            G_g_fixed = G_batches[g].copy()
            # print("BEFORE__",G_g_fixed)   # For Debugging
            G_g_fixed = list(set(G_g_fixed)-set([b]))
            if len(G_g_fixed) == k:
                x = random.sample(G_g_fixed,1)[0]
                G_g_fixed = list(set(G_g_fixed)-set([x]))
            # print("AFTER__",G_g_fixed)    # For Debugging
            Theta_hat = Score_Estimate(G_g_fixed,b,delta_l,P)
            ###################################
            
            t = int(np.ceil( 16 * Theta_hat / ( 1 * (eps_l ** 2) ) * np.log( 2*k / delta_l ) ))
            
            # NOTE: Rankbreaking is easier for top1-Ranking, cf. REMARK (ii)
            Query, w = P.pull_arm(S=G_batches[g], size=t)       # Note: Query = sorted(G_batches[g])            
            hat_P = np.zeros((len(w),len(w)))    
            for i in range(0,len(w)):
                for j in range(0,len(w)):
                    if i==j:
                        hat_P[i][j] = 0.5
                    else:
                        hat_P[i][j] = w[i] / (w[i] + w[j])
            
            # In the following, we choose c_g  at random from the set of 
            # all possible c_gs, cf. REMARK (iii)
            possible_cgs = []
            for i in range(0,len(w)):
                buf = True
                for j in range(0,len(w)):
                    if hat_P[i][j] + eps_l / 2 <0.5:
                        buf = False
                if buf is True:         # that element at position i can be chosen as c_g
                   possible_cgs = possible_cgs + [Query[i]]
            if len(possible_cgs) == 0:
                c_g = random.sample(G_batches[g],1)[0]
            else:
                c_g = random.sample(possible_cgs,1)[0]
            S = list((set(S) - set(G_batches[g])).union(set([c_g])))
        
        
        S = list(set(S).union(set(R_l)))
        if len(S) == 1:
            break           #Break out of While Loop.
        elif len(S) <= k:
            buf = list( set(A) - set(S) )
            S = S + list(random.sample(buf,k-len(S)))
            eps_l = 2*eps / 3
            delta_l = delta
        
        G = int(np.ceil( len(S) / k))
        random.shuffle(S)
        G_batches = Partition(S,k)
        R_l = []
        if len(G_batches[-1]) < k:
            R_l = G_batches[-1]
            G = G-1

    assert len(S) == 1, "An error occured: len(S)==1 is violated in PAC_Best_Item"
    return(S[0])



###############################################################################
#      Algorithm: PAC_Wrapper
#     Alg. 1 on p.11 in [Saha2020]
#
#   MAJOR ISSUES: 
#   (I) In line 15 of Alg. 1 in [Saha2020], \mathcal{A}_{s} is possibly increased
#           but never reduced. This way, the while-loop can never be terminated!           
#   -> FIX: In the text, the the authors write on p.5 (before "(1)") that each
#           element $i\in \mathcal{B}$ with $\hat{p}_{i,b_{s}} < 0.5-\epsilon_{s}$
#           is discarded from $\mathcal{B}$ and this is also what is done in l.29.
#           Hence, we also proceed this way here.
#   (II) In line 24 (of Alg. 1 in [Saha2020], Score-Estimate(A\setminus\{b_s\},b_s,delta_l) can 
#       not be executed properly:
#       It is required that |A\setminus \{b_s\}| = k-1 is fulfilled, but it is 
#       possible (and likely) that |A|<k-1 holds after some iterations of the loop.
#   -> Fix: Calculate Score-Estimate(B\setminus \{b_s}\}, b_s, delta_l) instead.
#   (III) In l.29 of Alg. 1 in [Saha2020], it seems that only SOME i with 
#       \hat{p}_{i,b_s} <0.5-\epsilon_s is removed from A. However, in the text
#       on p.4 the authors write that EACH such i is removed from A.
#   -> Fix: We remove EVERY such i from A (= \mathcal{A}_s) AS LONG AS there is 
#           one element left. (Of course, A shouldn't be the empty set!)
#           Note here that this fix only DECREASES the runtime of PAC-Wrapper!
#   
#   MINOR ISSUES:
#   (a) In l.23ff. the authors write $m$, which is not introduced formerly.
#       As we are dealing with "top-m-ranking feedback" with "m=1", we replace
#       m by 1 everywhere.
#
#   REMARKS:
#   (i) For the case of top-1-ranking (which is the only one we consider here)
#         RANK-BREAKING in l.13 and 27 is easy: If i wins, then the win-count w_{i,j}'s
#         are simply updated via "w_{i,j} = w_{i,j} + 1 for all j!=i".
#         Consequently, w_{i,j} in l.18 is just the number w[i] of times $i$ has won
#         any multi-duel (during the run of the algorithm, WITHOUT counting those
#         from generated in calls of Score_Estimate and PAC_Best_Item, cf. l.13
#         and 27), and we have in l.14 and 28
#           \hat{p}_{i,j} = w_{i,j} / (w_{i,j} + w_{j,i})  = w[i] / (w[i] + w[j]) 
#   (ii) Before using Partition on a set A, we randomly shuffle the elements in
#        it.
###############################################################################
def PAC_Wrapper(delta, P, return_nr_arm_plays = False):
    """
    
EXAMPLE
theta = np.array(list([43,76,68,96,33,87,57,79,21]))
k, gamma = 4, 0.1
P = pm.PLPM(theta,k)
result = PAC_Wrapper(gamma,P)
print("Result:",result,"Time:",P.get_time())
    """
    A = list(range(0,P.m))      # \mathcal{A}_{s}     ("s" omitted)
    m = P.m
    k = P.k                     
    s = 1                   
    w = np.zeros(m)                 # w[i]: number of wins throughout the run of the algorithm
                                    #       (where those required for calls for Score_Estimate or 
                                    #        PAC_Best_Item are NOT counted)
                                    # NOTE that w[i] is set to 0 again after 
                                    # termination of the 1st while loop.
    if return_nr_arm_plays is True:
        nr_arm_plays = np.zeros(m)
    while len(A) >= k:
        eps_s = 2 ** (-s-2)                                     #\epsilon_{s}
        delta_s = delta / (120 * (s ** 3))                      #\delta_{s}
        R_s = []                                                #\mathcal{R}_{s}
        # print("Time passed so far:",P.get_time())                     # For Debugging
        # print("Calling PAC_Best_item with params",A,eps_s,delta_s)    # For Debugging
        # buf__ = P.get_time()                                          # For Debugging
        b_s = PAC_Best_Item(A, eps_s, delta_s, P)
        # print(".Time spent for PAC_Best_Item",P.get_time() - buf__ )  # For Debugging
        
        A_without_bs = list(set(A) - set([b_s]))    # \mathcal{A}_{s} \setminus \{b_{s} \}
        random.shuffle(A_without_bs)                 
        B_batches = Partition(A_without_bs, k-1)    # \mathcal{B}_{1},...,\mathcal{B}_{B_s}
        B_s = len(B_batches)                        # B_{s}
        if len(B_batches[-1])<k-1:
            R_s = B_batches[-1]
            B_s = B_s -1
        
        for b in range(0,B_s):
            hat_Theta_S = Score_Estimate(B_batches[b], b_s, delta_s, P)     # \hat{\Theta}_{S}
            hat_Theta_S = max(2*hat_Theta_S +1 ,2)                          # \hat{\Theta}_{S}
            B_b = list(set(B_batches[b]).union(set([b_s]))).copy()          # \mathcal{B}_{b}
            t_s = int(np.ceil(2 * hat_Theta_S / (eps_s ** 2) * np.log(k / delta_s)))    # t_S
            Query, w_new = P.pull_arm(B_b,t_s)                                  # Note: Query = sorted(B_b)
            
            if return_nr_arm_plays is True:
                for i in range(0,k):
                    nr_arm_plays[Query[i]] += t_s
            
            # Update the win statistics w 
            for i in range(0,len(Query)):
                w[Query[i]] += w_new[i]
            
            # For top-1-ranking, Rank-Breaking is easier, cf. REMARK (i)
            hat_P = np.zeros((k,k))         # Note: k = len(Query) = len(w_new)
            for i in range(0,k):
               for j in range(0,k):
                   if i==j:
                       hat_P[i][j] = 0.5
                   else:
                       hat_P[i][j] = w[Query[i]] / (w[Query[i]] + w[Query[j]])
                       
            b_s_pos = np.where(np.array(Query) == b_s)[0][0]
            
            ###################################
            # FIX OF ISSUE (I)
            for i in range(0,len(w_new)):
                if hat_P[i][b_s_pos] < 0.5-eps_s:
                    A = list( set(A) - set([Query[i]]) )
            ###################################
                    
        A = list(set(A).union(set(R_s)))
        s += 1

    # END OF THE WHILE LOOP
    # print("1st WHILE loop of PAC_Wrapper terminated,","A:",A)   # For Debugging
    
    buf = list(set(range(0,P.m)) - set(A))
    B = sorted( list(set(A).union( set(random.sample(buf,k-len(A))) )) )    # \mathcal{B}
    w = np.zeros(m)
    while len(A) > 1:
        # print("Time passed so far:",P.get_time(), A)                     # For Debugging
        eps_s = 2 ** (-s-2)                                         # \epsilon_{s}
        delta_s = delta / (80 * (s**3))                             # \delta_{s}
        # buf__ = P.get_time()                                       # For Debugging
        b_s = PAC_Best_Item(B, eps_s, delta_s, P)
        # print(".Time spent for PAC_Best_Item",P.get_time() - buf__ )   # For Debugging
        
        ###################################
        # FIX OF ISSUE (II)
        hat_Theta_S = Score_Estimate(list(set(B) - set([b_s])), b_s, delta_s, P)
        hat_Theta_S = max(2*hat_Theta_S + 1, 2)
        ###################################
        
        t_s = int(np.ceil( 2 * hat_Theta_S / (1 * (eps_s ** 2)) * np.log( k / delta_s ) ))  # t_s
        Query, w_new = P.pull_arm(B,t_s)                    # Note: Query = sorted(B)
        
        if return_nr_arm_plays is True:
                for i in range(0,k):
                    nr_arm_plays[Query[i]] += t_s
        
        # Update the win statistics w 
        for i in range(0,k):                # Note: k = len(Query) = len(w_new)
            w[Query[i]] += w_new[i]
        
        # For top-1-ranking, Rank-Breaking is easier, cf. REMARK (i)
        hat_P = np.zeros((k,k))
        for i in range(0,k):
           for j in range(0,k):
               if i==j:
                   hat_P[i][j] = 0.5
               else:
                   hat_P[i][j] = w[Query[i]] / (w[Query[i]] + w[Query[j]])
        # print("B",B,"w",w,"hat_P")
        # print(hat_P)
                      
        b_s_pos = np.where(np.array(B)==b_s)[0][0] 
        
        ###################################
        # FIX OF ISSUE (III)
        for r in range(0,len(B)):
            if B[r] in A and hat_P[r][b_s_pos]  < 0.5 -eps_s and len(A) > 1:
                A = list(set(A) - set([B[r]]))
        ###################################
        s += 1
        
    # print(".....Number of Arm Plays",w)         # For Debugging
    assert len(A)==1, "Oops, an error occured."
    if return_nr_arm_plays is True:
        return(A[0],nr_arm_plays)
    return(A[0])
