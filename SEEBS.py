#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains an implementation SEEBS (as well as several auxiliary functions)
from  
[Ren2020]: Wenbo Ren, Jia Liu, Ness B. Shroff (2020). 
    The Sample Complexity of Best-$k$ Items Selection from Pairwise Comparisons. 
    In Proceedings of the International Conference on Machine Learning.
"""
import numpy as np
import PM as pm


def DI(P, i, v, eps, s_u, s_d, gamma, S_up, S_mid, S_down):     # P: Environment (PM)
    """
This is Alg. 1 ("DistributeItem") in [Ren2020].
The additional parameter P describes the environment/bandits and has to be of type
pm.PM or pm.PLPM .

IMPORTANT NOTE: There is a bug in [*], see below.

[*]: https://github.com/WenboRen/Topk-Ranking-from-Pairwise-Comparisons, accessed 16th May 2021
[Ren2020]: Wenbo Ren, Jia Liu, Ness B. Shroff (2020). 
    The Sample Complexity of Best-$k$ Items Selection from Pairwise Comparisons. 
    In Proceedings of the International Conference on Machine Learning.
    """
    t_max = int( np.ceil( ( 2/(eps**2)) * np.log( 4/gamma) ) )
    t = 0.0
    w_t = 0.0
    while t < t_max:
        t = t+1
        # print("(i,v)=",i,v)
        S, wins = P.pull_arm([i,v],size = 1)
        if (i==S[0] and wins[0]==1) or (i==S[1] and wins[1]==1):    # i has won.
            w_t = w_t + 1
        b_t = np.sqrt( 0.5 / t * np.log( (np.pi**2) * (t**2) / (3*gamma) ) )
        # print(w_t/t, b_t)
        if w_t / t - b_t > 0.5 + s_u:
            S_up = set(S_up).union(set([i]))
            return([S_up, S_mid, S_down])
        if w_t / t + b_t < 0.5 - s_d:
            S_down = set(S_down).union(set([i]))
            return([S_up, S_mid, S_down])
    if w_t / t > 0.5 + 0.5*eps + s_u:
        S_up = set(S_up).union(set([i]))
    elif w_t / t < 0.5 - 0.5*eps - s_d:     # In [*, RankingAlgorithms.h, l.39], it is erranously "0.5+0.5*ep-s_d"!
                                            # According to [Ren2020], it should be "0.5-0.5*eps-s_d"!
        S_down = set(S_down).union(set([i]))
    else:
        S_mid = set(S_mid).union(set([i]))
    return([S_up, S_mid, S_down])
        

def EQS(P, S, k=1, eps = 0.05, gamma= 0.05):
    """
Cf. Alg. 2 in [Ren2020].
The additional parameter P describes the environment/bandits and has to be of type
pm.PM or pm.PLPM .

IMPORTANT NOTE: In l.3 of the pseudocode of Alg.2 in [Ren2020], it has to be
"and i!=v". This is consistent with the author's implementation from [*].

[*]: https://github.com/WenboRen/Topk-Ranking-from-Pairwise-Comparisons, accessed 16th May 2021
[Ren2020]: Wenbo Ren, Jia Liu, Ness B. Shroff (2020). 
    The Sample Complexity of Best-$k$ Items Selection from Pairwise Comparisons. 
    In Proceedings of the International Conference on Machine Learning.
    """
    # print("EQS called with S=",S)
    v = np.random.choice(list(S),1)[0]
    S_up, S_mid, S_down = list([]), list([v]), list([])
    gamma_1 = gamma / (len(S) * (len(S)-1))
    for i in set(S)-set([v]):
        # print("(i,v,S)",i,v,S)
        [S_up, S_mid, S_down] = DI(P, i, v, eps/2, 0, 0, gamma_1, S_up, S_mid, S_down)
    if len(S_up) > k:
        return( EQS(P, S_up, k, eps, (len(S)-1)*gamma / len(S) ) )
    elif len(S_up) + len(S_mid) >= k:
        buf = np.random.choice(list(S_mid), size=k-len(S_up), replace=False)
        return( list( set(S_up).union(set(buf)) ))
    else:
        k_prime = k - len(S_up) - len(S_mid)
        buf = EQS(P, S_down, k_prime, eps, (len(S)-1)*gamma / len(S)) 
        return( list( set(S_up).union( set(S_mid), set(buf)) ))

def Random_Partition(S,k):
    """
Help function for TKS below.

EXAMPLE
print(Random_Partition(range(0,22),5))
    """
    B = int(np.ceil(len(S) / k))
    batches = list()
    S=list(S)
    np.random.shuffle(S)
    for i in range(0,B):
        batches.append(list(S[i*k:(i+1)*k]))
    return(batches)


def TKS(P, S, k=1, eps=0.05, gamma=0.05):
    """ 
This is Alg. 3 in [Ren2020].
The additional parameter P describes the environment/bandits and has to be of type
pm.PM or pm.PLPM .

IMPORTANT NOTE: We modify l.7 of Alg.3 in [Ren2020] such that A_{t,i} is EQS(...)
if |S_{t,i}|>k and choose A_{t,i} = S_{t,i} in case |S_{t,i}| <= k. (Otherwise, 
EQS(...) may throw an exception!)  This fix is also incorporated in the author's
implementation [*].

    
[*]: https://github.com/WenboRen/Topk-Ranking-from-Pairwise-Comparisons, accessed 16th May 2021
[Ren2020]: Wenbo Ren, Jia Liu, Ness B. Shroff (2020). 
    The Sample Complexity of Best-$k$ Items Selection from Pairwise Comparisons. 
    In Proceedings of the International Conference on Machine Learning.
    """
    t = 0
    R = set(S)      #R_t
    while len(R) > k:
        t = t+1
        m_t = int( np.ceil( len(R) / (2*k) ) )
        eps_t = 0.25 * (0.8**t)
        gamma_t = 6*gamma / ((np.pi**2) * (t**2))
        S_t = Random_Partition(R, 2*k)      # S_t[i] is S_{t,i}
        A_t = list([])
        for i in range(0,m_t):
            ## MODIFICATION ###################################################
            # This is also done in l.95 of [*, RankingAlgorithms.h, l.95]
            ###################################################################
            if len(S_t[i]) <= k:
                A_t.append(S_t[i])
            else:
                A_t.append( EQS(P, S_t[i], min(k, len(S_t[i])), eps_t, gamma_t/k) )
            ################################################################### 
        R = list([])
        for i in range(0,m_t):
            R = R + A_t[i]
        R = set(R)
    assert len(R)==k, "Error in TKS, R="+str(R)
    return(list(R))

def SEEBS(P, S,gamma=0.05):
    """ 
This is Alg. 4 in [Ren2020].
The additional parameter P describes the environment/bandits and has to be of type
pm.PM or pm.PLPM. Here, S is a set of arms considered, i.e. a subset of range(0,P.m).

[*]: https://github.com/WenboRen/Topk-Ranking-from-Pairwise-Comparisons, accessed 16th May 2021
[Ren2020]: Wenbo Ren, Jia Liu, Ness B. Shroff (2020). 
    The Sample Complexity of Best-$k$ Items Selection from Pairwise Comparisons. 
    In Proceedings of the International Conference on Machine Learning.

EXAMPLE: 
P = pm.PLPM([1,0.8,0.6,0.4],k=2)
result = SEEBS(P,range(0,4),gamma=0.01)
print("SEEBS: Result="+str(result)+" t:"+str(P.get_time()))
    """
    t=1
    R = set(S)          # R_t
    
    while len(R) > 1:
        # print("R=",R," t:",P.get_time())
        alpha_t, gamma_t = 2**(-t), 6*gamma/((np.pi**2) * (t**2))       #alpha_t, delta_t
        v_t = TKS(P, R, 1, alpha_t/3 , 2*gamma_t /3)[0]
        S_up, S_mid, S_down = set([]), set([v_t]), set([])
        for i in R-set([v_t]):
            # print("(R,i)",R,i,v_t)
            [S_up, S_mid, S_down] = DI(P, i, v_t, alpha_t/3, 0, alpha_t/3, gamma_t/3, S_up, S_mid, S_down)
            # print(i,v_t,len(S_up),len(S_mid),len(S_down))
        R = R - S_down
            #REMARK: This is what is done in Alg.4 in [Ren2020]
            #        In  [*,l.125] the authors use R = set(S_up).union(set(S_mid)). but this does not make a difference.
        t = t+1
        # print("(eps_t,delta_t)",alpha_t,gamma_t)
        # print(len(S_up),len(S_mid),len(S_down))
    assert len(R)==1, "Error in SEEBS, R="+str(R)
    return(list(R)[0])



def sanity_check(ms = [10,20,30,40,50],nr_runs = 10, random_seed = 1,show_progress = False):
    """
This works as a sanity_check for our implementation of SEEBS.

To obtain comparing results from [*], do the following:
- [In Linux:] "#include<tchar.h>" in l.2
- Choose "string Alg = "SEEBS"" in [*,topk.cpp], i.e. uncomment l.22 and comment l.30 therein.
- Choose "string instance = "Homo"" in [*,topk.cpp], i.e., uncomment l.32 and comment l.35 therein.
- Comment l.216--218 in [*,topk.cpp]
- Choose "Ns = { 10,20,30,40,50 };" and "Ks = vector<int>(Ns.size(), 1);" in l.92--95 of [*,topk.cpp]
- Replace l.213 in [*,topk.cpp] by 
    "cout << "n: " << n << " k: " << k << " t: " << cumulative_complexity / repetition << endl;"
-- [In Linux] Compile via "g++ topk.cpp -o topk", execute via "./topk"
    
[*]: https://github.com/WenboRen/Topk-Ranking-from-Pairwise-Comparisons, accessed 16th May 2021
[Ren2020]: Wenbo Ren, Jia Liu, Ness B. Shroff (2020). 
    The Sample Complexity of Best-$k$ Items Selection from Pairwise Comparisons. 
    In Proceedings of the International Conference on Machine Learning.
    
EXAMPLE: 
sanity_check(show_progress=True)
    """
    print("Sanity Check on rel. P=(p_{i,j}) with p_{i,j} = 0.6 iff i<j, with gamma=0.01")
    print("(Called with ms=",ms,"nr_runs=",nr_runs,"random_seed=",random_seed,\
          "show_progress=",show_progress,")\n")
    print("WITHOUT the fix (!) in the implementation of DI (i.e., if we falsely write",\
          "\"0.5+0.5*ep-s_d\" as in [*] and not \"0.5-0.5*ep-s_d\" as in Alg.1",\
          " in [Ren2020]) in this python file, the following results should be",\
          "consistent with those shown in Fig. 1(f) of [Ren2020].\n")
    print("[*]: https://github.com/WenboRen/Topk-Ranking-from-Pairwise-Comparisons, accessed 16th May 2021")
    print("[Ren2020]: Wenbo Ren, Jia Liu, Ness B. Shroff (2020).",\
          "The Sample Complexity of Best-$k$ Items Selection from Pairwise Comparisons.",\
              "In Proceedings of the International Conference on Machine Learning.\n")
    np.random.seed(random_seed)
    for m in ms:
        P = pm.PM(m=m,k=2)
        for i in range(0,m):
            for j in range(i+1,m):
                P.set_arm(([i,j],[0.6,0.4]))
        times = list([])
        for run in range(0,nr_runs):
            P.reset_observations()
            # result = TKS(P, set(range(0,m)), 1, 0.001 , 0.01)[0]
            result = SEEBS(P,range(0,m),gamma=0.01)
            if show_progress is True: 
                print("m=",m,"| Run=", run, "|  Result:",result," t:",P.get_time())
            times.append(P.get_time())
        print("--> For m=",m,"the mean (and std-error of) termination time of SEEBS is",\
              np.mean(np.array(times)),"(",np.std(np.array(times))/np.sqrt(nr_runs),")")