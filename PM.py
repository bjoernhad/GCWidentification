#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains a class PM that is used to represent the environment of a 
multi-dueling bandit.
It also provides functions to sample (uniformly at random) from 
PM_{k}^{m}(X) for several assumptions (X).
"""

import itertools as it
import numpy as np
import math

def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)    

class PM:
    """ 
    
EXAMPLE
P = PM(m=5,k=4)
for S in it.combinations(range(0,5),4):
    p = np.random.choice(range(20,100),4)
    p = p / sum(p)
    print(p)
    P.set_arm((S,p))
P.set_arm((S,[0.2,0.2,0.25,0.35]))
result = P.pull_arm((0,2,3,1),250)
print("S:",result[0],"Observations from the first 250 pulls:",result[1])
result = P.pull_arm((0,2,3,1),100)
print("S:",result[0],"Observations from the next 100 pulls:",result[1])
P.show()
    """
    def __init__(self,m=2,k=2):
        """
        Creates an EMPTY environment for k-subsets of [m]. Each arm has to be
        added via set_arm afterwards.
        """
        self.m = m
        self.k = k
        self.arms = dict()
        self.time = 0
        self.nr_arm_plays = np.zeros(self.m)
    
    def set_arm(self,new_arm):
        """
Defines new arm, may override an existing one.
Here, new_arm has to be a tuple (S,p), where S is a tuple (k-subset of [m]) and
p is the param. of a categorical prob. distribution with values in S.
        
EXAMPLE:
P = PM(m=10,k=4)
S = [3,2,0,1]
p = [0.45,0.45,0.05,0.05]
P.set_arm((S,p))
P.show()
        """
        assert type(new_arm) is tuple and len(new_arm) == 2, "new_arm has to be of the form (S,p), where S and p are tuples"
        assert len(new_arm[0]) == len(new_arm[1]), "S and p have to be of the same length"
        order = np.argsort(new_arm[0])
        S = tuple( np.array( new_arm[0] )[ order ] )
        p = tuple( np.array( new_arm[1] )[ order ] )
        assert len(S)==self.k and len(p) == self.k, "S and p have to be of length k"
        assert abs(sum(p)-1)<0.1**6, "p has to be param of a cat. prob. distribution, i.e. it has to sum up to 1"
        for i in S:
            assert i in range(0,self.m), "S has to be an element of [0,...,m-1]"
        initial_pulls = np.zeros(self.k,dtype=int)
        self.arms[S] = [p,initial_pulls]
        
        
    def _get_arm_prob(self,S):
        S=tuple(sorted(S))
        assert S in self.arms.keys(), "S="+str(S)+" is not an arm"
        return(self.arms[S][0])
        
    def show(self):
        """
Provides an overview over the current environment.
        """
        print("Arm(Set)  | Winning Probabilities  | #Observations  | h(arm)")
        for S in self.arms.keys():
            print(S," | ", self.arms[S][0], " | ",self.arms[S][1], " | ", self._get_arm_h(S))
        print("Completely filled? : ",self.is_filled())
        print("h (= minimum {h(S) | S\in [m]_k)}):",self._get_h())
        print("h2 (= mininum {h(S) | S\in [m]_{k} and GCW(P) in S}):",self._get_h2())
        print("GCW: ",self._get_GCW())
        print("Nr of arm pulls (per arm):",self.get_nr_arm_plays())
        print("Time: ",self.get_time())
            
    def pull_arm(self,S,size=1):
        """ 
Pulls the arm corresponding to the set S for 'size' times.
Here, S has to be a list (or tuple or np.array) which already has an arm in the
environment. (The order of S is not relevant)

EXAMPLE
P = PM(m=10,k=4)
S = [3,2,0,1]
p = [0.45,0.45,0.05,0.05]
P.set_arm((S,p))
result = P.pull_arm((0,2,3,1),250)
print("S:",result[0],"Observations from the first 250 pulls:",result[1])
result = P.pull_arm((0,2,3,1),100)
print("S:",result[0],"Observations from the next 100 pulls:",result[1])
P.show()
        """
        assert self.is_filled() is True, "The PM is not completely initialized yet."
        assert type(size) is int, "size has to be of type 'int'"
        max_pull_batch_size = 10**7
        S=tuple(sorted(S))
        assert S in self.arms.keys(), "S="+str(S)+" is not an arm"
        p  = self.arms[S][0]
        for i in range(0,self.k):
            self.nr_arm_plays[S[i]] += size
            
        new_observations = np.zeros(self.k,dtype = int)
        while size > 0:
            # print("B",end='')                               # For Debugging
            batch_size = min(max_pull_batch_size, size)
            size -= batch_size
            outcomes = np.random.choice(range(0,self.k),batch_size,p=p)
            batch_observations = np.zeros(self.k,dtype = int)
            for outcome in outcomes:
                new_observations[outcome] += 1
                self.arms[S][1][outcome] += 1
            self.time += batch_size
            new_observations += batch_observations
        return(S,list(new_observations))


    def is_filled(self):
        """
Returns True if each k-subset of [m] is contained as an arm in the environment,
otherwise it returns False.
        """
        return(len(self.arms) == binom(self.m,self.k))
    
    def get_nr_arm_plays(self):
        return(self.nr_arm_plays)
    
    def _get_arm_mode(self,S):
        """ 
Returns a list containing the mode(s) of the arm corresponding S.
Here, S has to be a k-subset of [m] and of type list/tuple/... (the order is not
important) 

For an illustration, see 'show()'
        """
        S=tuple(sorted(S))
        assert S in self.arms.keys(), "S is not an arm"
        buf = np.array(self.arms[S][0])
        pos = np.where(buf == max(buf))
        result = list( np.array(S)[pos])
        return(result)
    
    def _get_arm_h(self,S):
        """
Returns for arm S the minimal gap from the winning prob. of the mode of S to
that of any other elem. of S.

For an illustration, see 'show()'.
        """
        S=tuple(sorted(S))
        assert S in self.arms.keys(), "S is not an arm"
        p = np.array(self._get_arm_prob(S))
        pos = np.argmax(p)
        return(p[pos] - np.max( np.concatenate((p[0:pos],p[(pos+1):])) ))   # TO DO: CHECK IF THIS IS ALSO FALSE ABOVE!

    
    def _get_h(self):
        """
Returns the minimum over the h's of all arms.

For an illustration, see 'show()'.
        """
        result = 2
        for S in self.arms.keys():
            pos = np.argmax(self.arms[S][0])
            p = self.arms[S][0]
            result = min( result, p[pos] - np.max(p[0:pos]+p[pos+1:]) )
        return(result)
    
    def _get_h2(self):
        """ 
Returns -np.inf if no GCW exists.
        """
        assert self.is_filled() is True, "The PM is not yet initialized compleitely."
        GCWs = self._get_GCW()
        if GCWs is False:
            return(-np.inf)
        h2 = 2.0
        GCW = GCWs[0]                            
        buf = list(set(range(0,self.m))-set([GCW]))
        for S in it.combinations(buf,self.k-1):
            h2 = min( h2, self._get_arm_h( list(S)+list([GCW])) )
        return(h2)

    
    def _get_sample_mode(self,S):
        """
Returns a list containing the mode(s) of the distribution corresponding to arm
S.

For an illustration, see 'show()'.
        """
        S=tuple(sorted(S))
        assert S in self.arms.keys(), "S is not an arm"
        buf = np.array(self.arms[S][1])
        pos = np.where(buf == max(buf))
        result = list( np.array(S)[pos])
        return(result)
    
    def reset_observations(self):
        """
Resets all observations/pulls of all arms of the environment.
        """
        for S in self.arms.keys():
            self.arms[S][1] = np.zeros(self.k,dtype=int)
        self.time = 0
        self.nr_arm_plays = np.zeros(self.m)
        return(True)
    
    def _get_GCW(self):
        """
Returns False if no GCW exists or the envirnoment is not completely filled (cf. is_filled()).
Otherwise, it returns a list containing all GCWs.

For an illustration, see 'show()'.
        """
        if not self.is_filled():
            return(False)
        candidates = list(range(0,self.m))
        for S in self.arms.keys():
            mode_of_S = self._get_arm_mode(S)
            for i in S:
                if i in candidates and i not in mode_of_S:
                    candidates.remove(i)
                    # print("By observations on ",S,"we remove",i)
            if len(candidates)==0:
                return(False) 
        return(candidates)
    
    def get_time(self):
        """
Returns the time, i.e. the total number of pulls of any arm made so far.

For an illustration, see 'show()'.
        """
        return(self.time)
        # result = 0 
        # for S in self.arms.keys():
        #     result += sum(self.arms[S][1])
        # return(result)
    
    def get_size(self):
        return((self.m,self.k))
    
    
########################################################################
# 	Class PLPM
########################################################################
class PLPM():
    """
    
EXAMPLE
theta = list([43,76,68,44,97,87,57,79,21])
P = PLPM(theta,3)
P.show()
S=list([1,3,6])
print("S:",S,"Prob.:",P._get_arm_prob(S),"h(S):",P._get_arm_h(S))
print("Result of 10000 pulls of S",P.pull_arm(S,size=10000))
print("GCW:",P._get_GCW())


To check whether PLPM is correctly initialized in the sense that every set S 
can be pulled without raising an "ValueError: probabilities do not sum to 1" error,
one may execute the following:
for k in range(3,10):
    for S in it.combinations(range(0,10),k):
        P.pull_arm(S,1)
    print("k",k," ok.")  
    """
    def __init__(self,theta,k):
        self.theta = list( np.array(theta) / max(theta))
        self.m = len(self.theta) 
        self.k = k
        self.time = 0
        self.nr_arm_plays = np.zeros(self.m)    # self.nr_arm_plays[i] is the 
                                                # nr of times i has been in a
                                                # query set so far.

    
    def _get_arm_prob(self,S,rounding_precision=8):     #Rounding prec. not needed
        S=tuple(sorted(S))
        p = np.zeros(self.k)
        theta = np.array(self.theta)
        for l in range(0,self.k):
            # p[l] = round( theta[S[l]] / sum(theta[list(S)]), rounding_precision)  #OLD VERSION
            p[l] = theta[S[l]] / sum(theta[list(S)])
        return(p)
        
    ####################################################################
    # Private Methods 
    ####################################################################
    def _get_arm_h(self,S):
        """
Returns for arm S the minimal gap from the winning prob. of the mode of S to
that of any other elem. of S.

For an illustration, see 'show()'.
        """
        S=tuple(sorted(S))
        assert len(S) == self.k, "len(S) has to be k"
        p = np.array(self._get_arm_prob(S))
        pos = np.argmax(p)
        return(p[pos] - np.max( np.concatenate((p[0:pos],p[(pos+1):])) ))   # TO DO: CHECK IF THIS IS ALSO FALSE ABOVE!

    def _get_h(self):
        h = 2.0
        for S in it.combinations(range(0,self.m),self.k):
            h = min( h, self._get_arm_h(S))
        return(h)
    
    def _get_h2(self):
        h2 = 2.0
        GCW = self._get_GCW()[0]                            #NOTE: Due to the PL-assumption, a GCW exists.
        buf = list(set(range(0,self.m))-set([GCW]))
        for S in it.combinations(buf,self.k-1):
            h2 = min( h2, self._get_arm_h( list(S)+list([GCW])) )
        return(h2)


    def _get_GCW(self):
        """
Returns a list containing all GCWs.
NOTE: Due to the PL-assumption, at least one GCW exists.
        """
        theta = np.array(self.theta)
        return(np.where(theta == np.max(theta))[0])
    
    def pull_arm(self,S,size=1):    
        """ 
This is a batch-version, standard batch size is 10**7.        
        """
        assert type(size) is int, "size has to be of type 'int'"
        assert len(S) == self.k, "len(S) has to be k"
        max_pull_batch_size = 10**7
        S=tuple(sorted(S))
        p = self._get_arm_prob(S)
        for i in range(0,self.k):
            self.nr_arm_plays[S[i]] += size

        new_observations = np.zeros(self.k,dtype = int)
        while size > 0:
            # print("B",end='')                               # For Debugging
            batch_size = min(max_pull_batch_size, size)
            size -= batch_size
            outcomes = np.random.choice(range(0,self.k),batch_size,p=p)
            batch_observations = np.zeros(self.k,dtype = int)
            for i in outcomes:
                batch_observations[i] += 1
            self.time += batch_size
            new_observations += batch_observations
        return(S,list(new_observations))
    
    def get_size(self):
        return((self.m,self.k))
    
    def get_time(self):
        return(self.time)
    
    def show(self):
        """
Provides an overview over the current environment.
        """
        print("Arm(Set)  | Winning Probabilities | h(arm)")
        h = 2
        for S in it.combinations(range(0,self.m),self.k):
            h_S = self._get_arm_h(S)
            h = min( h, h_S)
            print(sorted(S),"\t |",self._get_arm_prob(S),"\t |",h_S)
        print("theta:",self.theta)
        print("h (= minimum {h(S) | S\in [m]_k)}):",h)
        print("h2 (= mininum {h(S) | S\in [m]_{k} and GCW(P) in S}):",self._get_h2())
        print("GCW:",self._get_GCW())
        print("Nr of arm pulls (per arm):",self.get_nr_arm_plays())
        print("Time:",self.get_time())

    def reset_observations(self):
        """
Resets all observations/pulls of all arms of the environment.
        """
        self.time = 0
        self.nr_arm_plays = np.zeros(self.m)
        return(True)

    def get_nr_arm_plays(self):
        """
EXAMPLE
theta, k = [0.5,0.2,0.3,0.1], 3
P = PLPM(theta,k)
P.pull_arm([0,1,2],100)
P.pull_arm([1,2,3],50)
print("Number of arm pulls (per arm):",P.get_nr_arm_plays())
        """
        return(self.nr_arm_plays)
    


###############################################################################
#   SAMPLING PROCEDURES
###############################################################################
#   Below, we implement the following sampling procedures.
#
#   FUNCTION                |       SAMPLES FROM 
# ----------------------------------------------------------------------------
#   sample_p                |       \Delta_{k}  
#   sample_p_h              |       \Delta_{k}^{h}
#   sample_PM               |       PM_{k}^{m}
#   sample_PM_h             |       PM_{k}^{m}(\Delta^{h})
#   sample_GCW_PM           |       PM_{k}^{m}(\exists GCW)
#   sample_hGCW_PM          |       PM_{k}^{m}(\exists hGCW)
#   sample_hGCW_PM_h        |       P_m^k(GCW \and \Delta^{h})
#   sample_hGCW_PM_0        |       P_m^k(GCW \and \Delta^{0})
#   sample_hGCW_PM_h2        |       P_m^k(GCW \and \Delta^{h2})
###############################################################################
def sample_p(k):
    """
Samples uniformly at random an element p \in \Delta_{k}, i.e.
p is an np.array of length k with non-negative entries and sum(p) = 1 (at least approx.)
    """
    p = np.zeros(k)
    for i in range(0,k):
        p[i] = np.random.uniform()
    return(p / sum(p))

def sample_p_h(k,h):
    """
    Samples uniformly at random an element p \in \Delta_{k}^{h}
    """
    assert 0<h and h<1, "h has to be a number between 0 and 1"
    p = np.array(sample_p(k))
    max_pos = np.argmax(p)    # Next, increase p[max_pos] such that the gap to 
                              # any other element is at least h.
    p_prime = np.zeros(k)
    p_prime[max_pos] = 1
    return((1-h)*p + h*p_prime)    

def sample_PM(m,k):
    """ 
Samples uniformly at random an element P from P_m^k 

EXAMPLE
P = sample_PM(6,5)
P.show()
    """
    P = PM(m,k)
    for S in it.combinations(range(0,m),k):
        P.set_arm((S,sample_p(k)))
    return(P)


def sample_PM_h(m,k,h):
    """ 
Samples uniformly at random an element P from P_m^k(\Delta^{h}) 

EXAMPLE
P = sample_PM_h(6,5,0.2)
P.show()
    """
    P = PM(m,k)
    for S in it.combinations(range(0,m),k):
        P.set_arm((S,sample_p_h(k,h)))
    return(P)

def sample_GCW_PM(m,k):
    """ 
Samples uniformly at random an element P from P_m^k(GCW)    

EXAMPLE
P = sample_GCW_PM(6,5)
P.show()
    """
    P = PM(m,k)
    gcw = np.random.choice(range(0,m),size=1)[0]
    for S in it.combinations(range(0,m),k):
        p = sample_p(k)
        # print("Before",S,p,gcw)                   # For Debugging
        if gcw in S:
            gcw_pos = np.where(np.array(S)==gcw)[0][0]
            max_pos = np.argmax(p)
            buf = p[gcw_pos]
            p[gcw_pos] = p[max_pos]
            p[max_pos] = buf
        # print("After",S,p)                        # For Debugging
        P.set_arm((S,p))
    return(P)

def sample_hGCW_PM(m,k,h):
    """ 
Samples P uniformly at random from P_m^k(\exists hGCW)

EXAMPLE
P = sample_hGCW_PM(6,5,0.18)
P.show()
    """
    P = PM(m,k)
    gcw = np.random.choice(range(0,m),size=1)[0]
    for S in it.combinations(range(0,m),k):
        if gcw not in S:
            p = sample_p(k)
        # print("Before",S,p,gcw)                   # For Debugging
        else:
            p = sample_p_h(k,h)
            gcw_pos = np.where(np.array(S)==gcw)[0][0]
            max_pos = np.argmax(p)
            buf = p[gcw_pos]
            p[gcw_pos] = p[max_pos]
            p[max_pos] = buf
            # print(p,sum(p))
        # print("After",S,p)                        # For Debugging
        P.set_arm((S,p))
    return(P)

def sample_hGCW_PM_h(m,k,h):
    """ 
Samples uniformly at random an element P from P_m^k(GCW \and \Delta^{h})    

EXAMPLE
P = sample_hGCW_PM_h(6,4,0.15)
P.show()
    """
    P = PM(m,k)
    gcw = np.random.choice(range(0,m),size=1)[0]
    for S in it.combinations(range(0,m),k):
        p = sample_p_h(k,h)
        if gcw in S:
            gcw_pos = np.where(np.array(S)==gcw)[0][0]
            max_pos = np.argmax(p)
            buf = p[gcw_pos]
            p[gcw_pos] = p[max_pos]
            p[max_pos] = buf
        P.set_arm((S,p))
    return(P)

def sample_hGCW_PM_0(m,k,h):
    """ 
Samples P uniformly at random from P_m^k(hGCW and \Delta^{0})

EXAMPLE
P = sample_hGCW_PM_0(6,5,0.18)
P.show()   
    """
    P = sample_hGCW_PM(m,k,h)
    while P._get_h() == 0:
        P = sample_hGCW_PM(m,k,h)
    return(P)


def sample_hGCW_PM_h2(m,k,h,h2):
    """ 
Samples uniformly at random an element P from P_m^k(hGCW \and \Delta^{h2})    

EXAMPLE
P = sample_hGCW_PM_h2(6,4,0.6,0.1)
P.show()
    """
    h = max(h,h2)
    P = PM(m,k)
    gcw = np.random.choice(range(0,m),size=1)[0]
    for S in it.combinations(range(0,m),k):
        if gcw in S:
            p = sample_p_h(k,h)
            gcw_pos = np.where(np.array(S)==gcw)[0][0]
            max_pos = np.argmax(p)
            buf = p[gcw_pos]
            p[gcw_pos] = p[max_pos]
            p[max_pos] = buf
        else:
            p = sample_p_h(k,h2)
        P.set_arm((S,p))
    return(P)
