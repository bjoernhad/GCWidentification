#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains functions to run the "experiments" from our paper. 
"""

import numpy as np
import PAC_Wrapper 
import GCW_algo 
import PM as pm
import random
import matplotlib.pyplot as plt
import SELECT_algorithm as selecta
import math
import SEEBS as seebs 
import ExploreThenVerify as explvrfy

from matplotlib.ticker import FormatStrFormatter

###############################################################################
#   EXPERIMENT_Dueling:
#
#   Compare our DKWT SELECT in the Dueling Bandits Case
###############################################################################    
def experiment_dueling(m,h,gamma, nr_runs = 100, random_seed = 1,\
                         filename="results_exp_dueling.txt",show_progress = True,\
                             algos = ["DKWT","SELECT","SEEBS","ExploreThenVerify"]):
    """
In this experiment, 'nr_runs' P are sampled from  PM_{2}^{m}(hGCW and \Delta^{h}).
Then, DKWT and SELECT are run on each of these instances once (with error prob.)
gamma, where SELECT also obtains the value of 'h' as parameter.
The mean and standard termination time as well as the observed accuracy (averaged)
over all runs) are saved (more precisely: appended) to the file 'filename'.
    
EXAMPLE
h_values = [0.2,0.1]
for i in range(0,len(h_values)):
    h = h_values[i]
    experiment_dueling(m=6, h=h, gamma=0.05, nr_runs = 10, random_seed = i,\
        algos = ["DKWT","SELECT","SEEBS"])
    """
    f = open(filename,"a")  
    f.write("\n \n")
    f.write(">> experiment_dueling(m="+str(m)+",h="+str(h)+",gamma="+str(gamma)+",nr_runs="+\
            str(nr_runs)+",random_seed="+str(random_seed)+",filename="+str(filename)+") \n")
    np.random.seed(random_seed)
    random.seed(random_seed)
    # print("Run \t | DKWT \t \t | SELECT")               # For Debugging
    times_DKWT = list()
    times_SELECT = list()
    times_SEEBS = list()
    times_ExplThenVrfy = list()
    accuracy_DKWT = 0.0
    accuracy_SELECT = 0.0
    accuracy_SEEBS = 0.0
    accuracy_ExplThenVrfy = 0.0
    if show_progress is True:
        print("\n Starting experiment_dueling with parameter (m,h,gamma)=(",m,h,gamma,")")
        print("Runs finished: (Total:",nr_runs,")")
    
    for run in range(0,nr_runs):
        if show_progress is True:
            print("Run",run,end=": ")
            
        P = pm.sample_hGCW_PM_h(m, 2, h)
        True_GCW = P._get_GCW()[0]
        
        # DKWT
        if "DKWT" in algos:
            P.reset_observations()    
            result_DKWT = GCW_algo.DKWT(gamma,P)
            times_DKWT.append(P.get_time())
            if result_DKWT == True_GCW:
                accuracy_DKWT +=1 
            print("DKWT done.",end=",")
        
        
        # SELECT 
        if "SELECT" in algos:
            P.reset_observations()
            epsilon = -np.log(gamma)/np.log(np.log2(m))
            m_h = math.floor((1+epsilon)*math.log(2)/2*math.log(math.log(m,2),2)/(h*h))+1
            result_SELECT, itera = selecta.SELECT(list(np.arange(m)), m_h, P)
            times_SELECT.append(P.get_time())
            if result_SELECT == True_GCW:
                accuracy_SELECT += 1
            print("SELECT done.",end=",")
        
        
        # SEEBS
        if "SEEBS" in algos:
            P.reset_observations()
            result_SEEBS = seebs.SEEBS(P,range(0,P.m),gamma=gamma)
            times_SEEBS.append(P.get_time())
            if result_SEEBS == True_GCW:
                accuracy_SEEBS +=1 
            print("SEEBS done.",end=",")
        
        # ExploreThenVerify
        if "ExploreThenVerify" in algos:
            P.reset_observations()
            result_ExplThenVrfy = explvrfy.ExploreThenVerify(P,gamma=gamma,show_progress = False)
            times_ExplThenVrfy.append(P.get_time())
            if result_ExplThenVrfy == True_GCW:
                accuracy_ExplThenVrfy +=1
            print("Explore-then-Verify done.", end=" ")
        
        print(" ")
        
        # print(run," | ", result_DKWT==True_GCW, "( t: ",time_DKWT,") \t | ",result_SELECT==True_GCW, "( t: ",time_SELECT,")")
                # For Debugging
    
    accuracy_DKWT = accuracy_DKWT / nr_runs
    accuracy_SELECT = accuracy_SELECT / nr_runs
    accuracy_SEEBS = accuracy_SEEBS / nr_runs
    accuracy_ExplThenVrfy = accuracy_ExplThenVrfy / nr_runs
    # print("RESULTS:")
    if "DKWT" in algos:
        f.write("DKWT: t:"+str(np.mean(times_DKWT))+"( "+str(np.std(times_DKWT)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_DKWT)+"\n")
    if "SELECT" in algos:
        f.write("SELECT: t:"+str(np.mean(times_SELECT))+"( "+str(np.std(times_SELECT)/np.sqrt(nr_runs))+" )"+\
              "\t Accuracy:"+str(accuracy_SELECT)+"\n")
    if "SEEBS" in algos:
        f.write("SEEBS: t:"+str(np.mean(times_SEEBS))+"( "+str(np.std(times_SEEBS)/np.sqrt(nr_runs))+" )"+\
              "\t Accuracy:"+str(accuracy_SEEBS)+"\n")
    if "ExploreThenVerify" in algos:
        f.write("Explore-then-Verify: t:"+str(np.mean(times_ExplThenVrfy))+"( "+str(np.std(times_ExplThenVrfy)/np.sqrt(nr_runs))+" )"+\
              "\t Accuracy:"+str(accuracy_ExplThenVrfy)+"\n")
    f.close()

###############################################################################
#   EXPERIMENT_PW_inst:
#
#   Compare our DKWT SELECT in the Dueling Bandits Case
###############################################################################   
    
def experiment_PW_inst(P,gamma,nr_runs = 20,random_seed = 1,filename="res.txt",show_progress = True):
    """ 
Evaluate our DKWT and PAC_Wrapper on the same instance.
    """
    assert len(P._get_GCW()) == 1, "P does not have a unique GCW!"
    np.random.seed(random_seed)
    random.seed(random_seed)
    # P.show()
    f = open(filename,"a")
    f.write("\n\n")
    f.write(">> experiment_PW_inst(P, gamma="+str(gamma)+", nr_runs="+\
            str(nr_runs)+", random_seed="+str(random_seed)+", filename="+str(filename)+"\n")
    if type(P) is pm.PLPM:
        f.write("theta : "+str(P.theta)+" k:"+str(P.k)+"\n")
    f.write("GCW of P : "+str(P._get_GCW())+"\n")
    if show_progress is True:
        print("Run \t | DKWT \t \t | PAC_Wrapper")
    
    times_DKWT = list()
    times_PAC_Wrapper = list()
    accuracy_DKWT = 0.0
    accuracy_PAC_Wrapper = 0.0
    True_GCW = P._get_GCW()[0]
    for run in range(0,nr_runs):
        # if show_progress is True:
        #     print(run,end=",")
            
        P.reset_observations()
        result_DKWT = GCW_algo.DKWT(gamma,P)
        time_DKWT = P.get_time()
        times_DKWT.append(time_DKWT)
        
        P.reset_observations()
        result_PAC_Wrapper = PAC_Wrapper.PAC_Wrapper(gamma,P)
        time_PAC_Wrapper = P.get_time()
        times_PAC_Wrapper.append(time_PAC_Wrapper)
        
        if show_progress is True:
            print(run," | ", result_DKWT, "( t: ",time_DKWT,") \t | ",result_PAC_Wrapper, "( t: ",time_PAC_Wrapper,")")
        if result_DKWT == True_GCW:
            accuracy_DKWT +=1 
        if result_PAC_Wrapper == True_GCW:
            accuracy_PAC_Wrapper += 1
            
    accuracy_DKWT = accuracy_DKWT / nr_runs
    accuracy_PAC_Wrapper = accuracy_PAC_Wrapper / nr_runs

    f.write("DKWT: t:"+str(np.mean(times_DKWT))+"( "+str(np.std(times_DKWT)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_DKWT)+"\n")
    f.write("PAC_Wrapper: t:"+str(np.mean(times_PAC_Wrapper))+"( "+str(np.std(times_PAC_Wrapper)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_PAC_Wrapper)+"\n")
    f.close()   
    

def experiment_DKWT_inst(P,gamma,nr_runs = 20,random_seed = 1,filename="res.txt",show_progress = True):
    """ 
Evaluate ONLY DKWT  some instance.
    """
    assert len(P._get_GCW()) == 1, "P does not have a unique GCW!"
    np.random.seed(random_seed)
    random.seed(random_seed)
    # P.show()
    f = open(filename,"a")
    f.write("\n\n")
    f.write(">> experiment_DKWT_inst(P, gamma="+str(gamma)+", nr_runs="+\
            str(nr_runs)+", random_seed="+str(random_seed)+", filename="+str(filename)+"\n")
    if type(P) is pm.PLPM:
        f.write("theta : "+str(P.theta)+" k:"+str(P.k)+"\n")
    f.write("GCW of P : "+str(P._get_GCW())+"\n")
    if show_progress is True:
        print("Run \t | DKWT \t \t")
    
    times_DKWT = list()
    accuracy_DKWT = 0.0
    True_GCW = P._get_GCW()[0]
    for run in range(0,nr_runs):
        # if show_progress is True:
        #     print(run,end=",")
        P.reset_observations()
        result_DKWT = GCW_algo.DKWT(gamma,P)
        time_DKWT = P.get_time()
        times_DKWT.append(time_DKWT)
        if show_progress is True:
            print(run," | ", result_DKWT, "( t: ",time_DKWT)
        if result_DKWT == True_GCW:
            accuracy_DKWT +=1 

    accuracy_DKWT = accuracy_DKWT / nr_runs

    f.write("DKWT: t:"+str(np.mean(times_DKWT))+"( "+str(np.std(times_DKWT)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_DKWT)+"\n")
    f.close()   


###############################################################################
#   experiment_DKWT_algo5
#
#   Compare DKWT with algo_5 
#   v1: Sample P from  PM_{k}^{m}(hGCW and \Delta^{h})
#   v2: Sample P from  PM_{k}^{m}(hGCW and \Delta^{0})
###############################################################################  
def experiment_DKWT_algo5_v1(m,k,h,gamma, nr_runs = 100, random_seed = 1,\
                         filename="results_exp_DKWT_algo5.txt",show_progress = True):
    """
In this experiment, 'nr_runs' P are sampled from  PM_{k}^{m}(hGCW and \Delta^{h}).
Then, DKWT and algo_5 are run on each of these instances once (with error prob.)
gamma, where algo_5 also obtains the value of 'h' as parameter.
The mean and standard termination time as well as the observed accuracy (averaged)
over all runs) are saved (more precisely: appended) to the file 'filename'.
    
EXAMPLE
h_values = [0.9,0.5,0.2,0.05]
for i in range(0,len(h_values)):
    h=h_values[i]
    experiment_DKWT_algo5_v1(m=5, k=3,h=h, gamma=0.05,nr_runs = 1000, random_seed = i)
    """
    f = open(filename,"a")  
    f.write("\n \n")
    f.write(">> experiment_DKWT_algo5_v1( m="+str(m)+", k="+str(k)+", h="+str(h)+",gamma="+str(gamma)+\
            ", nr_runs="+str(nr_runs)+", random_seed="+str(random_seed)+", filename="+str(filename)+") \n")
    np.random.seed(random_seed)
    random.seed(random_seed)
    # print("Run \t | DKWT \t \t | SELECT")               # For Debugging
    times_DKWT = list()
    times_algo_5 = list()
    # times_SELECT = list()
    accuracy_DKWT = 0.0
    accuracy_algo_5 = 0.0
    # accuracy_SELECT = 0.0
    if show_progress is True:
        print("\n Starting experiment_DKWT_algo5_v1 with parameter (m,k,h,gamma)=(",m,k,h,gamma,")")
        print("Runs finished: (Total:",nr_runs,")")
    
    for run in range(0,nr_runs):
        if show_progress is True:
            print(run,end=",")
        P = pm.sample_hGCW_PM_h(m, k, h)                    # HERE is the only difference to experiment_DKWT_algo5_v2             
        result_DKWT = GCW_algo.DKWT(gamma,P)
        time_DKWT = P.get_time()
        times_DKWT.append(time_DKWT)
        
        P.reset_observations()
        result_algo_5 = GCW_algo.algo_5(gamma, h, P)    # algo_5 obtains the 'h' as param.
        time_algo_5 = P.get_time()
        times_algo_5.append(time_algo_5)
        True_GCW = P._get_GCW()[0]
        
        # if k==2:
            # P.reset_observations()
            # epsilon = -np.log(gamma)/np.log(np.log2(m))
            # m_h = math.floor((1+epsilon)*math.log(2)/2*math.log(math.log(m,2),2)/(h*h))+1   # SELECT obtains 'h' as param.
            # result_SELECT, itera = selecta.SELECT(list(np.arange(m)), m_h, P)
            # times_SELECT.append(P.get_time())
        
        if result_DKWT == True_GCW:
            accuracy_DKWT +=1 
        if result_algo_5 == True_GCW:
            accuracy_algo_5 += 1
        # if k==2:
        #     if result_SELECT == True_GCW:
        #         accuracy_SELECT += 1
        # print(run," | ", result_DKWT==True_GCW, "( t: ",time_DKWT,") \t | ",result_SELECT==True_GCW, "( t: ",time_SELECT,")")
                # For Debugging
    
    accuracy_DKWT = accuracy_DKWT / nr_runs
    accuracy_algo_5 = accuracy_algo_5 / nr_runs
    # accuracy_SELECT = accuracy_SELECT / nr_runs
    # print("RESULTS:")
    f.write("DKWT: t:"+str(np.mean(times_DKWT))+"( "+str(np.std(times_DKWT)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_DKWT)+"\n")
    f.write("algo_5: t:"+str(np.mean(times_algo_5))+"( "+str(np.std(times_algo_5)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_algo_5)+"\n")
    # if k==2:
    #     f.write("SELECT: t:"+str(np.mean(times_SELECT))+"( "+str(np.std(times_SELECT)/np.sqrt(nr_runs))+" )"+\
    #       "\t Accuracy:"+str(accuracy_SELECT)+"\n")
    f.close()


def experiment_DKWT_algo5_v2(m,k,h,h2,gamma, nr_runs = 100, random_seed = 1,\
                         filename="results_exp_DKWT_algo5.txt",show_progress = True):
    """
In this experiment, 'nr_runs' P are sampled from  PM_{k}^{m}(hGCW and \Delta^{h2}).
Then, DKWT and algo_5 are run on each of these instances once (with error prob.)
gamma, where algo_5 also obtains the value of 'h' as parameter.
The mean and standard termination time as well as the observed accuracy (averaged)
over all runs) are saved (more precisely: appended) to the file 'filename'.
    
EXAMPLE
h_values = [0.9,0.5,0.2,0.05]
h2 = 0.01
for i in range(0,len(h_values)):
    h=h_values[i]
    experiment_DKWT_algo5_v2(m=5, k=3,h=h,h2=h2, gamma=0.05,nr_runs = 10, random_seed = i)
    """
    f = open(filename,"a")  
    f.write("\n \n")
    f.write(">> experiment_DKWT_algo5_v2( m="+str(m)+", k="+str(k)+", h="+str(h)+",gamma="+str(gamma)+\
            ", nr_runs="+str(nr_runs)+", random_seed="+str(random_seed)+", filename="+str(filename)+") \n")
    np.random.seed(random_seed)
    random.seed(random_seed)
    # print("Run \t | DKWT \t \t | SELECT")               # For Debugging
    times_DKWT = list()
    times_algo_5 = list()
    accuracy_DKWT = 0.0
    accuracy_algo_5 = 0.0
    if show_progress is True:
        print("\n Starting experiment_DKWT_algo5_v2 with parameter (m,k,h,gamma)=(",m,k,h,gamma,")")
        print("Runs finished: (Total:",nr_runs,")")
    
    for run in range(0,nr_runs):
        P = pm.sample_hGCW_PM_h2(m, k, h, h2)                    # Here is the only difference to "experiment_DKWT_algo5"                
        if show_progress is True:
            print("Run:",run,"(h(P)=",P._get_h(),")", "(h2(P)=",P._get_h2(),")")
        result_DKWT = GCW_algo.DKWT(gamma,P)
        time_DKWT = P.get_time()
        times_DKWT.append(time_DKWT)
        print("DKWT done.",end=",")
        P.reset_observations()
        # P.show()
        result_algo_5 = GCW_algo.algo_5(gamma, h, P)    # algo_5 obtains the 'h' as param.
        time_algo_5 = P.get_time()
        times_algo_5.append(time_algo_5)
        print("algo_5 done.")
        True_GCW = P._get_GCW()[0]
        if result_DKWT == True_GCW:
            accuracy_DKWT +=1 
        if result_algo_5 == True_GCW:
            accuracy_algo_5 += 1
        # print(run," | ", result_DKWT==True_GCW, "( t: ",time_DKWT,") \t | ",result_SELECT==True_GCW, "( t: ",time_SELECT,")")
                # For Debugging
    
    accuracy_DKWT = accuracy_DKWT / nr_runs
    accuracy_algo_5 = accuracy_algo_5 / nr_runs
    # print("RESULTS:")
    f.write("DKWT: t:"+str(np.mean(times_DKWT))+"( "+str(np.std(times_DKWT)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_DKWT)+"\n")
    f.write("algo_5: t:"+str(np.mean(times_algo_5))+"( "+str(np.std(times_algo_5)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_algo_5)+"\n")
    f.close()


###############################################################################
#   experiment_DKWT_arms
#
#   Compare our DKWT SELECT in the Dueling Bandits Case
#   Note: The results produced from this are NOT contained in our paper.
###############################################################################   
def experiment_DKWT_arms(P,gamma=0.05, nr_runs = 100, random_seed = 1,\
                         filename="Experiment_DKWT_arms.txt",plotfilename = "plot_exp_4_arms.png",show_progress = True):
    """
    Runs DKWT several times on P and plots a bar chart showing the average
    number of times an arm has been present in a query (="arm pull").
    The plot is saved to the file "filename".
    
EXAMPLE
h_values = [0.2,0.1]
for i in range(0,len(h_values)):
    h = h_values[i]
    experiment_dueling(m=6, h=h, gamma=0.05, nr_runs = 10, random_seed = i)
    """
    f = open(filename,"a")
    f.write("\n \n")
    f.write(">> experiment_DKWT_algo5( P="+str(P)+",gamma="+str(gamma)+", nr_runs="+\
            str(nr_runs)+", random_seed="+str(random_seed)+", filename="+str(filename)+\
                ", plotfilename="+str(plotfilename)+",) \n")
    if type(P) is pm.PLPM:
        f.write("theta:"+str(P.theta)+"\n")
    np.random.seed(random_seed)
    random.seed(random_seed)
    all_nr_arm_plays = list([])
    times_DKWT = list()
    accuracy_DKWT = 0.0
    if show_progress is True:
        print("\n \n Starting experiment_DKWT_arms with parameter plotfilename = "+str(plotfilename))
        print("Runs finished: (Total:",nr_runs,")")
    for run in range(0,nr_runs):
        P.reset_observations()
        result_DKWT = GCW_algo.DKWT(gamma,P)
        time_DKWT = P.get_time()
        times_DKWT.append(time_DKWT)
        all_nr_arm_plays.append(P.get_nr_arm_plays())
        True_GCW = P._get_GCW()[0]
        if result_DKWT == True_GCW:
            accuracy_DKWT +=1 
        if show_progress is True:
            print(run,end=",")
    accuracy_DKWT = accuracy_DKWT / nr_runs
    avg_nr_arm_plays = np.mean(all_nr_arm_plays,axis = 0)
    f.write("DKWT: t:"+str(np.mean(times_DKWT))+"( "+str(np.std(times_DKWT)/np.sqrt(nr_runs))+" )"+\
          "\t Accuracy:"+str(accuracy_DKWT)+"\n")
    f.write("Avg_nr_arm_plays: "+str(avg_nr_arm_plays)+"\n")
    # print("\nRESULTS (DKWT):")
    # print(" t:"+str(np.mean(times_DKWT))+"( "+str(np.std(times_DKWT)/np.sqrt(nr_runs))+" )"+\
    #       "\t Accuracy:"+str(accuracy_DKWT)+"\n")
    # print("Avg. nr of arm plays:", avg_nr_arm_plays)
    m=P.m
    plt.bar(range(0,m),avg_nr_arm_plays)
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 10,
        }
    fig, ax =  plt.subplots()
    plt.xlabel("arm",fontdict=font)
    plt.ylabel("avg. nr. of arm plays",fontdict=font)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%dK'))
    plt.bar(range(0,m),avg_nr_arm_plays/1000)
    # plt.boxplot(np.array(all_nr_arm_plays))
    plt.savefig(plotfilename,dpi=300)
    plt.show()
    f.close()
