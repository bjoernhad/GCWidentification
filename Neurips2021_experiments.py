#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the explicit function calls to produce the experimental
results in our paper "Identification of the Generalized Condorcet Winner
in Multi-dueling Bandits".
"""

import itertools as it
import numpy as np
import PAC_Wrapper 
import GCW_algo 
import PM as pm
import data_examples as dataS
import random
import matplotlib.pyplot as plt
# import SELECT_algorithm as selecta
import math
import experiment_procedures as expproc


###############################################################################
#   EXPERIMENT 1:
#
#   Compare DKWT with SELECT, SEEBS and Select-then-verify in the dueling 
#   bandit case, cf. expproc.experiment_dueling()
###############################################################################   
# # #  CASE 1: m=5
filename_1 = "Experiment1_m5_gamma_005.txt"
f = open(filename_1,"w")
f.close()
h_values = [0.2,0.15,0.1,0.05]
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_dueling(m=5, h=h, gamma=0.05, nr_runs = 100, \
                                random_seed = i, filename = filename_1)
        
# CASE 2: m=10
filename_2 = "Experiment1_m10_gamma_005.txt"
f = open(filename_2,"w")
f.close()
h_values = [0.2,0.15,0.1,0.05]
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_dueling(m=10, h=h, gamma=0.05, nr_runs = 100, \
                                random_seed = i,filename = filename_2)


# # CASE 3: m=15 -- with 10 repetitions only !
filename_2 = "Experiment1_m15_gamma_005.txt"
f = open(filename_2,"w")
f.close()
h_values = [0.05,0.1,0.15,0.2]
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_dueling(m=15, h=h, gamma=0.05, nr_runs = 10, \
                                random_seed = i,filename = filename_2)
        
# CASE 4: m=20 -- with 10 repetitions only !
filename_2 = "Experiment1_m20_gamma_005.txt"
f = open(filename_2,"w")
f.close()
h_values = [0.05,0.1,0.15,0.2]
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_dueling(m=20, h=h, gamma=0.05, nr_runs = 10, \
                                random_seed = i,filename = filename_2)

###############################################################################
#   EXPERIMENT 2:
#
#   Compare DKWT with algo_5 on 
#   (a) PM_{k}^{5}(hGCW and \Delta^{h})
#   (b) PM_{k}^{5}(hGCW and \Delta^{h2}) for h2=0.01
###############################################################################  

################
## Version (a) 
################    
h_values = [0.9,0.7,0.5,0.3,0.1]
nr_runs = 1000
filename = "Experiment_DKWT_vs_algo5_v1.txt"
f = open(filename,"w")
f.close()
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_DKWT_algo5_v1(m=5, k=2,h=h, gamma=0.05,nr_runs = nr_runs, random_seed = i,filename=filename)
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_DKWT_algo5_v1(m=5, k=3,h=h, gamma=0.05,nr_runs = nr_runs, random_seed = i,filename=filename)
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_DKWT_algo5_v1(m=5, k=4,h=h, gamma=0.05,nr_runs = nr_runs, random_seed = i,filename=filename)
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_DKWT_algo5_v1(m=5, k=5,h=h, gamma=0.05,nr_runs = nr_runs, random_seed = i,filename=filename)


###############
## Version (b) 
############### 
h_values = [0.9,0.7,0.5,0.3,0.1]
nr_runs = 1000
filename = "Experiment_DKWT_vs_algo5_v2.txt"
f = open(filename,"w")
f.close()
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_DKWT_algo5_v2(m=5, k=2,h=h, h2=0.01, gamma=0.05,nr_runs = nr_runs, random_seed = i,filename=filename)
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_DKWT_algo5_v2(m=5, k=3,h=h, h2=0.01, gamma=0.05,nr_runs = nr_runs, random_seed = i,filename=filename)
for i in range(0,len(h_values)):
    h = h_values[i]
    expproc.experiment_DKWT_algo5_v2(m=5, k=4,h=h, h2=0.01, gamma=0.05,nr_runs = nr_runs, random_seed = i,filename=filename)



###############################################################################
#   EXPERIMENT 3: COMPARISON OF DKWT WITH PAC-WRAPPER
#
#   Compare our DKWT with PAC_Wrapper (from Saha et al.) on different instances
############################################################################### 
# # # # CASE 1: PL-Instance, m=5, k>=2
filename_3 = "Experiment_PW_m5.txt"
f = open(filename_3,"w")
f.close()
nr_runs = 10
P = pm.PLPM(theta = [1,0.8,0.6,0.4,0.2],k=2)
expproc.experiment_PW_inst(P=P,gamma=0.05,nr_runs=nr_runs,random_seed=1,\
                              filename=filename_3)
P = pm.PLPM(theta = [1,0.8,0.6,0.4,0.2],k=3)
expproc.experiment_PW_inst(P=P,gamma=0.05,nr_runs=nr_runs,random_seed=1,\
                              filename=filename_3)
P = pm.PLPM(theta = [1,0.8,0.6,0.4,0.2],k=4)
expproc.experiment_PW_inst(P=P,gamma=0.05,nr_runs=nr_runs,random_seed=1,\
                              filename=filename_3)
P = pm.PLPM(theta = [1,0.8,0.6,0.4,0.2],k=5)
expproc.experiment_PW_inst(P=P,gamma=0.05,nr_runs=nr_runs,random_seed=1,\
                              filename=filename_3)

# # # CASE 2: PL-Instance: m=10, k>=2
# filename_3 = "Experiment_PW_m10.txt"
# f = open(filename_3,"w")
# f.close()
# nr_runs = 10  
# y = 0.5
# theta = np.zeros(10)
# for i in range(0,len(theta)):
#     theta[i] = y**i
# print(theta)
# P = pm.PLPM(theta = theta,k=2)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=2,\
#                               filename=filename_3)
# P = pm.PLPM(theta = theta,k=3)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=3,\
#                               filename=filename_3)
# P = pm.PLPM(theta = theta,k=4)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=4,\
#                               filename=filename_3)
# P = pm.PLPM(theta = theta,k=5)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=5,\
#                               filename=filename_3)
# P = pm.PLPM(theta = theta,k=6)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=6,\
#                               filename=filename_3)
# P = pm.PLPM(theta = theta,k=7)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=7,\
#                               filename=filename_3)
# P = pm.PLPM(theta = theta,k=8)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=8,\
#                               filename=filename_3)
# P = pm.PLPM(theta = theta,k=9)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=9,\
#                               filename=filename_3)
# P = pm.PLPM(theta = theta,k=10)
# expproc.experiment_PW_inst(P=P,gamma=0.1,nr_runs=nr_runs,random_seed=10,\
#                               filename=filename_3)

    
# # # # CASE 2: Evaluate our DKWT on synthetic data from Saha et al.
# filename_3 = "Experiment_PW_PWData.txt"
# f = open(filename_3,"w")
# f.close() 

# P = pm.PLPM(theta = dataS.theta_arith,k=5)
# print("Starting the evaluation of DKWT on the synthetic data theta_arith from Saha et al.")
# expproc.experiment_DKWT_inst(P,gamma=0.01,nr_runs = 1000,random_seed = 1,filename="Experiment_PW_PWData.txt",show_progress = False)

# P = pm.PLPM(theta = dataS.theta_geo,k=5)
# print("Starting the evaluation of DKWT on the synthetic data theta_geo from Saha et al.")
# expproc.experiment_DKWT_inst(P,gamma=0.01,nr_runs = 1000,random_seed = 1,filename="Experiment_PW_PWData.txt",show_progress = False)




###############################################################################
#       END OF THE EXPERIMENTS IN OUR PAPER.
###############################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


###############################################################################
#   The following Experiment 4 was NOT included in our paper, as it only provides
#   insights into DKWT, which are more or less trivial.
#   Therefore, it is not essential for our paper.
###############################################################################
#   EXPERIMENT 4:
#
#   Run DKWT on some instances and plot the avg. nr. of arm pulls
###############################################################################         
# f = open("Experiment4.txt","w")
# f.close()
# nr_runs = 10000
# P = pm.PLPM([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],k=4)
# expproc.experiment_DKWT_arms(P,gamma=0.05, nr_runs = nr_runs, random_seed = 1,\
#                           filename="Experiment4.txt",plotfilename="Exp4_Plot1.png",show_progress = True)

# P = pm.PLPM([1,0.9,0.89,0.88,0.6,0.5,0.4,0.3,0.2,0.1],k=4)
# expproc.experiment_DKWT_arms(P,gamma=0.05, nr_runs = nr_runs, random_seed = 1,\
#                           filename="Experiment4.txt",plotfilename="Exp4_Plot2.png",show_progress = True)
  
# P = pm.PLPM([1,0.9,0.8,0.7,0.6,0.5,0.49,0.48,0.2,0.1],k=4)
# expproc.experiment_DKWT_arms(P,gamma=0.05, nr_runs = nr_runs, random_seed = 1,\
#                           filename="Experiment4.txt",plotfilename="Exp4_Plot3.png",show_progress = True)

