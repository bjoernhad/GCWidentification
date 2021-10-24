#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the PL-parameters for the synthetic datasets considered in 
[Saha2020], cf. Appendix E (p. 56) therein.
The small datasets (with 15 arms) are:
    theta_g1, theta_g4, theta_arith, theta_geo, theta_b1 
The large datasets (with 50 arms) are:
    theta_g4b, theta_arithb, theta_geob

[Saha2020]: Aadirupa Saha, Aditya Gopalan
From PAC to Instance-Optimal Sample Complexity in the Plackett-Luce Model, 
Proceedings of ICML (2020)
URL: http://proceedings.mlr.press/v119/saha20b.html
"""

import numpy as np

###############################################################################
#   PART 1: Small datasets with 15 arms
###############################################################################

theta_g1 = [0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
theta_g4 = [1,0.7,0.7,0.7,0.7,0.7,0.5,0.5,0.5,0.5,0.5,0.01,0.01,0.01,0.01,0.01]

theta_arith = np.zeros(16)
theta_arith[0] = 1
for i in range(0,15):
    theta_arith[i+1] = theta_arith[i] - 0.06

theta_geo = np.zeros(16)
theta_geo[0] = 1
for i in range(0,15):
    theta_geo[i+1] = 0.8 * theta_geo[i]

theta_b1 = 0.6 * np.ones(16)
theta_b1[0] = 0.8

###############################################################################
#   PART 2: Big datasets with 50 arms
###############################################################################

theta_g4b = np.zeros(50)
theta_g4b[0] = 1
for i in range(1,18):
    theta_g4b[i] = 0.7 
for i in range(18,45):
    theta_g4b[i] = 0.5
for i in range(45,50):
    theta_g4b[i] = 0.01

theta_arithb = np.zeros(50)
theta_arithb[0] = 1 
for i in range(0,49):
    theta_arithb[i+1] = theta_arithb[i] - 0.02 
    # NOTE: There is a typo on p.56 in [Saha2020]: There they write "-0.2" instead of "0.02".
    #       This would clearly violate the assumption that all \theta_{i}'s are positive.

theta_geob = np.zeros(50)
theta_geob[0] = 1
for i in range(0,49):
    theta_geob[i+1] = 0.9 * theta_geob[i]