#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This file contains an implementation of the SELECT algorithm from [Mohajer2017]

[Mohajer2017]: Soheil Mohajer, Changho Suh, and Adel Elmahdy. Active learning for top-k rank aggregation from
noisy comparisons. In Proceedings of International Conference on Machine Learning (ICML),
pages 2488–2497, 2017.

"""

import math
import numpy as np

def SELECT(X, m, P):
    # X: data to sort (in our case arms)
    # m: number of comparisons per pair... something like confidence but for all pairs
    assert P.k == 2, "SELECT requires k=2."
    itera = 0
    n = len(X)
    a = list(np.arange(n))
    if int(math.log(n, 2)) == math.log(n, 2):
        last_l = int(math.log(n, 2))
    else:
        last_l = int(math.log(n, 2)) + 1
    for l in range(last_l):
        if int(n / math.pow(2, l + 1)) == n / math.pow(2, l + 1):
            last_i = int(n / math.pow(2, l + 1))
        else:
            last_i = int(n / math.pow(2, l + 1)) + 1
        for i in range(last_i):
            T = 0
            if len(a) >= 2 * i + 2:
                if a[2 * i] == a[2 * i + 1]:
                    continue
            else:
                if a[2 * i] == a[2 * i - 1]:
                    continue
            for t in range(m):
                if len(a) > 2 * i + 1:
                    Query, result = P.pull_arm([int(a[2 * i]), int(a[2 * i + 1])],size=1)
                    Y = False
                    if (Query[0] is int(a[2*i]) and result[0]==1) or \
                        (Query[0] is int(a[2*i+1]) and result[0]==0):
                            Y = True
                    # print([int(a[2 * i]), int(a[2 * i + 1])], Query, result, "Y",Y,type(Y))   #For Debugging
                else:
                    Query, result = P.pull_arm([int(a[2 * i]), int(a[2 * i - 1])],size=1)
                    Y = False
                    if (Query[0] is int(a[2*i]) and result[0]==1) or \
                        (Query[0] is int(a[2*i-1]) and result[0]==0):
                            Y = True
                itera += 1
                if Y is True:
                    T += 1
            if T >= m/2:
                a[i] = a[2*i]
            else:
                if len(a) > 2 * i + 1:
                    a[i] = a[2*i+1]
                else:
                    a[i] = a[2 * i - 1]
    return a[0], itera
