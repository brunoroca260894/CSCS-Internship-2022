#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat June 16 16:44:44 2022

@author: bruno
"""

import numpy as np
import matplotlib.pyplot as plt

#################

file_name_gpu = 'time_matrices.txt'
file_name_cpu = 'time_matrices_cpu.txt'

N = np.array([100, 500, 1000, 5000, 10000])

data_gpu = np.loadtxt(file_name_gpu, unpack = True)
data_gpu = data_gpu.T

data_cpu = np.loadtxt(file_name_cpu, unpack = True)
data_cpu = data_cpu.T

fig1, ax1 = plt.subplots()
ax1.loglog( N, data_gpu.mean(axis = 0), label = "GPU, NVIDIA GeForce 840M", marker = 'x', lw =1)
ax1.loglog( N, data_cpu.mean(axis = 0), label = "CPU", marker = 'x', lw =1)
plt.title(r'time using chrono lib, mean over 10 runs')
ax1.set_xticks(N)
plt.xlabel(r'matrix size $N$')
plt.ylabel(r'time in $\mu s$')
plt.legend(loc='upper left')
plt.savefig('computation_ime.pdf', dpi = 300, bbox_inches = 'tight')
plt.show()

