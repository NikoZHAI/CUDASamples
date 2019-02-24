#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:10:39 2019

@author: niko
"""

import numpy as np
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(7, 3), dpi=150)

# Load elapsed time by varing number of threads. (20 times for each configuration)
t_threads = np.loadtxt(fname='data.dat').reshape(6, 20)


x_left = [5, 6, 7, 8, 9, 10]
x_lab_left = np.exp2(x_left).astype('u4')
mean_t_threads = t_threads.mean(axis=1) # u
std_t_threads = t_threads.std(axis=1)   # std_dev
upper_threads = mean_t_threads + 3*std_t_threads
lower_threads = mean_t_threads - 3*std_t_threads

ax1 = axs[0]
l_u = ax1.plot(x_left, mean_t_threads, label='mean elapsed time u (ms)',
               color='royalblue')
l_plus_3std = ax1.plot(x_left, upper_threads, linestyle=':', color='orangered',
                       label='u$\pm$3$\sigma$ (ms)', )
l_minus_3std = ax1.plot(x_left, lower_threads, linestyle=':', color='orangered')
ax1.set_xticks(ticks=x_left)
ax1.set_xticklabels(x_lab_left, fontsize=8)
ax1.set_yticks([550,600,650,700, 750, 800])
ax1.set_yticklabels([550,600,650,700, 750, 800], fontsize=8)
ax1.set_xlabel('Number of threads per block [N]', fontsize=9)
ax1.set_ylabel('Mean elapsed time [ms]', fontsize=9)
ax1.legend(loc='center right', fontsize='x-small', bbox_to_anchor=(1., 0.85))
ax1.grid(ls=':')


# ============================================================================ #

# Load elapsed time by varing number of blocks. (20 times for each configuration)
t_blocks = np.loadtxt(fname='data1.dat').reshape(13, 20)


x_right = np.arange(5, 18, dtype='u4')
x_lab_right = ['$2^{{ {} }}$'.format(k) for k in x_right] # np.exp2(x_right).astype('u4')
mean_t_blocks = t_blocks.mean(axis=1) # u
std_t_blocks = t_blocks.std(axis=1)   # std_dev
upper_blocks = mean_t_blocks + 3*std_t_blocks
lower_blocks = mean_t_blocks - 3*std_t_blocks

ax2 = axs[1]
l_u = ax2.plot(x_right, mean_t_blocks, label='mean elapsed time u (ms)',
               color='royalblue')
l_plus_3std = ax2.plot(x_right, upper_blocks, linestyle=':', color='orangered',
                       label='u$\pm$3$\sigma$ (ms)', )
l_minus_3std = ax2.plot(x_right, lower_blocks, linestyle=':', color='orangered')
ax2.set_xticks(ticks=x_right)
ax2.set_xticklabels(x_lab_right, fontsize=8)
ax2.set_yticks(np.arange(500, 725, 25).astype(int))
ax2.set_yticklabels(np.arange(500, 725, 25).astype(int), fontsize=8)
ax2.set_xlabel('Number of blocks in grid [N]', fontsize=9)
ax2.set_ylabel('Mean elapsed time [ms]', fontsize=9)
ax2.legend(loc='center right', fontsize='x-small', bbox_to_anchor=(1., 0.86))
ax2.grid(ls=':')

fig.show()
