#!/usr/bin/env python
"""
This script will solve the Lorenz dynamical system.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# model parameters
sigma = 10.0
rho = 28.0
beta = 8.0/3.0

# define system
def f(y, t):
    f0 = sigma*(y[1]-y[0])
    f1 = y[0]*(rho-y[2])-y[1]
    f2 = y[0]*y[1]-beta*y[2]
    return np.array([f0, f1, f2])

# initial condition
y0 = np.array([0.0, 1.0, 1.05])

# time interval
dt = 0.01
max_T = 10000.0
t = np.linspace(0.0, dt, max_T)

# Simple integrator
y = np.zeros((3, t.size))
y[:, 0] = y0

for i, time_index in enumerate(t[1:]):
    y[:, i+1] = y[:, i] + dt*f(y[:, i], time_index)

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(*y)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.savefig('blah2.png')
