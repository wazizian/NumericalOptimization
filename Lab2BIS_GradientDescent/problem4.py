#!/usr/bin/env python
# coding: utf-8

# # Problem 4
# 
# 
# The objective of Problem 4 is to minimize a non-convex function $t$ with two minimizers on $\mathbb{R}^2$ (unconstrained): 
# 
# $$\begin{array}{rrcll}
# t: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  & (0.6 x_1 + 0.2 x_2)^2 \left((0.6 x_1 + 0.2 x_2)^2 - 4 (0.6 x_1 + 0.2 x_2)+4\right) + (-0.2 x_1 + 0.6 x_2)^2
# \end{array}$$





##### Function definition
def f(x):
	x1 = x[0]
	x2 = x[1]
	return (0.6*x1 + 0.2*x2)**2 * ((0.6*x1 + 0.2*x2)**2 - 4*(0.6*x1 + 0.2*x2)+4) + (-0.2*x1 + 0.6*x2)**2
####

##### Plot parameters f
x1_min = -1
x1_max = 4
x2_min = -1
x2_max = 4
nb_points = 200
levels = [0.05,0.5,1,2,5]
vmin = 0
vmax = 5
title = 'two pits'
####





###### Useful Parameters
L = 8        # Lipschitz constant of the gradient


### Oracles

# Q: Complete the first order oracle `f_grad`.




import numpy as np

a, b = 0.6, 0.2

##### Gradient oracle
def f_grad(x):
    x, y = x[0], x[1]
    df_dx = 2*a*(a*x + b*y)**2*(a*x + b*y - 2) - 2*a*(a*x + b*y)*(4*a*x + 4*b*y - (a*x + b*y)**2 - 4) - 2*b*(a*y - b*x)
    df_dy = 2*a*(a*y - b*x) + 2*b*(a*x + b*y)**2*(a*x + b*y - 2) - 2*b*(a*x + b*y)*(4*a*x + 4*b*y - (a*x + b*y)**2 - 4)
    return np.array( [ df_dx  ,  df_dy ] )
# Q: Does a second order oracle exist for any point?
def f_grad_hessian(x):
    g = f_grad(x)
    x, y = x[0], x[1]
    
    d2f_dx2 = 12*a**4*x**2 + 24*a**3*b*x*y - 24*a**3*x + 12*a**2*b**2*y**2 - 24*a**2*b*y + 8*a**2 + 2*b**2
    d2f_dy2 = 12*a**2*b**2*x**2 + 2*a**2 + 24*a*b**3*x*y - 24*a*b**2*x + 12*b**4*y**2 - 24*b**3*y + 8*b**2
    d2f_dx_dy = 6*a*b*(2*a**2*x**2 + 4*a*b*x*y - 4*a*x + 2*b**2*y**2 - 4*b*y + 1)
    H = np.array([[d2f_dx2, d2f_dx_dy],
                  [d2f_dx_dy, d2f_dy2]])
    return g,H
