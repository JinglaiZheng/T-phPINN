# -*- coding: utf-8 -*-
"""
The code for solving non-fourier heat conduction equation by hPINN based on deepxde
If you use this code for scientific research, please cite the following articles:
Improved physics-informed neural networks for solving 2D non-Fourier heat conduction equation
@author: JinglaiZheng
"""

# Import requirements
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import math

# parameter definition
end_time = 1
m1 = 2.5
m2 = 1.5
theta = 0.25*math.pi
r = 1.25
t0 = 0.6
L = 5
m1_jian = m1/L
m2_jian = m2/L
r_jian = r/L
k = math.tan(theta)
x1 = m1_jian-r_jian*math.cos(theta)
y1 = m2_jian-r_jian*math.sin(theta)
x2 = m1_jian+r_jian*math.cos(theta)
y2 = m2_jian+r_jian*math.sin(theta)

# Residual error terms
def pde(X, T):
    dT_xx = dde.grad.hessian(T, X, i=0, j=0)
    dT_yy = dde.grad.hessian(T, X, i=1, j=1)
    dT_t = dde.grad.jacobian(T, X, i=0, j=2)
    dT_tt = dde.grad.hessian(T, X, i=2, j=2)
    return dT_tt+dT_t-(dT_xx+dT_yy)
def r_boundary(X, on_boundary):
    x, y, t = X
    return on_boundary and np.isclose(x, 1)
def l_boundary(X, on_boundary):
    x, y, t = X
    return on_boundary and np.isclose(x, 0)
def up_boundary(X, on_boundary):
    x, y, t = X
    return on_boundary and np.isclose(y, 1)
def down_boundary(X, on_boundary):
    x, y, t = X
    return on_boundary and np.isclose(y, 0)
def boundary_initial(X, on_initial):
    x, y, t = X
    return on_initial and np.isclose(t, 0)
def crack_boundary(X, on_boundary):
    x, y, t = X
    return ((x-m1_jian)*(x-m1_jian)+(y-m2_jian)*(y-m2_jian))<=(r_jian*r_jian) and np.isclose(k*x+m2_jian-m1_jian*k,y)
def init_func(X):
    return 0*np.ones((len(X), 1))
def func_one(X):
    return 1*np.ones((len(X), 1))
def func_zero(X):
    return np.zeros((len(X), 1))
def crack_points(x1, y1, x2, y2, n_xy, num_time):
    points = []
    time_points = np.linspace(0, 1, num_time)
    for t in time_points:
        for i in range(n_xy):
            x = x1 + (i / (n_xy - 1)) * (x2 - x1)
            y = y1 + (i / (n_xy - 1)) * (y2 - y1)
            points.append([x, y, t])
    return np.array(points)
n_xy = 50
num_time = 100
anchors = crack_points(x1=x1, y1=y1, x2=x2, y2=y2, n_xy=n_xy ,num_time=num_time)

# Basic model
num_domain = 10000
num_boundary = 5000
num_initial = 5000
layer_size = [3] + [40] * 4 + [1]
activation_func = "tanh"
initializer = "Glorot uniform"
lr = 1e-3
loss_weights = [1, 1, 1, 1, 1]
optimizer = "adam"
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])
timedomain = dde.geometry.TimeDomain(0, end_time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
bc_l = dde.NeumannBC(geomtime, func_zero, l_boundary, component=0)
bc_r = dde.NeumannBC(geomtime, func_zero, r_boundary, component=0)
bc_up = dde.DirichletBC(geomtime, func_zero, up_boundary, component=0)
bc_low = dde.DirichletBC(geomtime, func_one, down_boundary,  component=0)
bc_cracks = dde.OperatorBC(geomtime,
                           lambda x, y, _: math.sin(theta)*dde.grad.jacobian(y, x, i=0, j=0)+math.cos(theta)*dde.grad.jacobian(y, x, i=0, j=1),
                           crack_boundary)
ic = dde.IC(geomtime, init_func, boundary_initial, component=0)
ic_2 = dde.OperatorBC(geomtime,
                      lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=2),
                      lambda _, on_initial: on_initial)
data = dde.data.TimePDE(
    geomtime, pde, [bc_l, bc_r, ic, ic_2],
    num_domain=num_domain, num_boundary=num_boundary,
    num_initial=num_initial, train_distribution='Hammersley')
net = dde.nn.FNN(layer_size, activation_func, initializer)
def output_transform(x, y):
    return (1 - x[:, 1:2])*x[:, 1:2] * y[:, 0:1] + 1 - x[:, 1:2]
net.apply_output_transform(output_transform)
model = dde.Model(data, net)
model.compile("L-BFGS-B")

# Plot the solution compared with COMSOL Multiphysics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
t = 1.0
datalist = np.loadtxt()
x = datalist[:,0]
y = datalist[:,1]
u = datalist[:,2]
t_val = np.full_like(x, t)
coords = np.column_stack((x.flatten(), y.flatten(), t_val.flatten()))
u_pred = model.predict(coords)
u_pred1 = np.reshape(u_pred[:,0], x.shape)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
color = "viridis"
scatter1 = axs[0].scatter(x, y, c=u, cmap=color)
axs[0].set_title("COMSOL")
scatter2 = axs[1].scatter(x, y, c=u_pred1, cmap=color)
axs[1].set_title("PINN")
scatter3 = axs[2].scatter(x, y, c=- u + u_pred1, cmap=color)
axs[2].set_title("ABSOLUTE ERROR")
plt.show()
