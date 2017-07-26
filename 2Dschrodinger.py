#!/usr/bin/python
# Simulates time dependent Schrodinger equation in 2D
# Martin Pittermann 21.07.2017
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sizeX = 100
width = 4
dx = 0.08

x = np.linspace(-sizeX * dx/2, sizeX * dx/2, sizeX)
y = np.linspace(-sizeX * dx/4, sizeX * dx/4, sizeX / 2)
X, Y = np.meshgrid(x, y)

#initial wave function
x0 = -2
y0 = 0
A = np.exp(-4*((X-x0)*((X-x0)-1500j) + (Y-y0)*(Y-y0)))		#gaussian centered around (1500, 0) in momentum space

#potential
barrierxpos = -0.5
barrierwidth = 0.25
slitpos = 0.25
slitwidth = 0.25
Vbarrier = 1 * (np.abs(X - barrierxpos) < barrierwidth/2)
Vbarrier *= 1 - 1*((np.abs(Y - slitpos) < slitwidth/2) + (np.abs(Y + slitpos) < slitwidth/2))
Vbarrier = 100*np.array(Vbarrier, dtype=np.complex)		#double slit potential

Vzero = X - X	#no potential, free particle

V = Vbarrier
##############


Vabs = np.abs(V**2)
v = np.abs(A)**2

fig, ax = plt.subplots()
im = plt.imshow(v)	#show wave function once to set z range
im.set_cmap('nipy_spectral')

def animate(frame):
	global A, V

	if frame > 100: #pause first 100 frames
		for i in range(60):
			gx, gy = np.gradient(A, dx)
			gtot = np.gradient(gx, dx, axis=0) + np.gradient(gy, dx, axis=1)
			A += 0.000025j * (gtot - V * A)

	v = np.abs(A)**2
	print(np.sum(v))
	im.set_data(v + Vabs)

	return im,

fps = 100
ani = animation.FuncAnimation(fig, animate, np.arange(1, 20000), interval=1e3/fps, blit=True)
plt.show()
