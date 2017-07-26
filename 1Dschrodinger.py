#!/usr/bin/python
# Simulates time dependent Schrodinger equation in 1D
# Martin Pittermann 21.07.2017
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.polynomial.hermite import Hermite

#wraps over, slower
def D_wrap(A, dX):
	d = A - np.roll(A, 1)
	d = 0.5 * (d + np.roll(d, -1))
	return d / dX

#doesn't wrap over, faster
def D_nowrap(A, dX):
	d = A[1:] - A[:-1]
	d = np.concatenate(([d[0]], d, [d[-1]]))
	d = 0.5 * (d[1:] + d[:-1])
	return d / dX

D = D_nowrap	#select derivative function

def normalize(A, norm):	#normalize wave function -->  <A|A> = norm
	return A * np.sqrt(norm / np.sum(np.abs(A**2)))

N = 400		#number of points
size = 10	#interval size: [-size, size]
dX = 1		#x step
dt = 0.001	#time step
X = np.linspace(-size, size, N)#X coordinates of points, used to generate initial wave function and potential



#CONFIGURATION

############ initial wave function for moving particle
x0 = -5		#initial position
speed = 1500
Aparticle = np.exp(-4*(X-x0)*((X-x0)+speed*1j))	#initial wave function, derived from gaussian curve in momentum space shifted by speed, see http://www.wolframalpha.com/input/?i=integrate+from+-inf+to+inf+exp(-(p-5)%5E2)+*+exp(i*x*p)+dp

########### initial wave functions for harmonic oscillator Vho = X**2/80
HO_norm = 20
k = 1/40

V_HO = k*X**2/2		#harmonic oscillator potential

HO_cl = (8*k)**(1/4)	#characteristic length
Xc = X / HO_cl			#X in units of characteristic length

def HOsol(n):	#get nth eigenstate of harmonic oscillator
	coeffs = np.zeros(n + 1)
	coeffs[n] = 1
	A = Hermite(coeffs)(Xc) * np.exp(-Xc**2/2)
	return normalize(A, HO_norm)

########### select initial wave function
# A = (HOsol(0) + HOsol(2))/np.sqrt(2)		#superposition of eigenfunctions
# A = (HOsol(0) + HOsol(1))/np.sqrt(2)		#superposition of eigenfunctions
# A = Aparticle
A = HOsol(6)

########### potential
Vnone = np.zeros(N)	#no potential

barrierheight = 0.5
barrierwidth = 1.
barrierposition = 2.
Vbarrier = -barrierheight*(np.abs(X - barrierposition) < barrierwidth/2)		#potential barrier

############# select potential
V = V_HO
# V = Vbarrier




#INTERNALS

A = np.array(A, dtype=np.complex)	#convert A to np.complex array

fig, ax = plt.subplots()	#set up matplotlib
line, = ax.plot(X, A)
ax.set_ylim([0, 1])

ax.plot(X, V)

def animate(i):	#advance simulation by one frame
	global A	#needed to access global variable from function

	for i in range(1000):	#multiple time steps per frame
		A += dt*1j*(D(D(A, dX), dX) - V * A)	#schrodinger equation

	probdist = np.abs(A**2)

	print('\r<Psi|Psi> =', np.sum(probdist), end='')	#print <Psi|Psi>

	line.set_ydata(probdist)
	return line,

fps = 100	#pretty optimistic
ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), interval=1e3/fps, blit=False)	#no idea what this does exactly, it just works
plt.show()

print()
