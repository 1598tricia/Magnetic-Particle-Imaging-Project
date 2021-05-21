import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Equations and theory taken from https://arxiv.org/pdf/cond-mat/0404049.pdf pg.21

mu_0 = 4 * np.pi * 10**(-7) #Permeability of free space
a = 0.012 #radius of the coil sections in m
d = 0.02375 #distance between the centers of the cancelation coils and the central coil in m
N = 72 #Number of turns in the upper 
mu = 1 #magnitude of the single monment which moves through the coil (arbitrarily given as 1)

def f(z):
    return a**2 / (a**2 + z**2)**(3/2)

def Phi(z):
    '''
    Calculated the total flux through a finite z portion of the coil for a moment mu
    '''
    z -= 0.0144 #Modify z so that z = 0 is the bottom of the sample volume
    return N * (mu_0 / 2) * (f(z - d) - (2 * f(z)) + f(z + d)) * mu

z = np.linspace(0, 0.03, 100) #List of z heights to calculate for, z = 0 is the center of the coil.
Phi_z = []
#Plot to show the coil sensitivity over the current 30mm sample measuring volume
for val in z:
    Phi_z.append(Phi(val))


plt.rcParams.update({'font.size': 15})
p = plt.figure(figsize=(10,10))
plt.plot(Phi_z, z*1000)
plt.xlabel("Flux (Wb)"); plt.ylabel("Height (mm)")
plt.grid()
plt.show()

'''
The units of the flux do not represent realistic values, simply the correct relationship between
flux though the coil and z height, this gives the correct sensitivity profile of the coil, since
flux is proportional to the measured voltage through Faraday's law.

In terms of implementing this into the simulation the Phi values represents the sensitiviy of the coil
so the particle 'signal' can be simulated by some sort of multiplication based on the height of the
particle, may want to normalize the Phi values between -1 and 1 to make this simple.

I.e instead of taking the signal to be total number of particles in the simulation, it could instead be
the sum total of the flux contribution of all the particles
'''