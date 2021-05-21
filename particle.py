import numpy as np
import random

class Particle:

    def __init__(self, diameter, susceptibility):
        self.position = [0, 0]
        self.velocity = [0, 0]
        self.diameter = diameter
        self.susceptibility = susceptibility
        self.volume = (4 / 3) * np.pi * (self.diameter / 2)**3

    def apply_force(self, field_strength, field_gradient, viscosity):
        mu_0 = 4 * np.pi * 10**(-7); k_b = 1.38*10**(-23); T = 298.15
        iron3ox_density = 5240; poly_density = 1050 #kgm^-3
        
        m = self.susceptibility * field_strength
        self.Diffusivity = (k_b*T)/(6*np.pi*viscosity*(self.diameter/2))
        
        V_FeO = self.volume*(1+((2*iron3ox_density)/poly_density))**(-1)
        V_ps = self.volume - self.volume*(1+((2*iron3ox_density)/poly_density))**(-1)
        m_T = iron3ox_density*V_FeO + poly_density*V_ps
        
        force_m = mu_0 * V_FeO * m * field_gradient
        force_g = m_T * 9.81
        
        self.velocity[1] = - (force_m + force_g) / (6 * np.pi * viscosity * (self.diameter / 2))
        self.velocity[0] = 0
        
        return True

    def move(self, timestep):
        stdev = np.sqrt(2*self.Diffusivity*timestep)
        factor = 100

        self.brownian_y = (np.random.normal(0.0, stdev*factor))
        self.brownian_x = (np.random.normal(0., stdev*factor))
        
        self.position[1] += (self.velocity[1] * timestep) + (self.brownian_y)

        self.position[0] += (self.velocity[0]* timestep) + (self.brownian_x)

        return True

    def set_position(self, width, height):
        self.position[0] =  random.random() * width
        self.position[1] = random.random() * height
