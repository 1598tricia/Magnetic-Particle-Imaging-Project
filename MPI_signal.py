import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from gradiometer_sensitivity import  Phi
plt.rcParams.update({'font.size': 18})

height = 0.025

H_amp = 5500000#200000#200000; #Change for different particle size
f_H = 16000 #Drive field frequency

def applied_field(t):
    return H_amp * np.sin(2 * np.pi * f_H * t) 

#component values, index of each array represents the stage of the filter
R1 = [910E3, 309E3, 196E3, 150E3, 130E3, 121E3]
R2 = [182E3, 187E3, 200E3, 215E3, 226E3, 237E3]
R3 = [453E3, 511E3, 649E3, 1E6, 2.4E6, 20E6]
C1 = [47E-12, 33E-12, 33E-12, 33E-12, 33E-12, 33E-12]
C2 = [22E-12, 33E-12, 33E-12, 33E-12, 33E-12, 33E-12]

def H(s, stages=6):
    H_s = 0 
    for i in range(stages):
        V_ratio = ((1 / (C1[i]*C2[i]*R3[i]*R2[i])) + s**2) / ((1 / (C1[i]*C2[i]*R2[i]**2)) + (s / (C1[i]*R2[i])) + s**2)
        if H_s == 0:
            H_s = V_ratio 
        else:
            H_s *= V_ratio 
    return H_s 

T = 298.15
mu_0 = 4*np.pi *10**(-7)
k_b = 1.38*10**(-23)
M0 = 0.044 #saturation magnetisation 

numcoil_1 = 144.
radius = (24*10**(-3))/2 #m
area = np.pi*radius**2 #m^2
n1A = numcoil_1*area
numcoil_2 = 143
n2A = numcoil_2*area

def emf(diameter, bins, time, t_stamp, yconc ):
    Ms = (1 / 6) * np.pi * diameter**3 * M0 #magnetic dipole
    magn = []
    dMdt = [];
    dHdt = []; 
    
    phival = []

    for k in range(len(bins)-1):
        
        t_mag = np.linspace(0, 0.001, 10000) + time[t_stamp]
        
        conc = yconc[t_stamp, k]
        
        phi = Phi(bins[k])
        phival.append(phi)
        
        def Particle_M(H):
            eta = (Ms * mu_0* H) / (k_b * T)
            M = conc * (Ms/mu_0) * ((np.cosh(eta) / np.sinh(eta)) - (1 / eta))
            return M
        
        def magnetisation(t):
            return Particle_M(applied_field(t))
        
        M_t = magnetisation(t_mag)
        magn.append(M_t)
        
        dMdt1 = derivative(magnetisation, t_mag, dx=(0.001/10000), order=9) 
        dMdt.append( dMdt1)
        
        dHdt1 =  2 * np.pi * f_H * H_amp * np.cos(2 * np.pi * f_H * t_mag) 
        dHdt.append( dHdt1)
    
        emf1 = [-mu_0*numcoil_1*area * (dMdt[i]+dHdt[i]) for i in range(len(dMdt))]
        
        exp_fac = np.exp(1j*np.pi/1000 )
        
        emf2 = [-mu_0*exp_fac*numcoil_2*area*dHdt[i] for i in range(len(dHdt))]
        
    emfa = np.array([phival[i]*(emf1[i] - emf2[i]) for i in range(len(emf1)) ])
    emf = np.sum(emfa, axis=0)

    return emf
    
maxamp = None
maxamp_f = None
def FFT(sig):
    freq = np.fft.fftfreq(len(sig))
    fftsig = np.fft.fft(sig) #performs the FFT on the signal
    freqs = [(1 / 0.0000001) * freq[i] for i in range(len(freq))]#calculates the frequencies for the x-axis of the graph

    # f = plt.figure(figsize=(10,10))
    # plt.plot(freqs, fftsig.real)
    # plt.yscale('log')
    # plt.xlim(0, 1000000)
    # plt.title("fftsig vs freqs")
    # plt.show()
    
    amp = np.abs(fftsig)
    # f = plt.figure(figsize=(10,10))
    # plt.plot(freqs, amp.real)
    # plt.yscale('log')
    # plt.xlim(0,1000000)
    # plt.title('f vs amps')
    # plt.show()
    
    # global maxamp
    # maxamp = np.max(amp)
    # #maxamp_f = np.abs(freqs[fftsig.argmax()])
    # #print(maxamp, maxamp_f)
    # amp_dB = 20*np.log10(amp/maxamp)

    # f = plt.figure(figsize=(10,10))
    # plt.plot(freqs,amp_dB)
    # plt.xlim(0, 1000000)
    # plt.title('f vs amps (dB)')
    # plt.show()
    
    #global gain
    gain=[]
    for freq in freqs:
        s = 1j * 2 * np.pi * freq #s is the angular frequency in rads/s, converted from Hz, comes from the formula for impedence of a capacitor
        gain.append((H(s, stages=3))) #Formula for gain is 20*log(Vout/Vin)  
    
    #global G
    freqs = np.array(freqs)
    G = amp * gain
    G_abs = np.abs(G)
   #  # w = 2*np.pi*np.array(freqs)
   #  f = plt.figure(figsize=(10,10))
   #  plt.plot(freqs/1000000, G_abs.real)#w,G_abs.real)
   # # plt.title('G vs w')
   #  #plt.xscale('log')
   #  plt.yscale('log')
   #  plt.ylabel(r'Amplitude ($V$)'); plt.xlabel('Frequency (MHz)')
   #  plt.xlim(0,1)#0E5)#10E5*2*np.pi)
   #  plt.show()
    
    # #index 48 is the amplitude of 3rd harmonic
    #global maxamp
    maxamp3H = np.abs(G_abs[48])#np.abs(freqs[G_abs.argmax()])

    # global maxphase
    # phase = np.angle(G, deg=True)
    # # f = plt.figure(figsize=(14,10))
    # # plt.plot(w, phase)
    # # plt.title('phase (deg) vs w')
    # # plt.xlim(0,0.3E6*2*np.pi)
    # # plt.show()
    # # maxphase = np.abs(phase[48])
    
    return maxamp3H
