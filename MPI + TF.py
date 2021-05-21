import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

plt.rcParams.update({'font.size': 15})

#Function to calculate particle magnetisation 
#Note M0 and c could change based on sample
#H is the applied field and ccan be changed
T = 298.15
mu_0 = 4*np.pi *10**(-7)
k_b = 1.38*10**(-23)
c = 2.77 * 10**9 #number of particles / m**3 (estimate)
M0 = 0.044 #saturation magnetisation 
D = 2*50 * 10**(-7) 
Ms = (1 / 6) * np.pi * D**3 * M0 #magnetic dipole
def Particle_M(H):
    eta = (Ms * mu_0* H) / (k_b * T)
    M = c * (Ms/mu_0) * ((np.cosh(eta) / np.sinh(eta)) - (1 / eta))
    return M

#calculations to demonstrate the langevin function
H_t = np.linspace(-5000, 5000, 1000)
M_t = [] 

for val in H_t:
    M_t.append(Particle_M(val))

#Plot of Langevin function
f = plt.figure(figsize=(10,10))
plt.plot(H_t/1000, M_t, linewidth=5)
#plt.title('M vs H')
plt.ylabel('$M (Am^{-1})$', fontsize=15); plt.xlabel('$H (kT)$', fontsize=15)
plt.grid()
plt.show()

#calculations for time domain applied field and sample magnitisation
t = np.linspace(0, 0.001, 10000) #array of time values used for time domain values
t = t[1:]

f_Ht = 16000 #Frequency of applied field
H_amp = 5000 #Amplitude of applied field
def applied_field(t):
    return H_amp * np.sin(2 * np.pi * f_Ht * t) #16000 represents frequency of applied field

def magnetisation(t):
    return Particle_M(applied_field(t))

H_t = []
for time in t:
    H_t.append(applied_field(time)) 

M_t = []
for time in t:
    M_t.append(magnetisation(time))

#Plot of time domain applied field and sample magnetisation
f = plt.figure(figsize=(10,10))
p = plt.subplot(2,1,1)
plt.title('H vs t')
p.plot(t, H_t)

p = plt.subplot(2,1,2)
plt.title('M vs t')
p.plot(t, M_t)
plt.show()

# f = plt.figure(figsize=(10,10))
# plt.plot(H_t, M_t)
# plt.show()

#calculations for the derivatives of the applied field and magnetisation
dHdt = []
for time in t:
    dHdt.append(derivative(applied_field, time))

dHdt = 2 * np.pi * f_Ht * H_amp * np.cos(2 * np.pi * f_Ht * t)

dMdt = []
for time in t:
    dMdt.append(derivative(magnetisation, time, dx=(0.002/10000), order=9)) #langevin givens M from values of H
dMdt = np.array(dMdt)

#Plot of applied field and magnetisation differentials
f = plt.figure(figsize=(10,10))
p = plt.subplot(2,1,1)
plt.title('dH/dt vs t')
plt.plot(t, dHdt)

p = plt.subplot(2,1,2)
plt.title('dM/dt vs t')
plt.plot(t, dMdt)
plt.show()

H_t = np.array(H_t); M_t = np.array(M_t)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,14))
ax2.plot(H_t/1000, M_t*1000); ax2.set(ylabel='$M (mAm^{-1})$', xlabel='H (kT)'); ax2.grid()
#ax3.plot((H_t[:1500])/1000, t[:1500]*1000000, 'tab:orange'); ax3.set(ylabel=r't ($\mu s$)', xlabel='H (kT)'); ax3.grid()
ax1.axis('off')
ax4.plot(t[:1500]*1000000, dMdt[:1500], 'tab:green'); ax4.set(ylabel='u', xlabel=r't ($\mu s$)'); ax4.grid(); ax4.set_yticklabels([])
ax3.plot(t[:1500]*1000000, M_t[:1500]*1000, 'tab:red'); ax3.set(ylabel='$M $', xlabel=r't ($\mu s$)'); ax3.grid(); ax3.set_yticklabels([])


fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(12,5))
ax1.plot((t[:1500]*1000000), (H_t[:1500])/1000, 'tab:orange'); ax1.set_yticklabels([])
ax1.set(ylabel='H', xlabel=r't ($\mu s$)'); ax1.grid()
ax3.plot(t[:1500]*1000000, dMdt[:1500], 'tab:green'); ax3.set(ylabel='u', xlabel=r't ($\mu s$)'); ax3.grid(); ax3.set_yticklabels([])
ax2.plot(t[:1500]*1000000, M_t[:1500]*1000, 'tab:red'); ax2.set(ylabel='$M $', xlabel=r't ($\mu s$)'); ax2.grid(); ax2.set_yticklabels([])

#component values, index of each array represents the stage of the filter
R1 = [910E3, 309E3, 196E3, 150E3, 130E3, 121E3]
R2 = [182E3, 187E3, 200E3, 215E3, 226E3, 237E3]
R3 = [453E3, 511E3, 649E3, 1E6, 2.4E6, 20E6]
C1 = [47E-12, 33E-12, 33E-12, 33E-12, 33E-12, 33E-12]
C2 = [22E-12, 33E-12, 33E-12, 33E-12, 33E-12, 33E-12]

gain = [] 

def H(s, stages=6):
    H_s = 0 
    for i in range(stages):
        V_ratio = ((1 / (C1[i]*C2[i]*R3[i]*R2[i])) + s**2) / ((1 / (C1[i]*C2[i]*R2[i]**2)) + (s / (C1[i]*R2[i])) + s**2)
        if H_s == 0:
            H_s = V_ratio 
        else:
            H_s *= V_ratio 
    return H_s 

#frequency space function, returns the frequency domain version of a given signal
#use as FFT(sig) where sig is H_t, dMdt etc, time domain signals.
maxamp = None
maxamp_f = None
def FFT(sig):
    freq = np.fft.fftfreq(len(sig))
    fftsig = np.fft.fft(sig) #performs the FFT on the signal
    #global freqs
    freqs = [(1 / 0.0000001) * freq[i] for i in range(len(freq))]#calculates the frequencies for the x-axis of the graph

    f = plt.figure(figsize=(10,10))

    plt.plot(freqs, fftsig)
    plt.yscale('log')
    plt.xlim(0, 1000000)
    plt.title("fftsig vs freqs")
    plt.show()
    
    freqs = np.array(freqs)
    amp = np.abs(fftsig)
    f = plt.figure(figsize=(10,10))
    plt.plot(freqs/1000000, amp.real)
    plt.yscale('log'); plt.ylabel('Amplitude (V)')
    plt.xlabel('Frequency (MHz)'); plt.xlim(0,1);
    #plt.title('f vs amps')
    plt.show()
    
    global maxamp
    maxamp = np.max(amp)
    global maxamp_f
    maxamp_f = np.abs(freqs[fftsig.argmax()])
    #print(maxamp, maxamp_f)
    amp_dB = 20*np.log10(amp/maxamp)

    f = plt.figure(figsize=(10,10))
    plt.plot(freqs,amp_dB)
    plt.xlim(0, 1000000)
    plt.title('f vs amps (dB)')
    plt.show()
    
    #global gain
    gain=[]
    for freq in freqs:
        s = 1j * 2 * np.pi * freq #s is the angular frequency in rads/s, converted from Hz, comes from the formula for impedence of a capacitor
        gain.append((H(s, stages=3))) #Formula for gain is 20*log(Vout/Vin)  
        
    gain = np.array(gain)

    f = plt.figure(figsize=(10,10))
    plt.plot(freqs.real, gain.real)
    plt.rcParams.update({'font.size': 15})
    #plt.title('Freqs vs gain')
    plt.xscale('log');plt.xlabel('Frequency (Hz)'); plt.ylabel('Gain')
    plt.grid()
    plt.show()
    
    #global G
    G = amp * gain
    G_abs = np.abs(G)
    w = 2*np.pi*np.array(freqs)
    f = plt.figure(figsize=(10,10))
    plt.plot(freqs.real/1000000,G_abs.real)
    #plt.title('G vs w')
    #plt.xscale('log')
    plt.yscale('log'); plt.ylabel('Amplitude (V)'); plt.xlabel('Frequency (MHz)')
    plt.xlim(0,1)#0E5)#*2*np.pi)
    plt.grid()
    plt.show()

    #global phase
    phase = np.angle(G, deg=True)
    f = plt.figure(figsize=(14,10))
    plt.rcParams.update({'font.size': 20})
    plt.plot(freqs.real/1000000, phase)
    plt.ylabel('Phase ($^\circ$)')
    plt.xlabel('Frequency (MHz)')#'$\omega$ (Mrads$^{-1}$)')
    #plt.title('phase (deg) vs w')
    plt.xlim(0,0.048)#*2*np.pi)
    plt.show()
    
    # global G_max 
    # global maxf_G
    # G_max = np.max(G)
    # maxf_G = np.abs(freqs[G.argmax()])
    # print(G_max)
    # print(maxf_G)
    
    # global Gabs_max 
    # Gabs_max = np.max(G_abs)
    # global maxf_Gabs
    # maxf_Gabs = np.abs(freqs[G_abs.argmax()])
    # print(Gabs_max)
    # print(maxf_Gabs)

dBdt = [dHdt[i] + dMdt[i] for i in range(len(dHdt))] #dBdt calculated as a combination of applied field and magnetisation differentials

#Plot of dBdt
f = plt.figure(figsize=(10,10))
plt.plot(t, dBdt)
plt.title("dBdt vs t")
plt.show()

#combine dMdt and dHdt for use in Faraday's law of induction
numcoil_1 = 144.
radius = (24*10**(-3))/2 #m
area = np.pi*radius**2 #m^2
n1A = numcoil_1*area
numcoil_2 = 143
n2A = numcoil_2*area

emf1 = [-mu_0*numcoil_1*area * dBdt[i] for i in range(len(dBdt))]

sup = (numcoil_1-numcoil_2)/numcoil_1
exp_fac = np.exp(1j*np.pi/1000 )

emf2 = [-mu_0*numcoil_2*area*dHdt[i] for i in range(len(dBdt))]
emf2 = [emf2[i]*exp_fac for i in range(len(emf2))]

emf = [emf1[i] - emf2[i] for i in range(len(emf1)) ]

f = plt.figure(figsize=(10,10))
plt.plot(t*1000,emf)
#plt.title("emf vs t")
plt.ylabel(' emf (V)')
plt.xlabel('time (ms)')
plt.show()

FFT(emf)


