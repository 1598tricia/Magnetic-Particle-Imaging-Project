from particle import Particle
from field_gradient import load_field_strength, load_field_gradients, get_field_values
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from MPI_signal import emf, FFT
plt.rcParams.update({'font.size': 25})

width = 0.02
height = 0.025
particles = []

Diameter = 26 * 10**(-7)

simtime = 100 *60; timestep = 0.8
tsteps = simtime/timestep
time = np.linspace(1, simtime, int(tsteps))

H_un = 32981.82497; H_un_grad = H_un/height
f_H = 16000

for i in range(1000):
    particle = Particle(Diameter, 2.64) #diameter and magnetic susceptibility
    particle.set_position(width, height) # of container 
    particles.append(particle)

print('Particles added')

field_strengths = load_field_strength("FEMM Data\\applied_field.txt")
field_grads = load_field_gradients("FEMM Data\\applied_gradients.txt")

ypos = []
xpos = []

#Get x and y positions 
for i in range(len(particles)):   
    xpos2 = []
    ypos2 = []
    
    #Get field values for non-uniform field
    # field_strength, field_gradient = get_field_values(particles[i], field_strengths, field_grads, 0.02, 0.025)
    # particles[i].apply_force(field_strength, field_gradient, 0.000893)
    
    #Field values for uniform field
    particles[i].apply_force(H_un, H_un_grad, 0.0014)
    
    for k in range(len(time)):
        particles[i].move(timestep)
        siv_y1 = (particles[i].velocity[1]) 
        brownian_y = particles[i].brownian_y ; diffusivity = particles[i].Diffusivity
           
        ypos1 = (particles[i].position[1])
        if (particles[i].position[1]) > height: 
            (particles[i].position[1]) = height
            ypos1 = (particles[i].position[1])
        elif (particles[i].position[1]) < 0:
            (particles[i].position[1]) = 0.0
            ypos1 = (particles[i].position[1])
            if (particles[i].position[1]) == 0.0 and brownian_y > 0: 
                (particles[i].position[1]) = 0.0 + brownian_y
                ypos1 = (particles[i].position[1])  
        ypos2.append(ypos1)

        xpos1 = (particles[i].position[0])
        if xpos1 > width:
            (particles[i].position[0]) = width - np.abs(xpos1 - width)
            xpos1 = (particles[i].position[0])
        elif xpos1 < 0:
            (particles[i].position[0]) = 0 + np.abs(xpos1)
            xpos1 = (particles[i].position[0])
        xpos2.append(xpos1)
        
    ypos.append(ypos2) ; xpos.append(xpos2)
ypos = np.array(ypos)
xpos = np.array(xpos)

p = plt.figure(figsize=(10,10))
for i in range(len(particles)):
    plt.plot(xpos[i]*1000,ypos[i]*1000, linewidth=0.2)
plt.ylim(0,height*1000); plt.xlim(0, width*1000)
plt.ylabel('y (mm)') ; plt.xlabel('x (mm)') 
plt.show()

#histogram of particle positions
nbins = 100
y_bin = np.linspace(0, height, nbins) ; x_bin = np.linspace(0, width, nbins)
vol_bin = (height/nbins)*width*0.02 #ybin volume in m^3

tlen = (len(time)-1)/8
times = [0, tlen, 2*tlen, 3*tlen, 4*tlen, 5*tlen, 6*tlen, 7*tlen, (8*tlen)]
times = np.array(times); times = times.astype(int)

for t in times:
    ytest = ypos[:,int(t)] 
    xtest = xpos[:,int(t)]
    
    left, width2 = 0.1, 0.65
    bottom, height2 = 0.1, 0.65
    spacing = 0.03
    
    rect_scatter = [left, bottom, width2, height2]
    rect_histx = [left, bottom + height2 + spacing+0.03, width2, 0.2]
    rect_histy = [left + width2 + spacing, bottom, 0.2, height2]
    
    h = plt.figure(figsize=(10,10))
    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    
    ax_histx = plt.axes(rect_histx); ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy); ax_histy.tick_params(direction='in', labelleft=False)
    
    ax_scatter.scatter(xtest*1000, ytest*1000, s=5)
    ax_histx.hist(xtest*1000, x_bin*1000)
    ax_histy.hist(ytest*1000, y_bin*1000, orientation='horizontal')
    
    ax_scatter.set_xlim((0,width*1000)); ax_scatter.set_ylim((0,height*1000))
    
    ax_histx.set_xlim(ax_scatter.set_xlim()); ax_histy.set_ylim(ax_scatter.set_ylim())
    
    ax_scatter.set_title('At time T = {:.3f}s'.format(time[int(t)]))
    ax_scatter.set_xlabel('x (mm)'); ax_scatter.set_ylabel('y (mm)')
    
    plt.show()
    
# for t in times:
#     ytest = ypos[:,int(t)] 
#     xtest = xpos[:,int(t)]
#     p = plt.figure(figsize=(10,10))
#     plt.ylim(0,height*1000);plt.xlim(0,width*1000)
#     plt.scatter(xtest*1000, ytest*1000, c='b', s=6)
#     plt.ylabel('y (m)'); plt.xlabel('x (mm)')
#     plt.title('At time t = {:.3f}s'.format(time[int(t)]))
#     plt.show()

def linfunc(x, m, c):
    return (m*x)+c

def expfunc(x, a, b, c):
    return a * np.exp(-b * x) + c

def sigmoid(x, L ,x0, k, c):#L - Max y, x0 - mid x value, k - slope gradient
    y = (L / (1 + np.exp(-k*(x-x0))))+c
    return y

y_binmidval = (np.max(y_bin)/len(y_bin))*0.5
ynewbin = y_bin + y_binmidval
ynewbin = ynewbin[1:len(y_bin)-1]

yhist_tally = []
for t in range(len(time)):
    yhist = np.histogram(ypos[:,t], y_bin)
    yhist0 = yhist[0]
    yhist_tally.append(yhist0)
yhist_tally = np.array(yhist_tally)

#Concentration
yconc = yhist_tally*5*10**5/vol_bin

mpit = len(time)/450
mpi_times = np.array([i*mpit for i in range(450)])
mpi_times = mpi_times.astype(int)

mag_t = [np.linspace(0, 0.01, 10000) + time[i] for i in mpi_times]
emf_f = [emf(Diameter, y_bin, time, mpi_times[i], yconc) for i in range(len(mpi_times))]
emff = np.array(emf_f)

mag_t = np.array(mag_t)
# for i in range(len(emf_f)):
#     p = plt.figure(figsize=(13,13))
#     plt.ticklabel_format(useOffset=False, style='plain')
    
#     plt.plot(mag_t[i,:5000], emff[i,:5000], linewidth=3)
#     #plt.title('time t = {:.3f}s'.format(mag_t[i,0]))
#     plt.xlabel('time (s)'); plt.ylabel(r'$\epsilon$ (V)', fontsize=35)
#     plt.show()
    
#     FFT(emf_f[i])


thirdH = np.array([FFT(emf_f[i]) for i in range(len(emf_f))])

p0 = [max(thirdH), simtime/2, 0.01, 0.]


mpi_t = time[mpi_times]

popt, pcov = curve_fit(sigmoid, (mpi_t), thirdH, p0, method='dogbox')
err = np.sqrt(np.diag(pcov))

fitsig = np.transpose(np.array([popt, err]))

sig = sigmoid(mpi_t, popt[0], popt[1], popt[2], popt[3])

p = plt.figure(figsize=(10,10))
plt.plot(mpi_t, thirdH, c='r')
plt.plot(mpi_t,sig, linewidth=2 )
plt.ylabel('Amplitude (V)'); plt.xlabel('Time (s)')
plt.grid()
plt.show()