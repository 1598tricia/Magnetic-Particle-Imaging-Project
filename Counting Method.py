from particle import Particle
from field_gradient import load_field_strength, load_field_gradients, get_field_values
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 25})

width = 0.02
height = 0.025
particles = []

Diameter = 26 * 10**(-7)

simtime = 100 *60; timestep = 0.8#0.0019
tsteps = simtime/timestep
time = np.linspace(1, simtime, int(tsteps))

# H_amp = 200000; 
H_un = 32981.82497; H_un_grad = H_un/height
f_H = 16000

for i in range(1000):#1000):
    particle = Particle(Diameter, 2.64) #diameter and magnetic susceptibility
    particle.set_position(width, height) # of container 
    particles.append(particle)

print('Particles added')

field_strengths = load_field_strength("FEMM Data\\applied_field.txt")
field_grads = load_field_gradients("FEMM Data\\applied_gradients.txt")

no_particles = []
ypos = []
xpos = []

#Get x and y positions 
for i in range(len(particles)):   
    xpos2 = []
    ypos2 = []

    # field_strength, field_gradient = get_field_values(particles[i], field_strengths, field_grads, 0.02, 0.025)
    # particles[i].apply_force(field_strength, field_gradient, 0.004814)
    
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

#Check for values outside boundaries
# negval = 0
# for t in range(len(time)):
#     for k in range(len(particles)):
#         if xpos[k,t] > width or xpos[k,t] < 0:
#             print(xpos[k,t])
#             negval += 1        
# print(negval)

# ynegval = 0
# for t in range(len(time)):
#     for k in range(len(particles)):
#         if ypos[k,t] < 0 or ypos[k,t] > height:
#             ynegval += 1
# print(ynegval)

#histogram of particle positions
nbins = 100
y_bin = np.linspace(0, height, nbins) ; x_bin = np.linspace(0, width, nbins)
area_ybin = (height/nbins)*width*0.02*10**9 #ybin area in mm^2

tlen = (len(time)-1)/8
times = [0, tlen, 2*tlen, 3*tlen, 4*tlen, 5*tlen, 6*tlen, 7*tlen, (8*tlen)]
times = np.array(times); times = times.astype(int)
#[0, 40, 75, 100, 200, 500, 800, 4500]

for t in times:
    ytest = ypos[:,int(t)] 
    xtest = xpos[:,int(t)]
    
    left, width2 = 0.1, 0.65
    bottom, height2 = 0.1, 0.65
    spacing = 0.03
    
    rect_scatter = [left, bottom, width2, height2]
    rect_histx = [left, bottom + height2 + spacing, width2, 0.2]
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
    #ax_histx.set_title('x histogram'); ax_histy.set_title('y histogram', rotation='vertical')
    
    ax_scatter.set_title('At time T = {:.3f}s'.format(time[int(t)]))
    ax_scatter.set_xlabel('x (mm)'); ax_scatter.set_ylabel('y (mm)')
    
    plt.show()

def linfunc(x, m, c):
    return (m*x)+c

def expfunc(x, a, b, c):
    return a * np.exp(-b * x) + c

y_binmidval = (np.max(y_bin)/len(y_bin))*0.5
ynewbin = y_bin + y_binmidval
ynewbin = ynewbin[1:len(y_bin)-1]

yhist_tally = []
for t in times:
    yhist = np.histogram(ypos[:,int(t)], y_bin)
    yhist0 = yhist[0]
    yhist_tally.append(yhist0)
yhist_tally = np.array(yhist_tally)
yhist_tally = yhist_tally[:, 1:]

#Concentration
yconc = yhist_tally/area_ybin

yall = []
for t in range(len(time)):
    yset = np.histogram(ypos[:,t], y_bin)
    yall.append(yset[0])
yall = np.array(yall)
ybottom = yall[:,0]

ybottomconc = ybottom/area_ybin
def sigmoid(x, L ,x0, k):#L - Max y, x0 - mid x value, k - slope gradient
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

p0 = [max(ybottomconc), simtime/3, 0.001]

popt, pcov = curve_fit(sigmoid, time, ybottomconc,p0, method='dogbox')
errp = np.sqrt(np.diag(pcov))
p = plt.figure(figsize=(10,10))
plt.scatter(time, ybottomconc, s=1, label='Data')
plt.plot(time, sigmoid(time, popt[0], popt[1], popt[2]), c='r', label='Fit')
plt.ylabel('$Particle Concentration (mm^{-3})$')
plt.xlabel('$Time (s)$')
plt.grid()
#plt.legend()
plt.show()

def sig2(x):
    y = popt[0] / (1 + np.exp(-popt[2]*(x-popt[1])))
    return (y)

def gaussfunc(x, a, b, c):
    return a * np.exp(-0.5*((x-b)**2)/(c**2))

from scipy.misc import derivative
dPcdt = []
for t in time:
    dPcdt.append(derivative(sig2, t, dx=timestep))

p0g = [0.006, popt[1], 10]
poptg, pcovg = curve_fit(gaussfunc, time, dPcdt, p0g)
errpg = np.sqrt(np.diag(pcovg))
p = plt.figure(figsize=(10,10))
plt.scatter(time, dPcdt, c='r', s=1, label='dC/dt')
plt.plot(time, gaussfunc(time, poptg[0], poptg[1], poptg[2]), label='Fitted dC/dt')
#plt.legend()
plt.ylabel('dC/dt'); plt.xlabel('Time (s)')
plt.show()  

siggauss = [popt, errp, poptg, errpg]; siggauss = np.transpose(np.array(siggauss))
