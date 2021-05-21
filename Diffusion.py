from particle import Particle
from field_gradient import load_field_strength, load_field_gradients, get_field_values
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})
from scipy.optimize import curve_fit

width = 0.02
height = 0.025
particles = []

for i in range(500):
    particle = Particle(26 * 10**(-7), 2.64) #diameter and magnetic susceptibility
    particle.set_position(width, height) # of container 
    particles.append(particle)

print('Particles added')

field_strengths = load_field_strength("FEMM Data\\applied_field.txt")
field_grads = load_field_gradients("FEMM Data\\applied_gradients.txt")

simtime = 100*60.; timestep = 0.8
tsteps = simtime/timestep
time = np.linspace(1, simtime, int(tsteps)); 
no_particles = []
ypos = []
xpos = []

#Get x and y positions 
for i in range(len(particles)):   
    xpos2 = []
    ypos2 = []

    # field_strength, field_gradient = get_field_values(particles[i], field_strengths, field_grads, 0.02, 0.025)
    # particles[i].apply_force(field_strength, field_gradient, 0.0010049)
    
    particles[i].apply_force(32981.82497, 0., 0.0014) #field strength, field gradient, viscosity 

    for interval in time:
        particles[i].move(timestep)
        v_y1 = (particles[i].velocity[1]) ; v_x1 = (particles[i].velocity[0])
        brownian_y = particles[i].brownian_y ; diffusivity = particles[i].Diffusivity
        
        ypos1 = (particles[i].position[1]); 
        if (particles[i].position[1]) > height: #and brownian_y < 0:
            (particles[i].position[1]) = height# + brownian_y
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

def gaussfunc(x, a, b, c):
    return a * np.exp(-0.5*((x-b)**2)/(c**2))

###DIFUSSION#################
p0 = [[500, 0.0125, 0.0001], [50, 0.0125, 0.001], [30, 0.0125, 0.002], [30, 0.0125, 0.002],
      [20, 0.0125, 0.003], [20, 0.0125, 0.003], [20, 0.0125, 0.004], [15, 0.0125, 0.004],
      [[15, 0.0125, 0.004]]]
gauss_coeff = []; gerr = []; gauss_coeff_err = []
for t in range(len(times)):
    popt, pcov = curve_fit(gaussfunc, ynewbin, yhist_tally[t,:], p0=p0[t])
    gauss_coeff.append(popt)
    gerr = np.sqrt(np.diag(pcov))
    gauss_coeff_err.append(gerr)

gauss_coeff =np.array(gauss_coeff) ; gauss_coeff_err = np.array(gauss_coeff_err)
p = plt.figure(figsize=(10,10))
for t in range(len(times)):
    plt.plot(ynewbin, gaussfunc(ynewbin, gauss_coeff[t,0], gauss_coeff[t,1], gauss_coeff[t,2]), 
              label="T = {:.3f}s".format(times[t]*timestep))
plt.xlabel('y (m)'); plt.ylabel('number of particles')
plt.legend(loc="upper left")
plt.show()

tuse = np.sqrt(time[times])
sigvals = np.abs((gauss_coeff[:,2]))#/(100)))
sigvalserr = np.abs(gauss_coeff_err[:,2])

for t in range(len(times)):
    ytest = [ypos[:,int(n)] for n in times]
    ytest = np.array(ytest)
    
    p = plt.figure(figsize=(10,10))
    plt.plot(ynewbin*1000, gaussfunc(ynewbin, gauss_coeff[t,0], gauss_coeff[t,1], gauss_coeff[t,2]))
    plt.hist(ytest[t,:]*1000, y_bin*1000)
    #plt.plot(ynewbin, yhist_tally[t,:])
    plt.ylabel('Number of Particles');plt.xlabel('Height (mm)')
    plt.show()

popt, pcov = curve_fit(linfunc, tuse, sigvals, sigma=sigvalserr)
plotdiffusivity = ((popt[0]**2)/2)/10000
plotffdiffu_err = popt[0] * np.sqrt(pcov[0,0])/10000
p = plt.figure(figsize=(10,10))
plt.plot(tuse, linfunc(tuse, popt[0], popt[1])*1000)#gradient should equal 2*diffusivity
plt.errorbar(tuse, sigvals*1000, yerr=sigvalserr*1000, fmt='o', c='r')
plt.xlabel(r'$t^{1/2}$ ($s^{1/2}$)'); plt.ylabel(r'$\sigma$ ($mm$)')
plt.show()

####With gravity#################################################################################
def trap(x, a, b, c, tau1, tau2, tau3):
    y = np.zeros(len(x))
    y[int(tau1):int(tau2)] = a #a = value of flat area, tau1 = flat area ends
    y[int(tau2):int(tau3)] = b*x[int(tau2):int(tau3)] + c #tau2 = downhill slope begins
    y[int(tau3):] = 0
    return y
