# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:50:20 2017

@author: Dell
"""
#gx U~[0,1] x=[1,9]
g = 1/8

def f(x):
    x = float(x)
    if x < 3 and x >=1:
        result = (x-1)/12
    elif x < 5 and x >=3:
        result =  -(x-5)/12
    elif x < 7 and x >=5:
        result = (x-5)/6
    elif x <=9 and x >= 7:
        result = -(x-9)/6
    return result


# domain limits
xmin = 1 # the lower limit of our domain
xmax = 9 # the upper limit of our domain

# plot our (normalized) function
xvals=np.linspace(xmin, xmax, 1000)
fvals = [f(i) for i in xvals]
gvals = [g for i in xvals]
dvals = [fvals[i]/gvals[i] for i in range(len(gvals))]
plt.plot(xvals, fvals, 'r', label=u'$f(x)$')
plt.plot(xvals, gvals, 'b', label=u'$g(x)$')
plt.plot(xvals, dvals, 'b', label=u'$f(x)/g(x)$')

plt.legend()
plt.show()

###
N = 10000 # the total of samples we wish to generate
accepted = 0 # the number of accepted samples
samples = np.zeros(N)
count = 0 # the total count of proposals
#
### generation loop
while (accepted < N):
    
    # Sample from g using inverse sampling
    xproposal = np.random.uniform(xmin, xmax)      
    samples[accepted] = xproposal
    accepted += 1        
    count +=1

    
#
### get the histogram info
hinfo = np.histogram(samples,30)
#
## plot the histogram
plt.figure()
plt.hist(samples,bins=30, label=u'Samples', normed=True);
#
## plot our (normalized) function
xvals=np.linspace(xmin, xmax, 1000)
yvals = [hinfo[1][0]*f(i) for i in xvals]
##
plt.plot(xvals, yvals, 'r', label=u'$f(x)$')
##
### turn on the legend
plt.legend()
#
##=====================================================#
##Running Importance Sampling Using Uniform Distribution
##evaluate E[h(x)]#
##=====================================================#

#hx
h = lambda x: (1/(3*np.sqrt(2)*np.pi))*np.exp(-(1/18)*((x-5)**2))
#
MC_ImpU = np.zeros(1000)
for k in np.arange(0,1000):
    N = 10000 # the total of samples we wish to generate
    accepted = 0 # the number of accepted samples
    samples = np.zeros(N)
    count = 0 # the total count of proposals

    # generation loop
    while (accepted < N):
        
        # Sample from g using inverse sampling
        xproposal = np.random.uniform(xmin, xmax)
        samples[accepted] = xproposal
        accepted += 1    
        count +=1

    x_mc = samples
    result = [h(dig)*f(dig)/g for dig in x_mc]
    MC_ImpU[k] = np.mean(result)
#
print("Mean basic MC estimate using important sampling: ", np.mean(MC_ImpU))
print("Standard deviation of our estimates: ", np.std(MC_ImpU))
#
plt.figure()
plt.hist(MC_ImpU,30, histtype='step', label=u'MC(Imp Uni)');
#
##################################################################
