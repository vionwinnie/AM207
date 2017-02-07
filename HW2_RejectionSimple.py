# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:37:36 2017

@author: Dell
"""

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

#P = lambda x: np.exp(-x)

# domain limits
xmin = 1 # the lower limit of our domain
xmax = 9 # the upper limit of our domain

# range limit (supremum) for y
ymax = 1/3
#you might have to do an optimization to find this.

N = 10000 # the total of samples we wish to generate
accepted = 0 # the number of accepted samples
samples = np.zeros(N)
count = 0 # the total count of proposals

# generation loop
while (accepted < N):
    
    # pick a uniform number on [xmin, xmax) (e.g. 0...10)
    x = np.random.uniform(xmin, xmax)
    
    # pick a uniform number on [0, ymax)
    y = np.random.uniform(0,ymax)
    
    # Do the accept/reject comparison
    if y < f(x):
        samples[accepted] = x
        accepted += 1
    
    count +=1
    
print("Count",count, "Accepted", accepted)

# get the histogram info
hinfo = np.histogram(samples,30)

# plot the histogram
plt.hist(samples,bins=30, label=u'Samples', normed=True);

# plot our (normalized) function
xvals=np.linspace(xmin, xmax, 1000)
yvals = [hinfo[1][0]*f(i) for i in xvals]

plt.plot(xvals, yvals, 'r', label=u'f(x)')

# turn on the legend
plt.legend()

#=====================================================#
#Running Rejection Sampling Using Uniform Distribution
#evaluate E[h(x)]  #
#=====================================================#

#hx
h = lambda x: (1/(3*np.sqrt(2)*np.pi))*np.exp(-(1/18)*((x-5)**2))

MC_RejU = np.zeros(1000)
for k in np.arange(0,1000):
    N = 10000 # the total of samples we wish to generate
    accepted = 0 # the number of accepted samples
    samples = np.zeros(N)
    count = 0 # the total count of proposals

    # generation loop
    while (accepted < N):
        
        # pick a uniform number on [xmin, xmax) (e.g. 0...10)
        x = np.random.uniform(xmin, xmax)
        
        # pick a uniform number on [0, ymax)
        y = np.random.uniform(0,ymax)
        
        # Do the accept/reject comparison
        if y < f(x):
            samples[accepted] = x
            accepted += 1
        
        count +=1

    x_mc = samples
    result = [h(dig) for dig in x_mc]
    MC_RejU[k] = np.mean(result)

print("Mean basic MC estimate using rejection sampling: ", np.mean(MC_RejU))
print("Standard deviation of our estimates: ", np.std(MC_RejU))

plt.figure()
plt.hist(MC_RejU,30, histtype='step', label=u'MC(Rej Unif)');

#################################################################