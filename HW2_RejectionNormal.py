# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:50:20 2017

@author: Dell
"""
#setting parameter M as 0.3
mu = 5;
sig =3;
M = 3.3
#gx
p = lambda x: (1/np.sqrt(2*np.pi*sig**2))*np.exp(-(x-mu)**2/(2.0*sig**2))
normfun = lambda x:  norm.cdf(x-mu, scale=sig)

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
gvals = [M*p(i) for i in xvals]
plt.plot(xvals, fvals, 'r', label=u'$f(x)$')
plt.plot(xvals, gvals, 'b', label=u'$g(x)$')
plt.legend()
plt.show()


### range limits for inverse sampling
umin = normfun(xmin)
umax = normfun(xmax)

##
N = 10000 # the total of samples we wish to generate
accepted = 0 # the number of accepted samples
samples = np.zeros(N)
count = 0 # the total count of proposals

## generation loop
while (accepted < N):
    
    # Sample from g using inverse sampling
    u = np.random.uniform(umin, umax)
    xproposal = mu + sig*norm.ppf(u)
    # pick a uniform number on [0, 1)
    y = np.random.uniform(0,1)
    
    
    # Do the accept/reject comparison
    if y < f(xproposal)/(M*p(xproposal)):
        samples[accepted] = xproposal
        accepted += 1    

    
    count +=1
    
print("Count", count, "Accepted", accepted)

## get the histogram info
hinfo = np.histogram(samples,30)

# plot the histogram
plt.figure()
plt.hist(samples,bins=30, label=u'Samples', normed=True);

# plot our (normalized) function
xvals=np.linspace(xmin, xmax, 1000)
yvals = [hinfo[1][0]*f(i) for i in xvals]
#
plt.plot(xvals, yvals, 'r', label=u'$f(x)$')
#
## turn on the legend
plt.legend()

#=====================================================#
#Running Rejection Sampling Using Normal Distribution
#evaluate E[h(x)]  #
#=====================================================#

#hx
h = lambda x: (1/(3*np.sqrt(2)*np.pi))*np.exp(-(1/18)*((x-5)**2))

MC_RejS = np.zeros(1000)
for k in np.arange(0,1000):
    print(k)
    N = 10000 # the total of samples we wish to generate
    accepted = 0 # the number of accepted samples
    samples = np.zeros(N)
    count = 0 # the total count of proposals

    # generation loop
    while (accepted < N):
        
        # Sample from g using inverse sampling
        u = np.random.uniform(umin, umax)
        xproposal = mu + sig*norm.ppf(u)
        # pick a uniform number on [0, 1)
        y = np.random.uniform(0,1)
        
        
        # Do the accept/reject comparison
        if y < f(xproposal)/(M*p(xproposal)):
            samples[accepted] = xproposal
            accepted += 1    

    
        count +=1


    x_mc = samples
    result = [h(dig) for dig in x_mc]
    MC_RejU[k] = np.mean(result)

print("Mean basic MC estimate using rejection sampling: ", np.mean(MC_RejS))
print("Standard deviation of our estimates: ", np.std(MC_RejS))

plt.figure()
plt.hist(MC_RejS,30, histtype='step', label=u'MC(Rej Norm)');

#################################################################
