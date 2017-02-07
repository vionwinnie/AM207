# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 00:08:01 2017

@author: Dell
"""


#setting parameter M as 0.3
mu = 7;
sig =2.5;

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
gvals = [p(i) for i in xvals]
plt.plot(xvals, fvals, 'r', label=u'$f(x)$')
plt.plot(xvals, gvals, 'b', label=u'$g(x)$')
plt.legend()
plt.show()


##hx
h = lambda x: (1/(3*np.sqrt(2)*np.pi))*np.exp(-(1/18)*((x-5)**2))
###
N = 10000 # the total of samples we wish to generate
accepted = 0 # the number of accepted samples
samples = np.zeros(N)
count = 0 # the total count of proposals
#
#
Iis = np.zeros(1000)
#
for k in np.arange(0,1000):
    # DRAW FROM THE GAUSSIAN mean =2 std = sqrt(0.4) 
    xis = mu + sig*np.random.randn(N,1);
    #hist(x)
    xis = xis[ (xis<xmax) & (xis>xmin)]
    fxis = [f(val) for val in xis]
#
    # normalization for gaussian from 1..9
    normal = normfun(9)-normfun(1);
    Iis[k] =np.mean(h(xis)*(fxis)/p(xis))*normal;
#
#
print("Mean basic MC estimate using importance sampling: ", np.mean(Iis))
print("Standard deviation of our estimates: ", np.std(Iis))
#    
### get the histogram info
hinfo = np.histogram(Iis,30)
plt.figure()
plt.hist(Iis,bins=30, label=u'Samples', normed=True);
plt.legend()

