# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:42:48 2017

@author: Dell

"""

#Inversion Sampling 
from math import sqrt

plt.figure(figsize=(18,8))  # set the figure size

#P = lambda x: np.exp(-x)
h = lambda x: (1/(3*np.sqrt(2)*np.pi))*np.exp(-(1/18)*((x-5)**2))

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

def fx_cdf(x):
    ans = np.zeros(1)
    if x < 3 and x >=1:
        ans[0] = (((x-1)**2/24)-((1-1)**2/24))
    elif x < 5 and x >=3:
        ans[0] =  1/6 + ((-(x-5)**2/24)-(-(3-5)**2/24))
    elif x < 7 and x >=5:
        ans[0] = 1/3 + (((x-5)**2/12)-(-(5-5)**2/12))
    elif x <=9 and x >= 7:
        ans[0] = 2/3 + ((-(x-9)**2/12)-(-(7-9)**2/12))
    return ans[0]


def invfx_cdf(a):
    root = np.zeros(2)
    if a < 1/6 and a >=0:
        root[0] = sqrt(24*a)+1
        root[1]  = -sqrt(24*a)+1
        root = [elem for elem in root if elem >= 1 and elem <=3]
        
    elif a < 1/3 and a >=1/6:
        root[0] = sqrt(24*(1/3-a))+5
        root[1] =-sqrt(24*(1/3-a))+5
        root = [elem for elem in root if elem >= 3 and elem <=5]
        
    elif a < 2/3 and a >=1/3:
        root[0]  = sqrt(12*(a - 1/3))+5
        root[1]  = -sqrt(12*(a - 1/3))+5
        root = [elem for elem in root if elem >= 5 and elem <=7]

    elif a >= 2/3:
        root[0]  = sqrt(12*(1-a))+9
        root[1]  = -sqrt(12*(1-a))+9   
        root = [elem for elem in root if elem >= 7 and elem <=9] 
    return root[0]    
    


# domain limits
xmin = 1 # the lower limit of our domain
xmax = 9 # the upper limit of our domain


#xvals=np.linspace(xmin, xmax, 1000)
#yvals = [fx_cdf(vak) for vak in xvals]
#plt.plot(xvals, yvals, 'r', label=u'f(x)')

#xvals=np.linspace(0, 1, 1000)
#yvals = [invfx_cdf(vak) for vak in xvals]
#plt.plot(yvals,xvals, 'r', label=u'f(x)')

N = 10000 # the total of samples we wish to generate

# generate uniform samples in our range then invert the CDF
# to get samples of our target distribution
R = np.random.uniform(0, 1, N)
Rvals = [invfx_cdf(run) for run in R]

# get the histogram info
hinfo = np.histogram(Rvals,200)

# plot the histogram
plt.hist(Rvals,bins=200, label=u'Samples', normed=True);

## plot our (normalized) function
xvals=np.linspace(xmin, xmax, 1000)
fxvals = [hinfo[1][0]*f(val) for val in xvals]
plt.plot(xvals, fxvals, 'r', label=u'f(x)')
#
## turn on the legend
plt.legend()


## =============================================================
## MONTE CARLO INTEGRATION WITH SAMPLING FROM INVERSION SAMPLING
## =============================================================
MC_Inv = np.zeros(1000)
for k in np.arange(0,1000):
    N = 10000 # the total of samples we wish to generate
    R = np.random.uniform(0, 1, N)
    Rvals = [invfx_cdf(run) for run in R]
    x_mc = Rvals
    result = [h(dig) for dig in x_mc]
    MC_Inv[k] = np.mean(result)

print("Mean basic MC estimate using rejection sampling: ", np.mean(MC_Inv))
print("Standard deviation of our estimates: ", np.std(MC_Inv))

plt.figure()
plt.hist(MC_Inv,30, histtype='step', label=u'MC(Inversion)');
