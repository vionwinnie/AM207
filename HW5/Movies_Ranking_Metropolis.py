# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 00:53:34 2017

"""

import itertools
from operator import eq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

result  = list(itertools.permutations([1,2,3,4,5], 5))
distance = []
orgi = [1,2,3,4,5]

for result_list in result:
    distance.append(sum(map(eq, orgi, result_list)))

    
p = lambda lambda1,dist: np.exp(-lambda1*dist)

####Changing the lambda values here ####
p_distance = [p(0.5,higgin)/63.766 for higgin in distance]
plt.plot(range(0,120),p_distance,'o')
plt.title('Probability of Each Combination at $\lambda$ = 0.5')
plt.ylabel('P(x)')
plt.xlabel('Combination Index')

###Defining a one-step markov chain   ###   
def prop_draw(ifrom):
    u = np.random.uniform()
   
    if ifrom ==119:
        if u < 0.5:
            ito=118
            print('yay')
        else:
            ito=119
            print('nah')
    elif ifrom ==0: 
        if u< 0.5:
            ito=0
        else:
            ito=1
    else:
        if u < 0.5:
            ito = ifrom -1
        else:
            ito = ifrom + 1
            
    return ito

def metropolis(pdf, qdraw, nsamp, xinit):
    samples=np.empty(nsamp)
    x_prev = xinit
    accepted=0
    for i in range(nsamp):
        
        x_star = qdraw(x_prev)
        p_star = pdf[x_star]
        p_prev = pdf[x_prev]
        pdfratio = p_star/p_prev
        if np.random.uniform() < min(1, pdfratio):
            samples[i] = x_star
            x_prev = x_star
            accepted+=1
        else:#we always get a sample
            samples[i]= x_prev
            
    return samples, accepted
    
samps, acc = metropolis(p_distance, prop_draw, 50000, 30)
df = pd.Series(samps)

##Frequency Count### 
plt.figure()
df.value_counts().plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Combination Index')
plt.title('Frequency of each combo from Metropolis algo')

####Count Shawshank Redemption as top at each draw###
shawshank_boolean = []
for samp in samps:
    shawshank_boolean.append(result[int(samp)][0]==1)

print('probability that shawshank is top-ranked is',sum(shawshank_boolean)*0.0005 )

plt.figure()
plt.plot(samps)
plt.xlabel('n_sample')
plt.ylabel('Combination Index')
plt.title('Frequency of each combo from Metropolis algo')
