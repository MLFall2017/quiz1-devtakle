# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.decomposition import PCA as sklPCA
import pandas as pd
import numpy as np
from numpy import genfromtxt as GFT
inputData = GFT('SCLC_study_output_filtered.csv', skip_header=1, delimiter=',')
inputData = np.delete(inputData,0,axis=1)
num_cols = len(inputData[0])
print num_cols
tot_var = 0
for i in range(num_cols):
    tot_var = tot_var + np.var(inputData[:,i])




#the function ensures zero empirical mean



#print np.var(a)
cov_mat= np.cov(inputData.T)
ei_vals, ei_vecs = np.linalg.eig(cov_mat)

Y = np.dot(inputData,ei_vecs)
print "After PCA on all data we get: "
print Y

new_tot_var = 0;
for i in range(len(Y[0])):
    new_tot_var = new_tot_var + np.var(Y[:,i])
        
print "total variance of original data: ",tot_var
print "total variance of PCA data:" ,new_tot_var
#They are equal 3.27459256785e+12

covariancePC1andPC2 = np.cov(Y[:,0],Y[:,1])

#output is -5.57141426e-04+0.j
ei_pairs = [(np.abs(ei_vals[i]), ei_vecs[:,i]) for i in range(len(ei_vals))]
#sorting them in descending order
print('Eigenvalues in descending order:')
print sorted(ei_vals, reverse=True)

#from this we can tell how many principal components to choose
#it gives us the cumulative variance starting from the first PC
tot = sum(ei_vals)
var_exp = [(i / tot)*100 for i in sorted(ei_vals, reverse=True)]
cumulative = np.cumsum(var_exp)
print cumulative 
#we get 75% of data by using first 4 PCs


    
#standardise data

for i in range(len(inputData)):
    inputData[i] = inputData[i] - np.mean(inputData[i])
#perform PCA again
cov_mat= np.cov(inputData.T)
ei_vals, ei_vecs = np.linalg.eig(cov_mat)

Y = np.dot(inputData,ei_vecs)

new_tot_var = 0;
for i in range(len(Y[0])):
    new_tot_var = new_tot_var + np.var(Y[:,i])
print new_tot_var

#3.00419021439e+12 new value changes slightly

#scores plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('scores plot')
ax.scatter(Y['scores'][:,0], Y['scores'][:,1], color='blue')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.show()