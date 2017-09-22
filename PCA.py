# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.decomposition import PCA as sklPCA
import pandas as pd
import numpy as np
#read data from csv into pandas list
orgX = pd.read_csv('dataset_1.csv')
#individual list for variables
x = orgX.x
y = orgX.y
z = orgX.z
#print variance
print "variance of x is: ",np.var(x) 
print "variance of y is: ",np.var(y)
print "variance of z is: ",np.var(z)
#print covariance
covXY = np.cov(x,y)
covYZ = np.cov(y,z)
print "covariance of x and y is:",covXY[1,0]
print "covariance of y and z is:",covYZ[1,0]
#the function ensures zero empirical mean

def standardise(array) :
    empMean = np.mean(array)
    array[:] = [x - empMean for x in array]
    
standardise(x) 
standardise(y) 
standardise(z)  
X = np.stack((x,y,z),axis = -1)
#print np.var(a)
cov_mat= np.cov(X.T)
ei_vals, ei_vecs = np.linalg.eig(cov_mat)

#print ei_vecs
#coverting to tuples (eigenvalues, eigenvectors)
#ei_pairs = [(np.abs(ei_vals[i]), ei_vecs[:,i]) for i in range(len(ei_vals))]
#sorting them in descending order
#print('Eigenvalues in descending order:')
#for i in ei_pairs:
 #   print(i[0])
#from this we can tell how many principal components to choose
#it gives us the cumulative variance starting from the first PC
#tot = sum(ei_vals)
#var_exp = [(i / tot)*100 for i in sorted(ei_vals, reverse=True)]
#cumulative = np.cumsum(var_exp)
#print cumulative 

#getting final Y
Y = np.dot(X,ei_vecs)
print "After PCA on all data we get: "
print Y


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y[:,0], Y[:,1], Y[:,2])
plt.show()

print "question 3 :"
matrix = np.array([[0,-1],[2,3]])
vals, vecs = np.linalg.eig(matrix)
print "checking values"
print vals
print vecs