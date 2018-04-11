## Juan Carlos Tapia Flores
## 14133

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy
from matplotlib.patches import Ellipse


## Funcion que genera los elipses
def plot_cov_ellipse(cov, pos, a=1, col='blue', nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, alpha=a, color=col , **kwargs)

    ax.add_artist(ellip)
    return ellip


def leerArchivo(s):
    file = open(s, "r")
    points = []

    for line in file:
        p = line[1:-2].split(",")
        point = [[float(p[0])],[float(p[1])]]

        
        points.append(point)
    
    file.close()
    return points

def generarGaussianos(K):
    mu = []
    sigma = []
    pi = [1, 0.60, 0.20, 0.20, 0.20]

    for i in range(0, K):
        u = np.random.rand(2,1)*2
        E = np.random.rand(2,2)*2
        mu.append(u)
        sigma.append(E)
        
    return pi, mu, sigma

def getE(point, i, pi, mu, sigma):
    e = pi[i] * math.pow(2*math.pi , -1) * math.pow(abs(np.linalg.det(sigma[i])), -0.5)
    e *= math.exp(-0.5 * ( np.dot(np.dot(np.transpose(np.array(point) - mu[i]) , np.linalg.inv(sigma[i])) ,  np.array(point) - mu[i]) ))
    return e

def WhatCluster(point, K, pi, mu, sigma, e):
    p = 0.0
    k = 0
    point = [[point[0]], [point[1]]]

    
    for i in range(0, K):
        R = 0
        for j in range(0, len(points)):
            R += e[i][j]
        p_i = getE(point, i , pi, mu, sigma) / R
        print p_i
        if (p_i > p ):
            p = p_i
            k = i
                
    return k

def Iteraciones(K, pi, mu, sigma, Iter):
    sigma_old = copy.copy(sigma)
    mu_old = copy.copy(mu)
    e = []
    for k in range(0, Iter):
        print k+1
        e = []
        for i in range(0, K):
            ee = []
            for j in points:
                ee.append(0)
            e.append(ee) 

        #Paso E
        for j in range(0, len(points)):
            R = 0
            for i in range(0, K):
    #            print sigma[i] 
                e[i][j] = getE(points[j], i , pi, mu, sigma)
                R += e[i][j]
            for i in range(0, K):
                e[i][j] = e[i][j] / R

        # Paso M


        for i in range(0, K):
            p1 = 0
            for j in range(0, len(points)):
                p1 +=e[i][j] 
            pi[i] = p1 / len(points)


        for i in range(0, K):
            p1 = 0
            for j in range(0, len(points)):
                p1 +=e[i][j] *  np.array(points[j])

            p2 = 0
            for j in range(0, len(points)):
                p2 +=e[i][j]

            mu[i] = p1 / p2


        for i in range(0, K):
            p1 = 0
            for j in range(0, len(points)):
                p1 +=e[i][j] * np.dot( (np.array(points[j]) - mu[i]) , np.transpose(np.array(points[j]) - mu[i]) )
                
            p2 = 0
            for j in range(0, len(points)):
                p2 +=e[i][j] 
            sigma[i] = p1 / p2

            
        
    return pi, mu, sigma

K = 5
N = 2


points = leerArchivo("test_gmm_4.txt")

pi, mu, sigma = generarGaussianos(K)

e = []
for i in range(0, K):
    ee = []
    for j in points:
        ee.append(0)
    e.append(ee) 



#print np.random.random_sample(3)



Iter = 100
pi, mu, sigma = Iteraciones(K, pi, mu, sigma, Iter)





print "Dibujando puntos"    
for i in range(0, K):
    m = [mu[i][0][0] , mu[i][1][0]]
 
    plt.plot(m[0], m[1], 'x', markersize=5)



for point in points:
    plt.plot(point[0][0], point[1][0], 'bo', markersize=1)



colors = ["yellow","gray","orange","green", "red", "brown"]




print "Dibujando Clusters"
for i in range(0, K):
    print "____"+colors[i] +"_____"
    print "mu"
    print mu[i]

    print "sigma"
    print sigma[i]
    print ""
    m = [mu[i][0][0] , mu[i][1][0]]
    
    x, y = np.random.multivariate_normal(m, sigma[i], 10000).T
    plt.plot(m[0], m[1], 'ro', markersize=10)
    plot_cov_ellipse(sigma[i], m, 0.8,colors[i], 2)
    plot_cov_ellipse(sigma[i], m, 0.5, colors[i],  2.5)
    plot_cov_ellipse(sigma[i], m, 0.5, colors[i],  3)


plt.axis('equal')

plt.show()


while True:

    for i in range(0, K):
        m = [mu[i][0][0] , mu[i][1][0]]
        
        x, y = np.random.multivariate_normal(m, sigma[i], 10000).T
        plt.plot(m[0], m[1], 'ro', markersize=10)
        plot_cov_ellipse(sigma[i], m, 0.8,colors[i], 2)
        plot_cov_ellipse(sigma[i], m, 0.5, colors[i],  2.5)
        plot_cov_ellipse(sigma[i], m, 0.5, colors[i],  3)
    
    point = input("ingrese un punto [a,b]:")

    k_p = WhatCluster(point, K, pi, mu, sigma, e)
    plt.plot(point[0], point[1], 'go', markersize=10)

    print ""
    print "El punto pertenece al cluster " + colors[k_p]
    print "mu"
    print mu[k_p]
    print "sigma"
    print sigma[k_p]
    plt.axis('equal')
    plt.show()
