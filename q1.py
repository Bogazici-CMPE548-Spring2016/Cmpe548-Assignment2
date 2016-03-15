import numpy as np
from numpy import *
import math
import matplotlib.pyplot as plt

N = 1000

mu = np.array([0, 0])
cov = np.matrix([[0.1, 0], [0, 0.1]])  # diagonal covariance

# this function is from stackoverflow
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def p_norm(x,y,p):
	return np.power(np.power(np.abs(x),p) + np.power(np.abs(y),p), 1.0/p)

def weightFunction(x, y):
	if p_norm(x,y,0.5)<=1:
		return 1.0 / norm_pdf_multivariate(np.array([x, y]), mu, cov)
	else:
		return 0

def estimateArea():
	x, y = np.random.multivariate_normal(mu, cov, N).T

	W = []
	for i in range(N):
		W.append(weightFunction(x[i],y[i]))
	Z = np.mean(W)
	return Z

Zs = []
for i in range(10):
	z = estimateArea()
	Zs.append(z)

print "My proposal distribution is a multivariate gaussian"
print "with mean = ", mu, " and covariance = "
print cov

print "    Mean of my estimate: ", np.mean(Zs)
print "Variance of my estimate: ", np.var(Zs)

#plt.plot(x, y, 'x')
#plt.axis('equal')
#plt.show()