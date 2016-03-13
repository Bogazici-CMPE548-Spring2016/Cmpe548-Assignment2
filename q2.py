import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

A = np.matrix([	[0.25, 	0, 		0, 		0.25, 	0.25, 	0.25],
			   	[0.25, 	0.25,	0,		0.25,	0.25,	0	],
			   	[0.25,	0,		0,		0.5,	0.25,	0	],
			   	[0.25,	0,		0.5,	0,		0.25,	0	],
			   	[0.25,	0.25,	0,		0.25,	0.25,	0	],
			   	[0.25,	0,		0,		0.25,	0.25,	0.25]])

A = np.transpose(A)#Transition matrixi tersten olusturduk

L, B = np.linalg.eig(A)

#print L
#print B

idx = L.argsort()[::-1]   
L = L[idx]
B = B[:,idx]

B_inv = np.linalg.inv(B)
#print B_inv
S = B * np.linalg.matrix_power(np.diag(L),100) * B_inv
print S[:,0]

P = S[:,0]

epsilon = 1e-8
isBalanced = True
for i in range(6):
	for j in range(6):
		LtR = A[i,j] * P[j]
		RtL = A[j,i] * P[i]
		if np.abs(LtR-RtL) > epsilon:
			isBalanced = False
		#print LtR, " - ", RtL

if not isBalanced:
	print "This process does NOT satisfy the detailed balance condition"
else:
	print "This process satisfies the detailed balance condition"


#Pi0 = np.matrix([[1.0/6],[1.0/6],[1.0/6],[1.0/6],[1.0/6],[1.0/6]])
Pi0 = np.matrix([[0],[0],[0.5],[0],[0.5],[0]]);
diff = 1
t = 1


# plt.ion()
# plt.show()
# while diff > epsilon:
# 	A_t = np.linalg.matrix_power(A,t)
# 	Pi = A_t * Pi0
# 	diff = np.linalg.norm(P - Pi, 2)
# 	t += 1
# 	plt.imshow(A_t, interpolation='nearest', cmap = cm.Greys_r)
# 	plt.grid(True)
# 	plt.draw()
# 	time.sleep(0.05)
# TTTT = np.transpose(A[:,1])
# print np.squeeze(np.asarray(A[:,1]))
# print 
# print "Tmix ",t



# print np.squeeze(np.asarray(A[:,1]))
# Arr = np.random.multinomial(1, np.squeeze(np.asarray(A[:,1])))
# print Arr
# index = (Arr==1).argmax()
# print index



###  f
N = 1000
Trajectories = []
Tmix = 30
for i in range(N):
	Xs = []
	X = np.random.randint(6)
	Xs.append(X)
	for t in range(Tmix):
		Arr = np.random.multinomial(1, np.squeeze(np.asarray(A[:,X])))
		X = (Arr==1).argmax()
		Xs.append(X)
	Trajectories.append(Xs)

Histogram = np.zeros((6,Tmix + 1))
for i in range(N):
	for t in range(Tmix + 1):
		Histogram[Trajectories[i][t]][t] += 1. / N

# plt.ion()
# plt.show()
# plt.imshow(Histogram, interpolation='nearest', cmap = cm.Greys_r)
# plt.grid(True)
# plt.show()



###   g
SingleTraj = []
X = np.random.randint(6)
for t in range(Tmix + N):
	Arr = np.random.multinomial(1, np.squeeze(np.asarray(A[:,X])))
	X = (Arr==1).argmax()
	if t > Tmix:
		SingleTraj.append(X)

SingleHist = np.zeros((6,1))
for t in range(N-1):
	SingleHist[SingleTraj[t]] += 1. / N



plt.imshow(np.concatenate((SingleHist,P),axis=1), interpolation='nearest', cmap = cm.Greys_r)
plt.grid(True)
plt.show()
