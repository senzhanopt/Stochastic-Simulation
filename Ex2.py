import numpy as np
import numpy.random as npr
import scipy.stats as stats
import matplotlib.pyplot as plt



### Geometric distribution
p = 1/3
N = 10000
GeoCrude = np.zeros(N)

k = np.arange(1,30,1)
pdfk = stats.geom.pmf(k,p)

c = (1-p)** 0 * p

# Do crude simulation
for i in range(N):
    U = npr.rand()
    X = np.int(np.log(U)/(np.log(1-p))) + 1
    GeoCrude[i] = X

plt.figure()
plt.hist(GeoCrude,density=True,ec='k',bins=range(0,25,1))
plt.plot(k,pdfk,'.-')
plt.title('Geometric with p = 1/3')
plt.show()

### 6-point distribution
p = np.array([7/48,5/48,1/8,1/16,1/4,5/16])
N = 10000
F = np.cumsum(p)

SixCrude = np.zeros(N)


# Do crude simulation
for i in range(N):
    U = npr.rand()
    X = np.nonzero(U < F)[0][0] + 1
    SixCrude[i] = np.int(X)

plt.figure()
plt.hist(SixCrude,density=True,ec='k',bins=range(0,8,1))
plt.plot(np.arange(1,7),p,'.-')
plt.title('Crude pmf')
plt.show()

# Reject sampling
c = np.max(p)
k = 6
ind = 0
SixReject = np.zeros(N)


while SixReject[N-1] == 0:
    I = 1 + np.int(k * npr.rand())
    if npr.rand() <= p[I-1]/c:
        SixReject[ind] = I
        ind += 1


plt.figure()
plt.hist(SixReject,density=True,ec='k',bins=range(0,8,1))
plt.plot(np.arange(1,7),p,'.-')
plt.title('Reject pmf')
plt.show()


# Alias method
F = np.array([1,1/2,15/16,1,1/4,11/16])
L = np.array([1,4,1,4,3,3])
SixAlias = np.zeros(N)

for i in range(N):
    I = 1 + np.int(k * npr.rand())

    if npr.rand() <= F[I-1]:
        SixAlias[i] = I
    else:
        SixAlias[i] = L[I-1]

plt.figure()
plt.hist(SixAlias,density=True,ec='k',bins=range(0,8,1))
plt.plot(np.arange(1,7),p,'.-')
plt.title('Alias pmf')
plt.show()
