import matplotlib.pyplot as plt

N = [i for i in range(3, int(1e3))]

ns = [i**3 for i in N]
nopt = [(i-2)**3 for i in N]

diff =  [(1 - (i-2)**3/i**3) for i in N]

#plt.plot(N, ns)
#plt.plot(N, nopt)
plt.plot(N, diff)
plt.show()
