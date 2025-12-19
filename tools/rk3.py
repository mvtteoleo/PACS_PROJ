import numpy as np

# Test the RK stepper
# du/dt = -αu
# u_ex = exp(-alpha * t)

def f(u, t):
    return -u*u + np.sin(t) * np.sin(t) + np.cos(t)

def u_ex(t):
    return np.sin(t)

def RK3(un, t, dt):
    a21 = 64.0/120
    a31 = 30.0/120
    a32 = 50.0/120
    b1 = 30.0/120
    b3 = 90.0/120
    f1 =  f(un, t)
    Y2 = un + a21*dt*f1
    t2 = t + dt * 64.0/120
    t3 = t + dt * 80.0/120
    f2 = f(Y2, t2)
    Y3 = un + dt*(a31*f1 + a32*f2)
    f3 = f(Y3, t3)
    u_new = un + dt*(b1*f1 + b3 * f3)
    return u_new




def testRK(t, t_max, dt):
    u = np.array( t_max // dt +1)
    uex = np.array( t_max // dt +1)

    ts = np.arange(t, t_max, dt)
    u = np.copy(ts)
    u[0] = u_ex(ts[0])

    uex = u_ex(ts)
    for i in range(1, np.size(ts)):
        u[i] = RK3(u[i-1], ts[i-1], dt)

    l2 = np.linalg.norm(u-uex)
    l2 *= np.sqrt(dt) 
    linf = np.max(u-uex)
    return l2, linf

t_max = 1
t=0
dt_0 = 0.1

dts = [dt_0/2**n for n in range(1, 5)]

errL2 = np.copy(dts)
errLinf = np.copy(dts)
for i, dt in enumerate(dts):
    errL2[i], errLinf[i] = testRK(t, t_max, dt)
    if(i>0):
        l2_c = np.log(errL2[i] / errL2[i-1]) / np.log(dts[i] / dts[i-1])
        linf_c = np.log(errLinf[i] / errLinf[i-1]) / np.log(dts[i] / dts[i-1])
        print(f"Conv : {l2_c = }")
        print(f"Conv : {linf_c = }")




import matplotlib.pyplot as plt
plt.loglog(dts, errL2, label="L2")
plt.loglog(dts, errLinf, label="Linf")
plt.loglog(dts, np.power(dts, 2), label="II ref")
plt.loglog(dts, np.power(dts, 3), label="III ref")

plt.legend()
plt.show()
    

        
"""
plt.plot( ts, u, label="rk3")
plt.plot( ts, uex, label="exact")
plt.legend()
plt.show()


"""
      
