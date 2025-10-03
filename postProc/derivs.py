import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

if 1:
    R = 0.5
    sigma = R / 3
    k = 134
    L = 2

    pTilde = lambda x: np.sin(x*2*np.pi/L)
    lapPtilde = lambda x:      2*np.pi/L  * np.sin(x*2*np.pi/L)


    x0 = 1
    gauss = lambda x: (1 - np.exp(-((x - x0) ** 2) / (2 * sigma**2))) ** 9

    logis = lambda x: 1.0 / (1.0 + np.exp(-k * (np.abs((x - x0)) - R)))

    th = lambda x: 0.5 * (1.0 + np.tanh( k * ((x - x0) ** 2 - R**2)  ))

    xx = np.linspace(0, L, 1000)
    plt.vlines(x0 - R, 0, 1)
    plt.vlines(x0 + R, 0, 1)
    plt.plot(xx, pTilde(xx), label="pTilde")
    plt.plot(xx, lapPtilde(xx), label="lap(pTilde)")
    plt.plot(xx, gauss(xx)* lapPtilde(xx), label="Gaussian step")
    ## plt.plot(xx, logis(xx), label="Logistic step")
    ## plt.plot(xx, th(xx), "o", label="Tanh step")
    plt.grid()
    plt.legend()
    plt.show()
# Symbols
exit()

if 1:
    x, y, z = sp.symbols("x y z", real=True)
    x0, y0, z0 = sp.symbols("x0 y0 z0", real=True)
    R, k = sp.symbols("R k", real=True)

    r2 = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
    f = 0.5 * (1.0 + sp.tanh(k * (r2 - R**2)))

    print("\nLaplacian Δf(x,y,z) =")
    print(sp.pretty(f))

    print("Starting to differentiate")

    # Compute Laplacian
    lap_f = sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)
    print("Finished to differentiate")
    print("\nLaplacian Δf(x,y,z) =")
    print(lap_f)
    u = k*(r2 - R)
    check = sp.simplify(k*(sp.tanh(u - 1)*sp.tanh(u - 1)) * (4*k*r2 * sp.tanh(u) -3))
    print("CHECK")

    lap_f_simplified = sp.pretty(sp.simplify(lap_f))

    print("Manufactured f(x,y,z) =")
    print(f)
    print("\nLaplacian Δf(x,y,z) =")
    print(lap_f_simplified)
