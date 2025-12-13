import numpy as np
import matplotlib.pyplot as plt


def poiss_mat(N):
    """
    Creates an N x N tridiagonal matrix with the pattern:
    1 on the superdiagonal (k=1)
    -2 on the main diagonal (k=0)
    1 on the subdiagonal (k=-1)

    Args:
        N (int): The size of the square matrix (N x N).

    Returns:
        numpy.ndarray: The resulting tridiagonal matrix.
    """
    if N < 2:
        # Handle small cases explicitly
        if N == 1:
            return np.array([[-2]])
        return np.array([])

    main_diag = -2 * np.ones(N)

    super_diag = np.ones(N - 1)

    sub_diag = np.ones(N - 1)

    A = np.diag(main_diag, k=0) + np.diag(super_diag, k=1) + np.diag(sub_diag, k=-1)

    return A


# N = N_points - 2 (Internal only))
def err_poiss(N) -> float:
    h = np.pi * 2 / (N - 1)

    # Impose the BC Using a Polinomial.
    # Retrieve the ghost point value by fitting a second order polinomial
    # such that I(h) = u_1; I(2h) = u_2 and I'(0) = G;
    # Retrive: 3u_0 - 4u_1 + u_2 = -2h * G
    # The scheme is then modified to impose u_0 = 4/3u_1 - 1/3u_2 - 2hG/3
    A = poiss_mat(N)

    def forc(x):
        return -np.cos(x)

    x = [h * i for i in range(0, N + 2)]
    b = [h * h * forc(i) for i in x[1:-1]]
    x_ex = [np.cos(i) for i in x[1:-1]]

    # Impose Neumann on x=0 side on A and b
    A[0, 0] += 4.0 / 3.0
    A[0, 1] -= 1.0 / 3.0

    G = 0
    b[0] += 2 * h * G
    """
    # Impose the Neumann BC on x=2pi
    b[0] -= x_ex[0]
    """

    b[N - 1] -= x_ex[-1]

    x_h = np.linalg.solve(A, b)
    plt.plot(x[1:-1], (x_h - x_ex), label="numrical")
    plt.show()
    err_L2 = np.linalg.norm(x_h - x_ex)
    err_L2 *= np.sqrt(h)

    print(f"{err_L2 = }")
    return err_L2


ns = np.array([10, 20, 40, 80, 160])

err = np.array([10, 20, 40, 80, 160])
for i, N in enumerate(ns):
    err[i] = err_poiss(N)

print(err)


"""
plt.plot(x[1:-1], x_h, label="numrical")
plt.plot(x[1:-1], x_ex, label="exact")
plt.legend()
plt.show()

plt.plot(x[1:-1], x_h - x_ex, label="error")
plt.legend()
plt.show()
"""
