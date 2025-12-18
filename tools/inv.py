from sympy import symbols, Eq, Matrix, solve, pprint, simplify

# Define symbolic variables
a, b, c, d, h, p1, p2, p3, g = symbols("a b c d h p1 p2 p3 g")

# I(x) = a*x**2 + b*x + c


def I(h):
    return a * h * h * h + b * h * h + c * h + d


def dI(h):
    return 3 * a * h * h + 2 * b * h + c


# Option 1: using equations directly
eq1 = Eq(I(h), p1)
eq2 = Eq(I(2 * h), p2)
eq3 = Eq(I(3 * h), p3)
# BC to impose
eq4 = Eq(dI(0), g)
# eq4 = Eq(I(0), g)

print("Solving system symbolically:")
pprint([eq1, eq2, eq3])

# Solve the system symbolically
solution = solve((eq1, eq2, eq3, eq4), (a, b, c, d), dict=True)[0]

print("\nSolution:")
pprint(solution)

# Simplify result
x_vec_simplified = simplify(solution)
print("\nSimplified symbolic solution:")
pprint(x_vec_simplified)


import sympy as sp
from sympy import pi

# --- 1. Define Symbolic Variables and Coordinate System ---
x, y, z, t, Re = sp.symbols("x y z t Re")

u = sp.cos(pi * x) * sp.sin(pi * y) * sp.cos(pi * z) * sp.sin(pi * t)
v = sp.sin(pi * x) * sp.cos(pi * y) * sp.cos(pi * z) * sp.sin(pi * t)
w = 2 * sp.sin(pi * x) * sp.sin(pi * y) * sp.sin(pi * z) * sp.sin(pi * t)

if( sp.diff(u, x) + sp.diff(v, y) + sp.diff(w,z) != 0):
    print("The velocity field is not solenoidal")
    exit()

p = sp.cos(pi * x) * sp.cos(pi * y) * sp.cos(pi * z)

def my_lap(h):
    return (sp.diff(h, x, 2) + sp.diff(h, y, 2) + sp.diff(h, z, 2))/Re

fx = (
    sp.diff(u, t)
    + u * sp.diff(u, x)
    + v * sp.diff(u, y)
    + w * sp.diff(u, z)
    + sp.diff(p, x)
    - my_lap(u)
)
fy = (
    sp.diff(v, t)
    + u * (sp.diff(v, x))
    + v * sp.diff(v, y)
    + w * sp.diff(v, z)
    + sp.diff(p, y)
    - my_lap(v)
)
fz = (
    sp.diff(w, t)
    + u * (sp.diff(w, x))
    + v * sp.diff(w, y)
    + w * sp.diff(w, z)
    + sp.diff(p, z)
    - my_lap(w)
)

fx = sp.simplify(fx)
fy = sp.simplify(fy)
fz = sp.simplify(fz)


print("C++ style code")

# a -ppend
# w-rite
# x-create ... If does not exists gets created
with open("manufactured_sols.hpp", "a") as f:
  f.write("namespace numPDE {\n")

print(sp.cxxcode(fx, standard='C++17'))
print(sp.cxxcode(fy, standard='C++17'))
print(sp.cxxcode(fz, standard='C++17'))

with open("manufactured_sols.hpp", "a") as f:
  f.write("namespace numPDE {\n")
