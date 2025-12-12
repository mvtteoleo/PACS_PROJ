from sympy import symbols, Eq, Matrix, solve, pprint, simplify

# Define symbolic variables
a, b, c, d, h, p1, p2, p3, g = symbols('a b c d h p1 p2 p3 g')

# I(x) = a*x**2 + b*x + c

def I(h):
    return a*h*h*h + b*h*h + c*h + d 


def dI(h):
    return 3*a*h*h + 2*b*h + c

# Option 1: using equations directly
eq1 = Eq(I(h), p1)
eq2 = Eq(I(2*h), p2)
eq3 = Eq(I(3*h), p3)
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

