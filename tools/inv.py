from sympy import symbols, Eq, Matrix, solve, pprint, simplify

# Define symbolic variables
a, c, h, p1, p2, g = symbols('a c h p1 p2 g')

# Option 1: using equations directly
eq1 = Eq(a*h*h + g*h + c, p1)
eq2 = Eq(4*a*h*h + g*2*h + c, p2)

print("Solving system symbolically:")
pprint([eq1, eq2])

# Solve the system symbolically
solution = solve((eq1, eq2), (a, c), dict=True)[0]

print("\nSolution:")
pprint(solution)

# Simplify result
x_vec_simplified = simplify(solution)
print("\nSimplified symbolic solution:")
pprint(x_vec_simplified)

