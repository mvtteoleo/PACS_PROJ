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

import sympy as sp
from sympy.printing.cxx import CXX17CodePrinter

# --- 1. Define Variables and Functions ---
x, y, z, t, Re = sp.symbols("x y z t Re")

# Define pi as a symbol so it prints as "M_PI" exactly as you want
# (Standard sympy pi might print as a number or different constant)
mpi = sp.Symbol("M_PI")

# Helper to scale arguments by pi (replacing sp.pi with our symbol)
def s(val): return sp.sin(mpi * val)
def c(val): return sp.cos(mpi * val)

# Velocity Fields
u = c(x) * s(y) * c(z) * s(t)
v = s(x) * c(y) * c(z) * s(t)
w = 2 * s(x) * s(y) * s(z) * s(t)
p = c(x) * c(y) * c(z)

# --- 2. Define Derivatives and Navier-Stokes Forces ---
# Check continuity (optional, good for verification)
div = sp.diff(u, x) + sp.diff(v, y) + sp.diff(w, z)
# Note: In your specific case, you used real sp.pi for derivatives, 
# but for printing we want M_PI. 
# We perform differentiation first using sp.pi, THEN substitute sp.pi -> M_PI for printing.
u_calc = u.subs(mpi, sp.pi)
v_calc = v.subs(mpi, sp.pi)
w_calc = w.subs(mpi, sp.pi)
p_calc = p.subs(mpi, sp.pi)

def lap(h):
    return (sp.diff(h, x, 2) + sp.diff(h, y, 2) + sp.diff(h, z, 2)) / Re

def material_derivative(f, u, v, w):
    return sp.diff(f, t) + u * sp.diff(f, x) + v * sp.diff(f, y) + w * sp.diff(f, z)

# Calculate Forces (Momentum Equation residuals)
# fx = du/dt + (u.grad)u + grad(p) - (1/Re)Lap(u)
fx_calc = material_derivative(u_calc, u_calc, v_calc, w_calc) + sp.diff(p_calc, x) - lap(u_calc)
fy_calc = material_derivative(v_calc, u_calc, v_calc, w_calc) + sp.diff(p_calc, y) - lap(v_calc)
fz_calc = material_derivative(w_calc, u_calc, v_calc, w_calc) + sp.diff(p_calc, z) - lap(w_calc)

# Substitute sp.pi back to M_PI symbol for the final C++ code
replacements = {sp.pi: mpi}
expressions = {
    "ux": u_calc.subs(replacements),
    "uy": v_calc.subs(replacements),
    "uz": w_calc.subs(replacements),
    "fx": fx_calc.subs(replacements),
    "fy": fy_calc.subs(replacements),
    "fz": fz_calc.subs(replacements)
}

# --- 3. Code Generation with Optimization (CSE) ---

class CustomCXXPrinter(CXX17CodePrinter):
    """Custom printer to ensure float literals have 'M_PI' and clean types."""
    def _print_Symbol(self, expr):
        if expr.name == "M_PI":
            return "std::numbers::pi_v<T>"
        return super()._print_Symbol(expr)

def generate_cpp_function(name, expr, printer):
    """Generates a single C++ template function with CSE optimization."""
    
    # 1. Perform Common Subexpression Elimination (CSE)
    # This finds repeated math (like sin(pi*x)) and creates temp variables
    sub_exprs, simplified_expr = sp.cse(expr)
    
    lines = []
    lines.append("template <typename T>")
    lines.append(f"auto {name}(const T& x, const T& y, const T& z, const T& t, const T& Re) -> T")
    lines.append("{")
    
    # 2. Print optimized temporary variables
    for var, sub_e in sub_exprs:
        # Convert sympy code to C++ code
        c_code = printer.doprint(sub_e)
        lines.append(f"    const T {var} = {c_code};")
        
    # 3. Print return statement
    # If there were no sub-expressions, simplify directly
    final_code = printer.doprint(simplified_expr[0])
    lines.append(f"    return {final_code};")
    lines.append("}")
    lines.append("") # Empty line for spacing
    
    return "\n".join(lines)

# --- 4. Write to File ---
printer = CustomCXXPrinter()

print("Generating manufactured_sols.hpp...")

with open("manufactured_sols.hpp", "w") as f:
    # Header boilerplate
    f.write("#pragma once\n")
    f.write('#include "../include/navier_stokes.hpp"\n')
    f.write("#include <numbers>\n")
    f.write("namespace numPDE{\n")

    # Generate functions
    # We loop through the list in the order you want
    order = ["ux", "uy", "uz", "fx", "fy", "fz"]
    
    for name in order:
        func_code = generate_cpp_function(name, expressions[name], printer)
        f.write(func_code)

with open("manufactured_sols.hpp", "a") as f:
    f.write("}; //end numPDE")

print("Done! Check 'manufactured_sols.hpp'.")

