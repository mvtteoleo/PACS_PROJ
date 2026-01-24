import sympy as sp
from sympy.printing.cxx import CXX17CodePrinter

destination =  "tools/manufactured_sols.hpp" 


# --- 1. Define Variables and Functions ---
x, y, z, t, Re = sp.symbols("x y z t Re")

# Define pi as a symbol so it prints as "M_PI" exactly as you want
# (Standard sympy pi might print as a number or different constant)
mpi = sp.Symbol("M_PI")

# Helper to scale arguments by pi (replacing sp.pi with our symbol)
def s(val): return sp.sin(mpi * val)
def c(val): return sp.cos(mpi * val)

# Velocity Fields
u = c(x) * s(y) * c(z) * sp.sin(t)
v = s(x) * c(y) * c(z) * sp.sin(t)
w = 2 * s(x) * s(y) * s(z) * sp.sin(t)
p = c(x) * c(y) * c(z) * sp.sin(t)

# --- 2. Define Derivatives and Navier-Stokes Forces ---
# Check continuity (optional, good for verification)
div = sp.diff(u, x) + sp.diff(v, y) + sp.diff(w, z)

assert(sp.simplify(div) == 0)
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

fx_calc = sp.diff(u_calc, t) - lap(u_calc)
fy_calc = sp.diff(v_calc, t) - lap(v_calc)
fz_calc = sp.diff(w_calc, t) - lap(w_calc)
# Substitute sp.pi back to M_PI symbol for the final C++ code
replacements = {sp.pi: mpi}
expressions = {
    "ux": u_calc.subs(replacements),
    "uy": v_calc.subs(replacements),
    "uz": w_calc.subs(replacements),
    "p" : p_calc.subs(replacements),
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
    if(name == "fx" or name == "fy" or name == "fz"):
        lines.append(f"auto {name}(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T")
    else:
        lines.append(f"auto {name}(const T& x, const T& y, const T& z, const T& t) -> T")

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

print("Generating", destination, "...")

with open(destination, "w") as f:
    # Header boilerplate
    f.write("#pragma once\n")
    f.write('#include "../include/navier_stokes.hpp"\n')
    f.write("#include <numbers>\n")
    f.write("namespace numPDE{\n")

    # Generate functions
    order = ["ux", "uy", "uz", "p", "fx", "fy", "fz"]
    
    for name in order:
        func_code = generate_cpp_function(name, expressions[name], printer)
        f.write(func_code)

with open(destination, "a") as f:
    f.write("}; //end numPDE")

print("Done! Check 'tools/manufactured_sols.hpp'.")

