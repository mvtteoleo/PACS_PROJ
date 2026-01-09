import numpy as np
import pyvista as pv

import vtk

# Manually use the VTK XML Structured Grid Reader
serial = vtk.vtkXMLStructuredGridReader()
serial.SetFileName("field_0000.vts")
serial.Update()

parallel = vtk.vtkXMLPStructuredGridReader()
parallel.SetFileName("paralle_p.pvts")
parallel.Update()
# Wrap the VTK object into a PyVista object
parallel_mesh = pv.wrap(parallel.GetOutput())
serial_mesh = pv.wrap(serial.GetOutput())

if serial_mesh.n_points == parallel_mesh.n_points:
    diff = serial_mesh['scalar'] - parallel_mesh['scalar']
    serial_mesh['Difference'] = diff

else:
    print("Error: Point counts differ.")

# 2. Find the index and value of the maximum absolute error
# We use abs() because a large negative error is still a large error
max_error_val = np.max(np.abs(diff))
max_error_idx = np.argmax(np.abs(diff))

# 3. Get the (x, y, z) coordinates of that point
max_loc = serial_mesh.points[max_error_idx]

print(f"Max Absolute Error: {max_error_val:.6f}")
print(f"Location (x, y, z): {max_loc}")
print(f"Index: {max_error_idx}")

y_loc = 0.602932
slice_mesh = serial_mesh.slice(normal='y', origin=(0, y_loc, 0))

# 2. Setup the plotter (off_screen=True prevents the window from popping up)
pl = pv.Plotter(off_screen=True)

# 3. Add the slice
# lighting=False is often better for 2D data slices to see raw colors without shadows
pl.add_mesh(slice_mesh, scalars='Difference', cmap='RdBu_r', lighting=False)

# 4. Set the camera to look at the Y-Z plane (looking down the X-axis)
pl.camera_position = "xz"

# Optional: Add a title or axes
pl.add_title(f"Difference at X = {y_loc:.2f}")
pl.show_axes()

# 5. Save the image
pl.save_graphic("diff_slice_max_err.pdf")

print("Saved 'diff_slice_x_pi_2.png' successfully.")


"""
# --- Visualization ---
p = pv.Plotter()

# Add the mesh with the difference scalar
serial_mesh['Difference'] = diff
p.add_mesh(serial_mesh, scalars='Difference', cmap='RdBu_r', opacity=0.5)

# Add a red sphere at the location of max error to highlight it
p.add_mesh(pv.Sphere(radius=0.1, center=max_loc), color='red', label='Max Error')

# Add a text label pointing to it
p.add_point_labels([max_loc], [f"Max Error: {max_error_val:.2e}"], 
                   point_size=20, font_size=24, text_color='red')

p.add_legend()
p.show()

"""
