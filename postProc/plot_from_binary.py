# Test made to visualize the porous mesh produced by my code and also to test python as makefile alternatice to integrate a better test workflow
import os
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import subprocess

N = int(20)
rad = 0.6
cc = 00.5
exe = "build/serial/bin_dump_from_py"
# Test 0 is the sanity check; 1, the plot; 2 is the more structured one
test = 2


def write_data():
    if test == 1:
        file_name = "build/my_binary_dump.bin"
        # READ THE BINARY CREATED
        # Pay lot of care to the size!! here we expect the count to be in c++  a uint_64 and the data to be double
        with open(file_name, "rb") as f:
            count = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            data = np.fromfile(f, dtype=np.float64, count=count * 4)  # 3 pos + 1 value
            data = data.reshape((count, 4))  # shape (N,4): x,y,z,val
        x, y, z, val = data.T
        return x, y, z, val

    if test == 2:
        mesh_file = "build/mesh_datas.bin"
        scal_file = "build/ScalTens_dump.bin"
        vect_file = "build/VectTens_dump.bin"

        # READ THE BINARY CREATED
        # Pay lot of care to the size!! here we expect the count to be in c++  a uint_64 and the data to be double
        """
        with open(mesh_file, "rb") as f:
            count = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            data = np.fromfile(f, dtype=np.float64, count=count)  # 3 pos + 1 value
            data = data.reshape((3, 3))  # shape (N,4): x,y,z,val
            x_0, x_end, n_nodes = data
            x = np.linspace(x_0[0], x_end[0], num=int(n_nodes[0]), endpoint=True)
            y = np.linspace(x_0[1], x_end[1], num=int(n_nodes[1]), endpoint=True)
            z = np.linspace(x_0[2], x_end[2], num=int(n_nodes[2]), endpoint=True)
        """
        x = np.linspace(0, 6.14, 10, endpoint=True)
        y = np.linspace(0, 6.14, 10, endpoint=True)
        z = np.linspace(0, 6.14, 10, endpoint=True)

        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        with open(scal_file, "rb") as f:
            count = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            val = np.fromfile(f, dtype=np.float64, count=count)  # 3 pos + 1 value
        return x, y, z, val


if os.path.exists("./build") == False:
    print("Creating the missing build folder")
    os.mkdir("./build")

# # COMPILE AND RUN THE C++ CODE
# try:
#     if os.path.exists(exe) == False:
#         print(f"Compiling and producing new executable")
#         subprocess.run(
#             ["g++", "-std=c++23", f"-DTEST={test}", "tests/serial/binary_dump.cpp", "-o", exe],
#             check=True,
#         )
#     else:
#         print(f"Executable already existing")
#     print(f"Running {exe}")
#     subprocess.run([f"./{exe}", str(N), str(rad), str(cc)], check=True)
# except subprocess.CalledProcessError as e:
#     print("Compilation or execution failed!")
#     print(e)

x, y, z, val = write_data()

# PLOT
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(x, y, z, c=val, cmap="Greys_r", s=10)
fig.colorbar(sc, ax=ax, label="Value")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
