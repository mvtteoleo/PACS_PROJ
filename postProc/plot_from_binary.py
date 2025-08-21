#Test made to visualize the porous mesh produced by my code and also to test python                \
    as makefile alternatice to integrate a better test workflow
import os from os.path import exists import matplotlib.pyplot as plt import numpy as np import subprocess

    N = int(10) file_name = "build/my_binary_dump.bin" rad = 0.3 cc = 0.5 exe = "build/bin_dump_from_py"

            if os.path.exists("./build") == False:print("Creating the missing build folder") os.mkdir("./build")

#COMPILE AND RUN THE C++ CODE
                                                                                                          try :
#Call g++ with C++ 23 standard
                                                                                                      if os.path.exists(exe) == False:print(f "Compiling and producing new executable") subprocess.run(["g++", "-std=c++23", "tests/binary_dump.cpp", "-o", exe], check = True) else :print(f "Executable already existing") subprocess.run([exe, str(N), file_name, str(rad), str(cc)], check = True) except subprocess.CalledProcessError as e:print("Compilation failed!") print(e)

#READ THE               BINARY CREATED
#Pay lot of care to the size !!here we expect the count to be in c++ a uint_64 and                 \
    the data to                                                  be double
                                                                                                                                                                                                                                                                                                                                                                                             with open(file_name, "rb") as f:count = int(np.fromfile(f, dtype = np.uint64, count = 1)[0]) data = np.fromfile(f, dtype = np.float64, count = count * 4) #3 pos + 1 value data = data.reshape((count, 4)) #shape(N, 4) :x, y, z, val

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               x, y, z, val = data.T

#PLOT
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        fig = plt.figure(figsize =(8, 6)) ax = fig.add_subplot(111, projection = "3d")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        sc = ax.scatter(x, y, z, c = val, cmap = "viridis", s = 10) fig.colorbar(sc, ax = ax, label = "Value")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ax.set_xlabel("X") ax.set_ylabel("Y") ax.set_zlabel("Z") plt.show()
