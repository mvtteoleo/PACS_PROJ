#C2Decomp
C++ and MPI implementation of 2-D pencil decomposition and transpose routines based on 2Decomp Library. Most of this content is almost verbatim from 2Decomp (2decomp.org) just translated from Fotran to C++.

As of how it is implemented the sizes refer to the oringinal (X) displacement.
The index are kept ROWMAJOR and go as follow: 


| PENCIL | fastest | medium | slowest |
| ------------- | -------------- | -------------- | -------------- |
| X | i | j | k |
| Y | j | k | i |
| X | k | i | j |

So that the indexing goes as follows:

X-DIRECTION
``` cpp
nx = xSizes[0]
ny = xSizes[1]
nz = xSizes[2]
l = i + nx * j + ny*nx*k
u[l] = u(i, j, k) 
```


Y-DIRECTION
``` cpp
nx = ySizes[0]
ny = ySizes[1]
nz = ySizes[2]
l = j + ny * k + ny*nz*i
u[l] = u(i, j, k) 
```

Z-DIRECTION
``` cpp
nx = zSizes[0]
ny = zSizes[1]
nz = zSizes[2]
l = k + nz * i + nz*nx*j
u[l] = u(i, j, k) 
```


In the tensor based loop this logic is NOT simply enforced by: 
    - `[k, j, i]` in the standard X pencil
    - `[i, k, j]` in the standard Y pencil
    - `[j, i, k]` in the standard Z pencil

Some minor changes to the class could be done, but I prefer to avoid confusion and keep it X-based only!

To check is enough to look at the reordering test logic and at the initialization test
where is clearly defined how the from the for loop how every thing works.

``` c++
    // FFT along X (local contiguously)
    for (int kp = 0; kp < xSizeArr[2]; ++kp)
        for (int jp = 0; jp < xSizeArr[1]; ++jp)
{
    std::copy_n(data1.ptr_at(0, jp, kp), Lx, xbuf);
    fftw_execute(fft_x);
    std::copy_n(xbuf, Lx, data1.ptr_at(0, jp, kp));
}

// transpose X -> Y (blocking)
decomp.transposeX2Y(u1, u2);

// FFT along Y (contiguous along jp; indexing for u2: ii = ip * ySize[2]*ySize[1] + kp*ySize[1]
// + jp)
for (int ip = 0; ip < ySizeArr[0]; ++ip)
    for (int kp = 0; kp < ySizeArr[2]; ++kp)
    {
        int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1];
        std::copy_n(data2.ptr_at(ii), Ly, xbuf);
        fftw_execute(fft_y);
        std::copy_n(xbuf, Ly, data2.ptr_at(ii));
    }

// transpose Y -> Z
decomp.transposeY2Z(u2, u3);

// FFT along Z (contiguous along kp; indexing for u3: ii = jp * zSize[2]*zSize[0] + ip *
// zSize[2] + kp)
for (int jp = 0; jp < zSizeArr[1]; ++jp)
    for (int ip = 0; ip < zSizeArr[0]; ++ip)
    {
        int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2];
        std::copy_n(data3.ptr_at(ii), Lz, xbuf);
        fftw_execute(fft_z);
        std::copy_n(xbuf, Lz, data3.ptr_at(ii));
    }
```
