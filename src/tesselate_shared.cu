#include <iostream>
#include <cuda.h>
#include <common/book.h>
#include <common/cpu_bitmap.h>

#ifndef DIM
#define DIM (int) 256
#endif

#ifndef NTHREADS
#define NTHREADS (int) 16
#endif

#define PI 3.1415926535897932f


// Decalration
__global__ void tesselate ( unsigned char * );


int main (void) {

    CPUBitmap       bitmap( DIM, DIM );
    unsigned char   *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );

    dim3 dimGrid( DIM/NTHREADS, DIM/NTHREADS, 1 );
    dim3 dimBlocks( NTHREADS, NTHREADS, 1 );

    tesselate <<< dimGrid, dimBlocks >>> ( dev_bitmap );
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(),
                              dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost )
                );
    
    cudaFree( dev_bitmap );
    bitmap.display_and_exit();
}


__global__ void tesselate ( unsigned char *ptr ) {

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float mossaic[NTHREADS][NTHREADS];

    const float period = 128.f;

    mossaic[threadIdx.x][threadIdx.y] = 
        255.f * (sinf(x*2.f*PI / period) + 1.f) *
                (sinf(y*2.f*PI / period) + 1.f) / 4.f;
    __syncthreads();

    ptr[offset*4u + 0u] = 0;
    ptr[offset*4u + 1u] = mossaic[NTHREADS-1 - threadIdx.x][NTHREADS-1 - threadIdx.y];
    ptr[offset*4u + 2u] = 0;
    ptr[offset*4u + 3u] = 255u;
}

