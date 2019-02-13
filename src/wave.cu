#include <iostream>
#include <string>
#include <chrono>
#include <cuda.h>
#include <common/book.h>
#include <common/cpu_anim.h>
#include <common/gpu_anim.h>


#ifndef NTHREADS
#define NTHREADS (int) 16
#endif

#ifndef DIM
#define DIM (int) NTHREADS*NTHREADS
#endif


struct DataBlock {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;     // book.h
};


void cleanup ( DataBlock *d ) {
    HANDLE_ERROR( cudaFree( d->dev_bitmap ) );
}


// Declarations
void generate_frame ( DataBlock*, int );
__global__ void wave ( unsigned char*, int );


int main ( void ) {
    DataBlock data;
    CPUAnimBitmap bitmap( DIM, DIM, &data );
    data.bitmap = &bitmap;

    HANDLE_ERROR( cudaMalloc( (void**) &data.dev_bitmap, bitmap.image_size() ) );

    bitmap.anim_and_exit( (void (*)(void*,int))generate_frame,
                          (void (*)(void*))cleanup );
}


void generate_frame ( DataBlock *d, int ticks ) {

    dim3 dimGrid( DIM/NTHREADS, DIM/NTHREADS, 1 );
    dim3 dimBlocks( NTHREADS, NTHREADS, 1 );

    wave <<< dimGrid, dimBlocks >>> ( d->dev_bitmap, ticks );
    cudaDeviceSynchronize();

    HANDLE_ERROR( cudaMemcpy(d->bitmap->get_ptr(),
                             d->dev_bitmap, 
                             d->bitmap->image_size(),
                             cudaMemcpyDeviceToHost) );
}


__global__ void wave (unsigned char *ptr, int ticks) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = x + y * blockDim.x * gridDim.x;   // uni-dimensional index

    float fx = x - DIM/2;   // distance to the center of the map
    float fy = y - DIM/2;
    float d = sqrtf( fx*fx + fy*fy );

    unsigned char grey = (unsigned char) ( 128.f + 127.f * 
                                           cosf(d/10.f - ticks/7.f) / 
                                           (d/10.f + 1.f)
                                         );
    
    // Every pixel contains 4 values RGBa? So offset * 4
    ptr[offset*4u + 0u] = grey;
    ptr[offset*4u + 1u] = grey;
    ptr[offset*4u + 2u] = grey;
    ptr[offset*4u + 3u] = 255;
}

