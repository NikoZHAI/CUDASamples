#include <iostream>
#include <cuda.h>
#include <string>
#include <common/book.h>
#include <common/cpu_bitmap.h>


#ifndef NTHREADS
#define NTHREADS (int) 16
#endif

#ifndef DIM
#define DIM (int) 256
#endif

#ifndef NSPHERES
#define NSPHERES (int) 20
#endif

#define INF 2e10f
#define rnd( x ) (x*rand() / RAND_MAX )


struct Sphere;
__global__ void ray_tracing ( unsigned char* );


struct Sphere {
    float x, y, z;  // center
    float rad;      // radius
    float r, g, b;  // color

    __device__
    float hit ( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < rad*rad) {
            float dz = sqrtf ( rad*rad - dx*dx - dy*dy );
            *n = dz / sqrtf( rad*rad );
            return dz + z;
        }
        return -INF;
    }
};


__constant__ Sphere dev_s[NSPHERES];
int main( int argc, char *argv[] ) {
    if ( argc > 1) {
        srand( std::stoi(argv[1]) );
    } else {
        srand( time(0) );
    }

    // Excise CUDA Event
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0) );

    CPUBitmap bitmap( DIM, DIM );
    unsigned char *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_s, sizeof(Sphere) * NSPHERES ) );
    Sphere *host_s = (Sphere*) malloc( sizeof(Sphere) * NSPHERES );

    for (size_t i=0; i < NSPHERES; ++i) {
        host_s[i].x   = rnd( 1000.f ) - 500.f;
        host_s[i].y   = rnd( 1000.f ) - 500.f;
        host_s[i].z   = rnd( 1000.f ) - 500.f;
        host_s[i].rad = rnd( 100.f ) + 20.f;
        host_s[i].r   = rnd( 1.f );
        host_s[i].g   = rnd( 1.f );
        host_s[i].b   = rnd( 1.f );
    }

    HANDLE_ERROR( cudaMemcpyToSymbol (dev_s, host_s,
                                      sizeof(Sphere) * NSPHERES) );
    
    free(host_s);

    dim3    dimGrid( DIM/NTHREADS, DIM/NTHREADS );
    dim3    dimBlocks( NTHREADS, NTHREADS );
    ray_tracing <<< dimGrid, dimBlocks >>> ( dev_bitmap );
    cudaDeviceSynchronize();

    HANDLE_ERROR( cudaMemcpy(bitmap.get_ptr(),
                             dev_bitmap,
                             bitmap.image_size(),
                             cudaMemcpyDeviceToHost) );
    
    // Benchmarking
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    
    float elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime ( &elapsedTime,
                                         start, stop ) );
    printf( "Time to generate:    %3.2f ms\n", elapsedTime );
    
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    cudaFree(dev_bitmap);
    bitmap.display_and_exit();

    return 0;
}


__global__ void ray_tracing (unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = ( x - DIM/2 );
    float oy = ( y - DIM/2 );

    float r=0.f, g=0.f, b=0.f;
    float maxz = -INF;
    for (unsigned int i=0; i<NSPHERES; ++i) {
        float n;
        float t = dev_s[i].hit( ox, oy, &n );

        if ( t > maxz ) {
            float fscale = n;
            r = dev_s[i].r * fscale;
            g = dev_s[i].g * fscale;
            b = dev_s[i].b * fscale;
        }
    }

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

