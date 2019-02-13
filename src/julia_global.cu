#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <cuda.h>
#include <common/book.h>
#include <common/cpu_bitmap.h>


#ifndef DIM
#define DIM (int) 256
#endif


struct CComplex {
    
    float   r;
    float   i;

    CComplex (float t_r, float t_i) : r(t_r), i(t_i) {}
    float mag (void) { return r*r + i*i; }
    CComplex operator*(const CComplex &a) {
        return CComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    CComplex operator+(const CComplex &a) {
        return CComplex(r+a.r, i+a.i);
    }
};


struct CuComplex {
    
    float   r;
    float   i;

    __device__ CuComplex (float t_r, float t_i) : r(t_r), i(t_i) {}
    __device__ float mag (void) { return r*r + i*i; }
    __device__ CuComplex operator*(const CuComplex &a) {
        return CuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ CuComplex operator+(const CuComplex &a) {
        return CuComplex(r+a.r, i+a.i);
    }
};


int  julia     (int, int);
void julia_cpu (unsigned char*);

__global__ void julia_gpu ( unsigned char * );
__device__ unsigned int julia_dev (const int, const int);


int main (int argc, char *argv[]) {

    std::cout << "DIM: " << DIM << '\n';
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr = bitmap.get_ptr();
    unsigned char *ptr_dev;

    // CPU Version
    auto t1 = std::chrono::system_clock::now();
    julia_cpu( ptr );
    auto t2 = std::chrono::system_clock::now();
    // bitmap.display_and_exit();
    double t_cpu = std::chrono::duration <double, std::milli> (t2 - t1).count();

    // GPU Version
    dim3 dimGrid( DIM, DIM );
    HANDLE_ERROR( cudaMalloc((void**)&ptr_dev, bitmap.image_size()) );
    auto t3 = std::chrono::system_clock::now();
    julia_gpu <<<dimGrid, 1>>> (ptr_dev);
    auto t4 = std::chrono::system_clock::now();
    HANDLE_ERROR( cudaMemcpy(bitmap.get_ptr(),
                             ptr_dev, 
                             bitmap.image_size(), 
                             cudaMemcpyDeviceToHost) );
    double t_gpu = std::chrono::duration <double, std::milli> (t4 - t3).count();

    printf("GPU time consummed: %.5f ms\n", t_gpu);
    printf("CPU time consummed: %.5f ms\n", t_cpu);

    bitmap.display_and_exit();
    cudaFree(ptr_dev);
    
    return 0;
}


__global__ void julia_gpu(unsigned char *ptr) {
    unsigned int x = blockIdx.x;
    unsigned int y = blockIdx.y;
    unsigned offset = x + y * gridDim.x;
    
    unsigned int julia_val = julia_dev(x, y);
    ptr[offset*4 + 0] = 255u * julia_val;
    ptr[offset*4 + 1] = 0u;
    ptr[offset*4 + 2] = 0u;
    ptr[offset*4 + 3] = 255u;
}


__device__ unsigned int julia_dev (int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    CuComplex c(-0.8f, 0.156f);
    CuComplex a(jx, jy);

    for (int i=0; i<200; ++i) {
        a = a*a + c;
        if (a.mag() > 1000)
            return 0u;
    }

    return 1u;
}


void julia_cpu(unsigned char *ptr) {
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int julia_val = julia(x, y);
            ptr[offset*4 + 0] = 255 * julia_val;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
}


int julia (int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    CComplex c(-0.8f, 0.156f);
    CComplex a(jx, jy);

    for (int i=0; i<200; ++i) {
        a = a*a + c;
        if (a.mag() > 1000)
            return 0;
    }

    return 1;
}

