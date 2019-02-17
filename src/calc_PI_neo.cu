#include <iostream>
#include <cuda.h>

typedef unsigned int U4;

#ifdef USE_FP32
    typedef float T_real;
#else
    typedef double T_real;
#endif

#ifndef N_SERIES
    #define N_SERIES ( U4 )( 1<<31 )
#endif

#ifndef NTHREADS
    #define NTHREADS ( U4 )512
#endif

#ifndef NBLOCKS
    #define NBLOCKS ( U4 )256
#endif

#ifdef USE_ATOMIC
    #define SIZE_SERIES NTHREADS
#else
    #define SIZE_SERIES NBLOCKS
#endif

#define TRUE_PI "3.141592653589793238462643383279"

#define STEP ( T_real )( 1./N_SERIES )

static void HandleError( cudaError_t err,
    const char *file,
    int line )
{
    if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
    file, line );
    exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#ifdef USE_ATOMIC
__global__ void calc_pi( T_real *pi_series ) {
    U4 tid = threadIdx.x + blockIdx.x * blockDim.x;
    T_real item;
    __shared__ T_real pi_cache[NTHREADS];
    pi_cache[threadIdx.x] = 0.;
    // __syncthreads();
    
    while( tid < N_SERIES ) {
        item = ( tid + 0.5 ) * STEP;
        item = 4. / (1. + item*item);
        pi_cache[threadIdx.x] += item; // No need for atomicAdd here
        tid += blockDim.x * gridDim.x;
    }

    atomicAdd( &pi_series[threadIdx.x], pi_cache[threadIdx.x] );
}
#else
__global__ void calc_pi( T_real *pi_series ) {
    U4 tid = threadIdx.x + blockIdx.x * blockDim.x;
    T_real item;
    __shared__ T_real pi_cache[NTHREADS];
    pi_cache[threadIdx.x] = 0.;
    // __syncthreads();
    
    while( tid < N_SERIES ) {
        item = ( tid + 0.5 ) * STEP;
        item = 4. / (1. + item*item);
        pi_cache[threadIdx.x] += item; // No need for atomicAdd here
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();
    
    // Reduction to NBLOCKS elements
    int i = NTHREADS/2;
    while( i ) {
        if ( threadIdx.x < i ) {
            pi_cache[threadIdx.x] += pi_cache[i + threadIdx.x];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
        pi_series[blockIdx.x] += pi_cache[0];
    // atomicAdd( &pi_series[threadIdx.x], pi_cache[threadIdx.x] );
}
#endif


int main ( void ) {

    U4     size = SIZE_SERIES * sizeof( T_real );
    T_real my_pi(0.);
    float  elapsed_time;

    // Reduced PI series of NTHREADS elements
    T_real *pi_series = ( T_real* )malloc( size );
    T_real *dev_pi_series;

    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    HANDLE_ERROR( cudaMalloc( (void**)&dev_pi_series, size ) );
    HANDLE_ERROR( cudaMemset( dev_pi_series, 0., size ) );

    dim3    dimGrid( NBLOCKS, 1, 1 );
    dim3    dimBlocks( NTHREADS, 1, 1 );
    calc_pi <<< dimGrid, dimBlocks >>> ( dev_pi_series );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( pi_series, dev_pi_series, size, cudaMemcpyDeviceToHost) );

    // Further Reduction
    for (int i=0; i<SIZE_SERIES; ++i) {
        my_pi += pi_series[i];
    }
    my_pi *= STEP;

    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsed_time, start, stop ) );
    HANDLE_ERROR( cudaFree( dev_pi_series ) );
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    free( pi_series );

    printf("\nResult with 30 digits:\n");
    printf("\n   PI calculated    = %1.30f\n", my_pi);
    printf("      True PI       = %s\n", TRUE_PI);
    printf("                        ");
    for (int i=1; i!=15; ++i)
        printf(" ");
    printf("^\n");
    printf("================================================\n\n");
    printf("Run with configuration: \n");
	printf("  N thread blocks   = %12u\n", NBLOCKS);
    printf("N threads per block = %12u\n", NTHREADS);
    printf("   Series Length    = %12u\n", N_SERIES);
    printf("  N Total Threads   = %12u\n\n", NTHREADS*NBLOCKS);
    printf("    Elapsed Time    = %.3f ms.\n\n", elapsed_time);

    return 0;
}

