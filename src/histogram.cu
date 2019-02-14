#include <iostream>
#include <chrono>
#include <cuda.h>
#include <common/book.h>


#ifndef BLOCK_SIZE
#define BLOCK_SIZE (int)100
#endif

#define SIZE (int)( BLOCK_SIZE*1024*1024 )
static const int CHAR_SIZE = sizeof(unsigned char);
static const int NCHARS = SIZE / CHAR_SIZE;

#ifndef NTHREADS
#define NTHREADS 256
#endif

#ifndef NBLOCKS
#define NBLOCKS 1024
#endif
#if  NCHARS < (NTHREADS * 1024)
#define NBLOCKS (int)( ( NCHARS + NTHREADS - 1 ) / NTHREADS )
#endif

#define N_MAX_CHAR 256  // chars
#if N_MAX_CHAR != NTHREADS
#define NTHREADS N_MAX_CHAR
#endif

#define CPU_NOW() (std::chrono::system_clock::now())

#ifdef USE_GPU
__global__ void buff_to_histo( unsigned char*, unsigned int* );
#endif


int main (void) {
    unsigned char *buffer = (unsigned char*)big_random_block( SIZE );
    unsigned int histo[N_MAX_CHAR]{0};

    /* Timer */
    float elapsedTime;

    #ifndef USE_GPU
    ///////////////////////////////////////////
    // //////////////// CPU //////////////// //
    ///////////////////////////////////////////
    auto t1 = CPU_NOW();
    for (int i=0; i<SIZE; ++i){
        ++histo[ buffer[i] ];
    }
    auto t2 = CPU_NOW();
    elapsedTime = std::chrono::duration<float, std::milli> (t2 - t1).count();
    #else
    //////////////////////////////////////////
    // //////////////// GPU /////////////// //
    //////////////////////////////////////////
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    unsigned char* dev_buffer;
    unsigned int*  dev_histo;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_buffer, SIZE ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_histo,
                              N_MAX_CHAR * sizeof(int) ) );
    HANDLE_ERROR( cudaMemcpy( dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemset( dev_histo, 0, N_MAX_CHAR * sizeof(int) ) );

    dim3    dimGrid( NBLOCKS, 1, 1 );
    dim3    dimBlocks( NTHREADS, 1, 1 );
    buff_to_histo <<< dimGrid, dimBlocks >>> ( dev_buffer, dev_histo );
    cudaDeviceSynchronize();
    cudaMemcpy(histo, dev_histo,
               N_MAX_CHAR * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime ( &elapsedTime, start, stop ) );
    HANDLE_ERROR( cudaFree( dev_buffer ) );
    HANDLE_ERROR( cudaFree( dev_histo ) );
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    #endif  // <-- USE_GPU
    
    
    long countHisto = 0;
    for (int i=0; i<N_MAX_CHAR; ++i){
        countHisto += histo[i];
    }
    
    free(buffer);

    if ( countHisto == NCHARS ) {
        printf( "Work done, time consummed: %.2f ms.\n", elapsedTime );    
    } else {
        printf( "\nNo match of Histogram count(%lo) and the Size of data block(%i).\n",
                countHisto, NCHARS );
        throw std::runtime_error("No match of Histogram count and the Size of data block.\n");
    }

    return 0;
}


#ifdef USE_GPU
#ifdef USE_SHARED_MEM
__global__ void buff_to_histo( unsigned char* buffer, unsigned int* histo ) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ unsigned int sub_histo[NTHREADS];

    ////////// !!! IMPORTANT !! ///////////////////////////////////
    sub_histo[threadIdx.x] = 0;
    __syncthreads();

    while ( tid < NCHARS ) {
        unsigned char ascii = buffer[tid];
        atomicAdd( &sub_histo[ascii], 1 );
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();    // Wait for all sub_histo calculated
    
    /* Cumulate sub_histos */
    atomicAdd( &histo[threadIdx.x], sub_histo[threadIdx.x] );
}
#else
__global__ void buff_to_histo( unsigned char* buffer, unsigned int* histo ) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while ( tid < NCHARS ) {
        unsigned char ascii = buffer[tid];
        atomicAdd( &histo[ascii], 1 );
        tid += blockDim.x * gridDim.x;
    }
}
#endif  // <-- USE_SHARED_MEM
#endif  // <-- USE_GPU
