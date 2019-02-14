#include <iostream>
#include <cuda.h>
#include <common/book.h>
#include <common/cpu_anim.h>


#ifdef USE_2DTEXTURE_MEMO
#define USE_TEXTURE_MEMO
#define TEXTURE_DIM 2
#else
#define TEXTURE_DIM 1
#endif


#ifndef NTHREADS
#define NTHREADS (int) 16
#endif

#ifndef DIM
#define DIM (int) 1024
#endif

#ifndef NSTEPS
#define NSTEPS (int) 90
#endif

#ifndef SPEED
#define SPEED 0.25f
#endif

#ifndef MAX_TEMP
#define MAX_TEMP 1.f
#endif

#ifndef MIN_TEMP
#define MIN_TEMP 0.0001f
#endif


// Declarations
struct DataBlock;

#ifndef USE_TEXTURE_MEMO
__global__ void copy_const_kernel( float*, const float* );
__global__ void step_run_kernel( float*, const float* );
#else
__global__ void copy_const_kernel( float* );
__global__ void step_run_kernel( float*, bool);
#endif

__global__ void my_float_to_color( unsigned char*, const float* ); // float to color defined in book.h, line 80
void anim_exit_callback( DataBlock* );
void anim_gpu( DataBlock*, int );
void my_swap( float**, float** );

struct DataBlock {
    CPUAnimBitmap   *bitmap;
    unsigned char   *dev_bitmap;
    float           *dev_constSrc;
    float           *dev_inSrc;
    float           *dev_outSrc;

    cudaEvent_t     start;
    cudaEvent_t     stop;
    float           totalElapsedTime;
    float           frames;
};


/* Texture references that resides on GPU */
#ifdef USE_TEXTURE_MEMO
texture <float, TEXTURE_DIM> texRefConstSrc;
texture <float, TEXTURE_DIM> texRefIn;
texture <float, TEXTURE_DIM> texRefOut;
#endif


int main (void) {

    DataBlock       data;
    CPUAnimBitmap   bitmap( DIM, DIM, &data );
    
    data.bitmap = &bitmap;
    data.frames = 0.f;
    data.totalElapsedTime = 0.f;

    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );
    // Note that one char takes 1 byte, one float takes 4 bytes, 
    // so the following allocation is correct
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_bitmap)  , bitmap.image_size() ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_constSrc), bitmap.image_size() ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_inSrc)   , bitmap.image_size() ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_outSrc)  , bitmap.image_size() ) );

    // Binding texture reference to device memory
    #ifdef USE_TEXTURE_MEMO
    #ifndef USE_2DTEXTURE_MEMO
    /*__host__ ​cudaError_t cudaBindTexture ( size_t* offset, 
     *                                       const textureReference* texref, 
     *                                       const void* devPtr, 
     *                                       const cudaChannelFormatDesc* desc, 
     *                                       size_t size = UINT_MAX )
    */
    HANDLE_ERROR( cudaBindTexture( NULL, 
                                   texRefConstSrc,
                                   data.dev_constSrc,
                                   bitmap.image_size() ) );
    HANDLE_ERROR( cudaBindTexture( NULL, 
                                   texRefIn,
                                   data.dev_inSrc,
                                   bitmap.image_size() ) );
    HANDLE_ERROR( cudaBindTexture( NULL, 
                                   texRefOut,
                                   data.dev_outSrc,
                                   bitmap.image_size() ) );

    #else
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR( cudaBindTexture2D( NULL, 
                                     texRefConstSrc,
                                     data.dev_constSrc,
                                     desc, DIM, DIM,
                                     sizeof(float) * DIM ) );
    HANDLE_ERROR( cudaBindTexture2D( NULL, 
                                     texRefIn,
                                     data.dev_inSrc,
                                     desc, DIM, DIM,
                                     sizeof(float) * DIM ) );
    HANDLE_ERROR( cudaBindTexture2D( NULL, 
                                     texRefOut,
                                     data.dev_outSrc,
                                     desc, DIM, DIM,
                                     sizeof(float) * DIM ) );
    #endif  // <-- #ifdef USE_2DTEXTURE_MEMO
    #endif  // <-- #ifdef USE_TEXTURE_MEMO


    /* Initial Condition */
    float *cond_init = (float*) malloc( bitmap.image_size() );
    for (int i=0; i<DIM*DIM; i++) {
        cond_init[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x>300) && (x<600) && (y>310) && (y<601))
            cond_init[i] = MAX_TEMP;
    }
    cond_init[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
    cond_init[DIM*700+100] = MIN_TEMP;
    cond_init[DIM*300+300] = MIN_TEMP;
    cond_init[DIM*200+700] = MIN_TEMP;
    for (int y=800; y<900; y++) {
        for (int x=400; x<500; x++) {
            cond_init[x+y*DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy(data.dev_constSrc,
                             cond_init,
                             bitmap.image_size(), 
                             cudaMemcpyHostToDevice) );

    for (int y=800; y<DIM; ++y) {
        for (int x=0; x<200; ++x) {
            cond_init[x+y*DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy(data.dev_inSrc,
                             cond_init,
                             bitmap.image_size(), 
                             cudaMemcpyHostToDevice) );
    free( cond_init );

    bitmap.anim_and_exit( (void (*)(void*, int))anim_gpu,
                          (void (*)(void*))anim_exit_callback );

}


void anim_exit_callback ( DataBlock* d ) {

    /* Unbind texture reference if neccessary */
    #ifdef USE_TEXTURE_MEMO
    cudaUnbindTexture( texRefIn );
    cudaUnbindTexture( texRefOut );
    cudaUnbindTexture( texRefConstSrc );
    #endif

    cudaFree( d->dev_bitmap );
    cudaFree( d->dev_inSrc  );
    cudaFree( d->dev_outSrc );
    cudaFree( d->dev_constSrc );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


#ifndef USE_TEXTURE_MEMO  /* Use Global Memory */
void anim_gpu ( DataBlock *d, int ticks ) {

    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );

    dim3    dimGrid( DIM/NTHREADS, DIM/NTHREADS );
    dim3    dimBlocks( NTHREADS, NTHREADS );

    for (int i=0; i<NSTEPS; ++i) {
        copy_const_kernel <<< dimGrid, dimBlocks >>> ( d->dev_inSrc,
                                                       d->dev_constSrc );
        step_run_kernel <<< dimGrid, dimBlocks >>> ( d->dev_outSrc, d->dev_inSrc );
        my_swap( &d->dev_inSrc, &d->dev_outSrc );
    }
    float_to_color <<< dimGrid, dimBlocks >>> ( d->dev_bitmap, d->dev_inSrc );
    cudaDeviceSynchronize();

    HANDLE_ERROR( cudaMemcpy( d->bitmap->get_ptr(),
                              d->dev_bitmap,
                              d->bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );

    float elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, d->start, d->stop ) );
    
    d->totalElapsedTime += elapsedTime;
    ++(d->frames);
    
    printf( "Mean computation time per frame: %3.2f ms\n",
            d->totalElapsedTime/d->frames );
}


__global__ void copy_const_kernel( float *inSrc, const float *constSrc ) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = x + y * blockDim.x * gridDim.x;

    if ( constSrc[offset] != 0 ) { inSrc[offset] = constSrc[offset]; }
}


__global__ void step_run_kernel( float *outSrc, const float *inSrc ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    /* AND THAT, IS WHAT PPL CALLED STUPID */
    // int x1  = (x!=(DIM-1)) ? (x+1) : x; // Right
    // int x4  = x;                        // Bottom
    // int x16 = x ? (x-1) : x;            // Left
    // int x64 = x;                        // Top

    // int y1  = y;
    // int y4  = y ? (y-1) : y;
    // int y16 = y;
    // int y64 = (y!=(DIM-1)) ? (y+1) : y;

    // int offset1  = x1 + y1 * blockDim.x * gridDim.x;
    // int offset4  = x4 + y4 * blockDim.x * gridDim.x;
    // int offset16 = x16 + y16 * blockDim.x * gridDim.x;
    // int offset64 = x64 + y64 * blockDim.x * gridDim.x;

    // outSrc[offset] = ( 1.f - 4.f * SPEED ) * inSrc[offset] +
    //                  SPEED * ( inSrc[offset1] + inSrc[offset4] +
    //                            inSrc[offset16] + inSrc[offset64] );
    /* END OF STUPIDITY */

    int top    =      y       ? (offset-DIM) :  offset;
    int right  = (x != DIM-1) ? (offset + 1) :  offset;
    int bottom = (y != DIM-1) ? (offset+DIM) :  offset;
    int left   =      x       ? (offset - 1) :  offset;

    outSrc[offset] = ( 1.f - 4.f * SPEED ) * inSrc[offset] + 
                     SPEED * ( inSrc[top] + inSrc[right] + inSrc[bottom] + inSrc[left]);
}

#else  /* Use Texture Memory */

void anim_gpu ( DataBlock *d, int ticks ) {

    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );

    dim3    dimGrid( DIM/NTHREADS, DIM/NTHREADS );
    dim3    dimBlocks( NTHREADS, NTHREADS );

    volatile bool dstIsOut = true;  // Ues keyword volatile to prevent caching
    for (size_t i=0; i<NSTEPS; ++i) {
        // float *in, *out;
        // if (dstIsOut) {
        //     in  = d->dev_inSrc;
        //     out = d->dev_outSrc;
        // }
        // else{
        //     in  = d->dev_outSrc;
        //     out = d->dev_inSrc;
        // }

        copy_const_kernel <<< dimGrid, dimBlocks >>> ( d->dev_inSrc );
        step_run_kernel <<< dimGrid, dimBlocks >>> ( d->dev_outSrc, dstIsOut );
        my_swap( &d->dev_inSrc, &d->dev_outSrc );
        dstIsOut = !dstIsOut;
    }
    float_to_color <<< dimGrid, dimBlocks >>> ( d->dev_bitmap, d->dev_inSrc );
    cudaDeviceSynchronize();

    HANDLE_ERROR( cudaMemcpy( d->bitmap->get_ptr(),
                              d->dev_bitmap,
                              d->bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );

    float elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, d->start, d->stop ) );
    
    d->totalElapsedTime += elapsedTime;
    ++(d->frames);
    
    printf( "Mean computation time per frame: %3.2f ms\n",
            d->totalElapsedTime/d->frames );
}

#ifndef USE_2DTEXTURE_MEMO

__global__ void copy_const_kernel( float *inSrc ) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = x + y * blockDim.x * gridDim.x;

    float constSrc = tex1Dfetch(texRefConstSrc, offset);
    if ( constSrc != 0 ) {
        inSrc[offset] = constSrc; 
    }
}


__global__ void step_run_kernel( float *outSrc, bool dstIsOut ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int top    =      y       ? (offset-DIM) :  offset;
    int right  = (x != DIM-1) ? (offset + 1) :  offset;
    int bottom = (y != DIM-1) ? (offset+DIM) :  offset;
    int left   =      x       ? (offset - 1) :  offset;

    float v_top, v_right, v_bottom, v_left, v_old;
    if (dstIsOut) {
        v_top    = tex1Dfetch( texRefIn, top    );
        v_right  = tex1Dfetch( texRefIn, right  );
        v_bottom = tex1Dfetch( texRefIn, bottom );
        v_left   = tex1Dfetch( texRefIn, left   );
        v_old    = tex1Dfetch( texRefIn, offset );
    } else {
        v_top    = tex1Dfetch( texRefOut, top    );
        v_right  = tex1Dfetch( texRefOut, right  );
        v_bottom = tex1Dfetch( texRefOut, bottom );
        v_left   = tex1Dfetch( texRefOut, left   );
        v_old    = tex1Dfetch( texRefOut, offset );
    }

    outSrc[offset] = ( 1.f - 4.f * SPEED ) * v_old + 
                     SPEED * ( v_top + v_right + v_bottom + v_left);
}


#else

__global__ void copy_const_kernel( float *inSrc ) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = x + y * blockDim.x * gridDim.x;

    float constSrc = tex2D(texRefConstSrc, x, y);
    if ( constSrc != 0 ) {
        inSrc[offset] = constSrc; 
    }
}


__global__ void step_run_kernel( float *outSrc, bool dstIsOut ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float v_top, v_right, v_bottom, v_left, v_old;
    if (dstIsOut) {
        v_top    = tex2D( texRefIn, x, y-1 );
        v_right  = tex2D( texRefIn, x+1, y );
        v_bottom = tex2D( texRefIn, x, y+1 );
        v_left   = tex2D( texRefIn, x-1, y );
        v_old    = tex2D( texRefIn, x ,  y );
    } else {
        v_top    = tex2D( texRefOut, x, y-1 );
        v_right  = tex2D( texRefOut, x+1, y );
        v_bottom = tex2D( texRefOut, x, y+1 );
        v_left   = tex2D( texRefOut, x-1, y );
        v_old    = tex2D( texRefOut, x ,  y );
    }

    outSrc[offset] = ( 1.f - 4.f * SPEED ) * v_old + 
                     SPEED * ( v_top + v_right + v_bottom + v_left);
}

#endif // <-- #ifndef USE_TEXTURE2D_MEMO
#endif // <-- #ifndef USE_TEXTURE_MEMO


__global__ void my_float_to_color(unsigned char *ptr, const float *inSrc) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = x + y * blockDim.x * gridDim.x;

    ptr[offset*4 + 0] = (int)( 255 * inSrc[offset] );
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 0;
}


void my_swap (float **in, float **out) {
    float *dummy = *out;
    *out = *in;
    *in = dummy;
}
