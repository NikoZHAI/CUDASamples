# CUDA 学习报告

## I 几句前言 

经过几周的CUDA学习, 我对CUDA编程有了初步的了解

## II 内存优化
CUDA 中存在多种类型的内存模型。一个 CUDA 程序可以选择只使用全局内存，这在逻辑上是没有任何问题的。但我们为了尽可能地提升程序的性能，应该尽量避免频繁地对全局内存进行访问。这是由于全局内存位于 DRAM 上，访问速度相较于位于芯片上(On-Chip)的内存慢。在不影响结果正确性的情况下，我们可以使用共享内存，常量内存，以及纹理内存 等特殊类型的内存来提升程序的性能。当然，使用特殊类型的内存也不见得一定能得到性能提升。有[文章](https://www.dcs.warwick.ac.uk/pmbs/pmbs/PMBS/papers/paper10.pdf)表明，在现代 GPU 架构(Maxwell 及其之后的架构)上，优化数据的存放位置有时并不能带来显著地性能提升。我们将在本部分的最后来简要探讨这个话题。


### II.1 使用共享内存
共享内存位于芯片上，具有低访问延迟的特点。共享内存的作用域为其所在的线程块，生命周期同此线程快。在很多情况下，可以在共享内存上建立缓冲区用来保留计算的中间结果，或者仅仅作为全局内存的一个 view，以此来降低访问延迟。需要注意的是，共享内存有大小限制。在我的设备(GTX 960M)上，每个线程快的可用共享内存大小为 49 152 字节。

**用例一：利用共享内存计算向量点积**

```Cuda
__global__ void dot_prod(int length, float *u, float *v, float *out) {
    __shared__ float cache[nThreadsPerBlock];
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned cid = threadIdx.x;

    float temp = 0;
    while (tid < length) {
        temp += u[tid] * v[tid];
        tid  += blockDim.x * gridDim.x;
    }

    cache[cid] = temp;
    __syncthreads();
    
    int i = nThreadsPerBlock/2;
    while (i != 0) {
        if (cid < i) {
            cache[cid] += cache[cid + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cid == 0) {
        out[blockIdx.x] = cache[0];
    }
}
```

在计算点积的算例中，我们利用在共享内存上创建的缓冲区`cache`来储蓄计算的中间结果。该缓冲区随后即被用于进一步的归约运算。整个过程对全局内存的访问次数仅为必要的向量长度`length`次，提升了程序的性能。


**用例二：利用共享内存做矩阵乘法**

```Cuda
__global__ void mat_mul(float *a, float* b, float *c, int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}
```

在矩阵乘法中，位于共享内存的`aTile`和`bTile`仅为位于全局内存的原矩阵的 view。之后的循环累加都访问这两个缓冲区，大大地提升了计算性能。


### II.2 常量内存

常量内存是位于DRAM上的只读内存。其访问速度与全局内存相同。不同的是，每个设备上允许有64 KB 的常量内存缓存。当所访问地址未被缓存的情况下，每个读取操作耗费一个设备内存读取时间，否则仅消耗一个缓存读取时间(与 registers 读取耗费相同)。当每个线程束(warp)均访问同一小部分内存(64 KB)时并且仅执行只读操作时，常量缓存便能体现出它的优势。

**用例：模拟光线追踪效果**

```Cuda
__constant__ Sphere dev_s[NSPHERES];
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
```

光线追踪的例子中，位于空间中的球面信息是常量，这正好满足了常量内存只读的特性。储蓄球面信息所需的大小为`sizeof(float) * N_ATTRIBUTES * N_SPHERES`， 此处为`4 * 8 * 20` = `640 字节`，小于 640 千字。综上，我们可以使用常量内存来优化光线追踪程序。


### II.3 纹理内存

纹理内存亦是位于DRAM上的只读内存。其优势体现在纹理内存的缓存是针对 2维/3维 索引访问优化的。此外，当我们访问超出纹理内存边界的地址即相当于访问位于该内存边界地址。对于一些程序，这种特性可以减少核函数的复杂程度，从而得到一定的性能提升。

**用例：模拟热传导**

```Cuda
texture <float, 2> texRefConstSrc;
texture <float, 2> texRefIn;
texture <float, 2> texRefOut;

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
```

热传导例子中，二维纹理内存很好地处理了边界访问情况。但在我的机器上，纹理内存的使用并没有带来性能的提升。反而，计算速度相比较使用全局内存还慢了 20% 到 30% 。我们将在下一个部分来探讨这个问题。


### II.4 关于现代 GPU 内存优化的探讨

我们之前提到，有时内存优化反而会带来性能的下降。例如在模拟热传导中，使用纹理内存虽然降低了核函数的复杂程度，但性能却略有下降。

Bari et al. 在其文章 [*Is Data Placement Optimization Still Relevant On Newer GPUs?*](https://www.dcs.warwick.ac.uk/pmbs/pmbs/PMBS/papers/paper10.pdf) 中提到，内存优化在现代 GPU (Maxwell 之后)上的性能提升作用不如老式架构(Kepler 及之前)。主要原因是全局缓存的性能提升。大部分实验结果表明使用特殊内存时性能往往不会得到提升，甚至有时性反而下降。尤其在最近的Volta架构上，统一缓存(unified cache)的存在使得全局内存的访问几乎和其他特殊内存相同。

即使如此，我认为内存优化是有必要的。我们需要在生产中进行对比试验，根据实验结果选择最优的内存方案。

## III 指令优化


## IV 运行参数优化

