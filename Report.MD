# CUDA 学习报告

## I 几句前言 

经过几周的CUDA学习, 我对CUDA编程有了初步的了解

## II 内存优化
CUDA 中存在多种类型的内存模型。一个 CUDA 程序可以选择只使用全局内存，这在逻辑上是没有任何问题的。但我们为了尽可能地提升程序的性能，应该尽量避免频繁地对全局内存进行访问。这是由于全局内存位于 DRAM 上，访问速度相较于位于芯片上(On-Chip)的内存慢。在不影响结果正确性的情况下，我们可以使用共享内存，常量内存，以及纹理内存 等特殊类型的内存来提升程序的性能。当然，使用特殊类型的内存也不见得一定能得到性能提升。有[文章](https://www.dcs.warwick.ac.uk/pmbs/pmbs/PMBS/papers/paper10.pdf)表明，在现代 GPU 架构(Maxwell 及其之后的架构)上，优化数据的存放位置有时并不能带来显著地性能提升。我们将在本部分的最后来简要探讨这个话题。


### II.1 使用共享内存
共享内存位于芯片上，具有低访问延迟的特点。共享内存的作用域为其所在的线程块，生命周期同此线程快。在很多情况下，可以在共享内存上建立缓冲区用来保留计算的中间结果，或者仅仅作为全局内存的一个 view，以此来降低访问延迟。需要注意的是，共享内存有大小限制。在我的设备(GTX 960M)上，每个线程块的可用共享内存大小为 49 152 字节。

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

光线追踪的例子中，位于空间中的球面信息是常量，这正好满足了常量内存只读的特性。储蓄球面信息所需的大小为`sizeof(float) * N_ATTRIBUTES * N_SPHERES`， 此处为`4 * 8 * 20` = `640 字节`，小于 64 千字。综上，我们可以使用常量内存来优化光线追踪程序。


### II.3 纹理内存

纹理内存亦是位于DRAM上的只读内存。其优势体现在纹理内存的缓存是针对 2维/3维 索引访问优化的。此外，当我们访问超出纹理内存边界的地址时，所返回的值是位于该内存边界地址所储存的值。对于一些程序，这种特性可以减少核函数的复杂程度，从而得到一定的性能提升。

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

Bari et al. 在其文章 [*Is Data Placement Optimization Still Relevant On Newer GPUs?*](https://www.dcs.warwick.ac.uk/pmbs/pmbs/PMBS/papers/paper10.pdf) 中提到，内存优化在现代 GPU (Maxwell 之后)上的性能提升作用不如在老式架构(Kepler 及之前)。主要原因是全局缓存的性能提升。大部分实验结果表明使用特殊内存时性能往往不会得到提升，甚至有时性反而下降。尤其在最近的Volta架构上，统一缓存(unified cache)的存在使得全局内存的访问几乎和其他特殊内存相同。

即便如此，我认为内存优化是有必要的。我们需要在生产中进行对比试验，根据实验结果选择最优的内存方案。


## III 指令优化

*《CUDA Best Prtactices Guide》* 中指出, 指令优化大体可分为数学指令优化和内存指令优化。其中内存指令优化基本上就是我们上述的内存优化。所以在这个部分我们主要研究数学指令优化。

### III.1 运用字面量

将常量声明成字面量(literals, 不知道这个翻译方法是否准确)有时可以避免计算中的不必要的类型转化。例如在下面的例子中:

```Cuda
float a(3.1415927);
float b = a * 1.414;	// 式(1)
float c = a * 1.414f;	// 式(2)
```

变量`b`和`c`的值往往会有一点点不同。这是由于在用单精度浮点型变量进行运算时，式(1)中的常量`1.414`一般会被编译器认为是双精度浮点型。在计算时 ，单精度变量`a`会被暂时转换为双精度参与计算，而`a * 1.414`的结果最后会被截短回单精度并赋给`b`。而在式(2)中，我们在常量`1.414f`后加了一个后缀`f`，相当于显式地告诉编译器这个`1.414f`是一个单精度常量，不要进行类型转换。正式这个细节造成了最后运算结果的不同。

值得注意的是，式(1)和式(2)中的两种指令往往会造成的性能差异。式(1)中会涉及到类型转化，而且GPU处理双精度运算的速度也比单精度情况下慢许多，所以用指令(1)的程序的性能会比用指令(2)的差。详见:[*Promotions to Doubles and Truncations to Floats*](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#promotions-to-doubles-and-truncations-to-floats)。除了 **单/双精度浮点型** 常量之外， **字符型** `char`和 **短整型** `short`在运算中一般会被转化为 **整型** `int`。

### III.2 整型除法和取余

*《CUDA Best Prtactices Guide》*中指出，整数除法和取余等操作一般是计算复杂的。我们可以尽可能地将这些操作转化为等价的比特偏移操作来优化性能。例如 若`n`是`2`的整数次幂时:

1. 相除指令`i/n`等价于`i >> log2(n)`;
2. 取余运算`i%n`等价于`i & (n-1)`;

另外，若`n`为字面量时，编译器会自动转化这些操作。

### III.3 利用专用数学函数

CUDA 中提供了丰富的数学函数库。为了保证程序的性能，我们要跟据情况调用合理的数学函数。例如计算`x`的平方`x*x`就要好过`pow(x,2)`。同样地，CUDA 提供常用的操作提供许多专有的数学函数就像`rcbrt(x)`好过`pow(x, -1/3)`，`sinpi(x)`好过`sin(PI*x)`等。

### III.4 指令顺序对计算精度可能的影响

我在计算矢量点积的例子中，分别探索了用全局内存，用共享内存，还有用串行版本的实现方法。用三种方式计算出来的结果均不同。我当时以为是不是我哪里没有理解清楚搞错了，还在 SO 上提了相关[问题](https://stackoverflow.com/questions/54642542/)。根据 Robert Crovella 提供的答案，
在单精度情况下，利用共享内存的计算结果是最接近双精度的计算结果的。这主要是由于计算时指令的顺序造成的系统误差。例如在[*《Floating Point and IEEE 754》*](https://docs.nvidia.com/cuda/floating-point/index.html#abstract)一文中，作者分别用 **串行法** ， **Fused Multiply-Add** ， **并行分治法** 计算了两个矢量`a = [1.907607, -.7862027, 1.148311, .9604002]`和`b = [-.9355000, -.6915108, 1.724470, -.7097529]`在单精度下的点积。结果如下：

| Approach |      Result     |  Float Value  |
|    ---   |       ---       |      ---      |
|   真值    |.0559587528435...|0x3D65350158...|
|   串行法  |   .0559588074   |  0x3D653510   |
|   FMA    |   .0559587515   |  0x3D653501   |
|  并行分治  |   .0559587478   | 	0x3D653500   |

在这个实例中，Fused Multily-Add 的结果最接近真值。其次为 **并行分治法** 和 **串行** 版本。

## IV 运行参数优化

在并行计算中，最大化设备的利用率，是提升程序性能的一个重要方法。在 CUDA 编程中，这一战略体现在最大化设备的 **占有率 (Occupancy)** 上。遵照*《CUDA Best Prtactices Guide》*中的说法，设备的占有率定义为每个 **SM** 上启动的线程束个数与最大线程束个数的比值。在计算 **占有率** 时，我们需要清楚地了解程序中每个线程所用的注册 (registers) 数量。例如在一个有 24 个线程束，每个线程束 32 个线程的 GPU 上，我们最多能同时执行 768 个线程。若此设备每个 **SM** 允许 8,192 个 32 位注册的话，我们每个线程上的注册数要保证小于 `8192 / 768 = 10.667 $\approx$ 10`，才能保证 100% 的占有率。
