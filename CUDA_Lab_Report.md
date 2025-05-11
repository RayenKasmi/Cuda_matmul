# CUDA Matrix Multiplication Lab Report

## Part 1: Basic Matrix Multiplication Implementations

### 1. NVIDIA GPU Specifications

The experiments were conducted on an NVIDIA GeForce RTX 3070 Laptop GPU with the following specifications:

| Specification | Value |
|---------------|-------|
| Architecture | Ampere (GA104) |
| Compute Capability | 8.6 |
| Total Global Memory | 8191 MB |
| Multiprocessor Count (SMs) | 40 |
| Warp Size | 32 threads |
| Max Threads per SM | 1536 |
| Max Warps per SM | 48 |
| Max Threads per Block | 1024 |
| Max Block Dimensions | [1024, 1024, 64] |
| Max Grid Dimensions | [2147483647, 65535, 65535] |
| Shared Memory per Block | 48 KB |
| Shared Memory per SM | 100 KB |
| Registers per Block | 65536 |
| L2 Cache Size | 4096 KB |
| Memory Bus Width | 256 bits |
| Memory Clock Rate | 7001 MHz |

### 2. MatmulXrow vs MatmulYrow (Float)

#### Performance Results

| Kernel | Host to Device (ms) | Kernel Execution (ms) | Device to Host (ms) | Total Time (ms) | Performance (GFLOPS) |
|--------|---------------------|----------------------|---------------------|-----------------|----------------------|
| MatmulXrow | 56.48 | 7479.20 | 45.76 | 7581.43 | 147.01 |
| MatmulYrow | 56.02 | 1035.42 | 44.38 | 1135.82 | 1061.90 |

#### Analysis
The MatmulYrow implementation is approximately 7 times faster than MatmulXrow. This significant performance difference is due to memory access patterns:

- **MatmulXrow (Uncoalesced)**: Threads access memory in a non-coalesced manner, causing multiple memory transactions for each warp.
- **MatmulYrow (Coalesced)**: Threads access memory in a coalesced manner, allowing for efficient memory transactions where threads in a warp access contiguous memory locations.

Coalesced memory access is crucial for CUDA performance as it maximizes memory bandwidth utilization, resulting in the observed 7x speedup for MatmulYrow.

### 3. Float vs Integer Matrix Multiplication

#### Expected Behavior
Before running the integer implementation, we predicted that the integer version might have different performance characteristics since the NVIDIA RTX 3070 has 16 integer CUDA cores compared to 32 float computing CUDA cores per SM.

#### Results (Total execution Time)

| Kernel | Float (ms) | Integer (ms) |
|--------|----------------|------------------|
| MatmulXrow |  7581.428711 | 7769.88085 |
| MatmulYrow |  1135.81921 | 1103.91735 |

#### Analysis
Surprisingly, the performance difference between float and integer implementations is minimal, despite the hardware having twice as many floating-point cores as integer cores. This indicates that:

1. The matrix multiplication kernels are **memory-bound** rather than compute-bound.
2. The performance bottleneck is memory bandwidth, not arithmetic throughput.
3. In memory-bound operations, the theoretical differences in compute capabilities have little impact on actual performance.

Our hypothesis aligns with observations: in the current matrix multiplication code, there are 2 memory accesses and 1 store operation for each product and addition, making memory bandwidth the limiting factor.

### 4. Block Tiling Approach

#### Performance Results

| Implementation | Block Size | Performance (GFLOPS) | Speedup vs MatmulYrow |
|----------------|------------|----------------------|----------------------|
| MatmulYrow | 32x32 | 1061.90 | 1.00x |
| Block Tiling | 16×16 | 1422.85 | 1.34x |
| Block Tiling | 32×32 | 1352.12 | 1.27x |

#### Analysis
The block tiling approach with both 16×16 and 32×32 block sizes outperforms the MatmulYrow implementation:

- The 16×16 block tiling is approximately 5% faster than the 32×32 block tiling. This is explained by better SM utilization with 16×16 blocks (100% warp utilization) compared to 32×32 blocks (66% warp utilization).
- Compared to MatmulYrow, the block tiling approach is about 34% faster (for 16×16 blocks).

#### Arithmetic Intensity Analysis

| Metric | Value |
|--------|-------|
| Memory Bandwidth | 448.06 GB/s |
| Compute Bandwidth | 16.6 TFLOPS |
| Critical Arithmetic Intensity | 34.5 |
| MatmulYrow Arithmetic Intensity | 0.48479 |
| Block Tiled Arithmetic Intensity | 7.9844 |

Theoretically, based on arithmetic intensity, the block tiled approach should be 16.5 times faster than the MatmulYrow approach. However, in practice, we observe only a 34% improvement. This discrepancy can be attributed to:

1. Hidden memory optimizations in the coalesced approach that reduce global memory access (observed AI for MatmulYrow : 1061.90/448.06 = 2.37 is higher than theoretical one = 0.487)
2. Overhead in the block tiled approach from shared memory usage and thread synchronization, reducing the observed arithmetic intensity below theoretical expectations.

## Part 2: Advanced Optimization Techniques

Using the 32×32 Block Tiled approach (1352.12 GFLOPS) as our baseline, we implemented several advanced optimization techniques:

### 1. Improved 1D Block Tiling 

#### Technique Description

This is an optimized **block tiling** method where each thread computes **multiple output elements** instead of only one.  
The key change is that the number of threads per block is reduced (for example from 32×32 to 32×8), so it is no longer possible to load an entire 32×32 tile in one go.  
Therefore, the K dimension is loaded **in smaller slices (BK)** to match the reduced number of threads available for loading data. 

#### How It Works

- Threads cooperate to load smaller tiles of A and B matrices into shared memory.
    
- Each thread caches partial computation results in **registers** (arrays in thread-private memory).
    
- Threads perform more computation before writing back to global memory.
    
- The loop over the K dimension advances in chunks of size **BK** instead of TILE_SIZE.
    

#### Why It Improves Performance

- **Higher computation per thread**: Each thread handles several outputs, reducing idle time.
    
- **Better latency hiding**: Threads spend more time computing relative to waiting for memory accesses.

#### Results
In our case we took each block computes 64x64 block with BK = 8, TM = 8
effectively here we are calculating 8 results per thread

Kernel Execution Performance: 3974.39 GFLOPS.

**This is 2.93 (193%) times faster than our baseline**

### 2. Improved 2D Block Tiling with Register Blocking

#### Technique Description

This is an optimized **2D block tiling** method where each thread computes a **tile of multiple output elements** (e.g., an 8×8 patch).  
The key change is that threads now work on **small tiles of the output matrix** instead of a single element or row, significantly increasing the computation per thread.  
Therefore, the K dimension is loaded **in smaller slices (BK)**, and the elements needed for computation are cached into **registers** for faster access.

#### How It Works

- Threads cooperate to load smaller tiles of A and B matrices into shared memory.
    
- Each thread loads **multiple elements** from A and B into shared memory using strided accesses.
    
- Each thread caches relevant elements of A and B into **register arrays**.
    
- Threads compute a **full TM×TN outer product** in registers before writing back to global memory.
    
- The loop over the K dimension advances in chunks of size **BK** instead of TILE_SIZE.
    

#### Why It Improves Performance

- **Higher computation per memory load**: Each thread produces a full tile of results, not just one output, improving data reuse.
    
- **Reduced memory traffic**: Once elements are loaded into shared memory and registers, many computations are performed before new data is needed.
    
- **Better latency hiding**: Threads spend more time on computation relative to memory waiting, leading to fewer stalls.

#### Results

BK = 8
TM = 8
TN = 8
BM = 128
BN = 128

Kernel Execution Performance: 7140.94 GFLOPS.

**This is 5.28 (428%) times faster than our baseline**

### 3. Vectorized Shared and Global Memory Accesses

#### Technique Description

This optimization focuses on improving memory bandwidth utilization by **vectorizing** both shared memory and global memory accesses.  
The key changes are **transposing the A tile** when loading it into shared memory, and **using vector datatypes (e.g., `float4`)** when accessing global memory.  
This enables the hardware to perform **128-bit memory transactions** instead of 32-bit, significantly boosting data transfer efficiency.

#### How It Works

- During the global memory load, matrix A is **transposed** while copying into shared memory to enable **coalesced and vectorized shared memory accesses**.
    
- Threads use `float4` loads and stores to move **four floats at once**, both when reading from global memory and writing into shared memory.
    
- For matrix B, threads use `float4` loads from global memory into shared memory without needing a transpose.
    
- `reinterpret_cast<float4*>` is used to assure the compiler that pointers are **128-bit aligned**, allowing efficient vector instructions to be generated.
    
- The loop structure remains the same, but now both global and shared memory accesses are **vectorized**.
    

#### Why It Improves Performance

- **Faster shared memory access**: Transposing A allows **vectorized shared memory loads**, reducing the number of SMEM load instructions.
    
- **Faster global memory access**: Using `float4` leads to **coalesced and wider GMEM transactions** (`LDG.E.128` and `STG.E.128`), decreasing memory access latency.
    
- **Better memory alignment**: Explicit pointer casting (`reinterpret_cast<float4*>`) guarantees memory alignment, allowing the compiler to generate efficient 128-bit load/store instructions.
    
- **Reduced instruction count**: Fewer memory load/store instructions means more bandwidth is available for useful computation.

#### Results
For:
BK = 8
TM = 8
TN = 8
BM = 128
BN = 128

Kernel Execution Performance: 8128.81 GFLOPS.

We notice a 6.01 times (501%) increase compared to the baseline and a 1.14 times increase (14%) compared to the simple 2D blocktiling approach.

### 4. Warptiling in Matrix Multiplication (Matmul)

#### Technique Description

This optimization **introduces an extra level of tiling** in the matrix multiplication, called **warptiling**, between blocktiling and threadtiling.  
It **organizes the work inside a warp** (32 threads) more systematically to better match the GPU hardware capabilities, especially warp schedulers and register usage.

Unlike blocks and threads, **warps are not explicit in CUDA code** — they are a hardware scheduling unit. We simulate warp-level organization by calculating a thread's `warpId` using:

`warpId = threadIdx.x % warpSize;`

where `warpSize = 32`.

By grouping threads into warp-sized tiles, **memory accesses** become more coalesced, **register usage** becomes more cache-friendly, and the **SM's warp schedulers** can be better utilized.

#### How It Works

- **Three levels of parallelism** are introduced:
    
    - **Blocktiling**: Different blocks run independently on different SMs (Streaming Multiprocessors).
        
    - **Warptiling**: Inside a block, different warps run independently on warp schedulers.
        
    - **Threadtiling**: Inside a warp, each thread computes a tiny piece using instruction-level parallelism (ILP).
        
- Inside the main loop:
    
    - **Load A and B subtiles** from shared memory into thread-local **registers**.
        
    - **Registers are organized** so each thread handles multiple small pieces of the matrix (multiple rows and columns).
        
    - **Small matrix multiplies** are performed using **registers** to maximize speed and locality.
        
    - This structure makes it easy to later map the computation to **tensor cores** using warp-wide matrix instructions (e.g., WMMA).
        
- Code breakdown:
    
    - For each slice (`dotIdx`):
        
        - Load each thread's **sub-rows** of A and **sub-columns** of B into registers.
            
        - Multiply and accumulate into `threadResults` using small nested loops over the sub-tiles.
            
- Each **warp** computes a chunk of matrix C of size:

    `(WSUBN * WNITER) × (WSUBM * WMITER)`
    
    and each thread within the warp computes:
    
    `WNITER × WMITER small blocks of size (TM × TN).`
    

#### Why It Improves Performance

- **Better warp scheduler utilization**: Warps are the unit scheduled onto warp schedulers. By warptiling, we naturally divide work into warp-sized chunks.
    
- **Higher register reuse**: Loading sub-tiles into registers improves **temporal locality**, reducing the need to reload from shared memory.
    
- **Minimized shared memory bank conflicts**: Since warps have their memory access patterns aligned, it reduces conflicts when accessing shared memory banks.
    
- **Improved instruction-level parallelism (ILP)**: Threads perform multiple independent multiply-adds, making better use of the GPU cores.
    

#### Results
Kernel Execution Performance: 9611.92 GFLOPS.

**This is 7.11 (611%) times faster than our baseline**

## Part 3: cuBLAS Implementation

### Implementation and Performance

#### Standard cuBLAS Implementation (FP32)
- Performance: 12431.02 GFLOPS
- Speedup vs. Baseline: 9.19× (819% improvement)

#### Mixed-Precision cuBLAS (CUBLAS_COMPUTE_32F_FAST_16BF)
- Performance: 30923.96 GFLOPS
- Speedup vs. Baseline: 22.87× (2187% improvement)

### Comparison with Custom Implementations

| Implementation | Performance (GFLOPS) | Relative to Best Custom (Warp Tiling) |
|----------------|----------------------|--------------------------------------|
| Warp Tiling | 9611.92 | 1.00× |
| cuBLAS (FP32) | 12431.02 | 1.29× (29% faster) |
| cuBLAS (FP16/BF16) | 30923.96 | 3.22× (222% faster) |

#### Analysis
- Our best custom implementation (Warp Tiling) achieves impressive performance but is outperformed by standard cuBLAS FP32 by approximately 29%.
- The mixed-precision cuBLAS implementation is 222% faster than our best custom implementation, demonstrating the extraordinary performance gains possible when leveraging tensor cores and reduced precision operations.


#### Important Note on Performance Measurement
To achieve these optimal cuBLAS performance results, it was necessary to add a warm-up run before the timed execution. This warm-up run amortizes initialization overhead, including kernel compilation, memory allocation, and internal cuBLAS configuration. Without this warm-up pass, the measured performance would be significantly lower due to these one-time initialization costs being included in the measurement.

## Conclusion

This lab demonstrates the importance of memory access patterns and hierarchical optimizations in CUDA programming for matrix multiplication. Key findings include:

1. **Memory access patterns** are critical: coalesced access (MatmulYrow) achieved a 7× speedup over uncoalesced access (MatmulXrow).

2. **Memory bandwidth** is often the limiting factor: integer and floating-point operations performed similarly despite differences in hardware capabilities.

3. **Hierarchical optimizations** provide cumulative benefits:
   - Block Tiling: 1.3× speedup over coalesced baseline
   - 1D Block Tiling with register reuse: 2.9× speedup
   - 2D Block Tiling: 5.3× speedup
   - Vectorization: 6.0× speedup
   - Warp Tiling: 7.1× speedup

