# CUDA Matrix Multiplication Lab Report

## Table of Contents

- [Part 1: Basic Matrix Multiplication Implementations](#part-1-basic-matrix-multiplication-implementations)
  - [1. NVIDIA GPU Specifications](#1-nvidia-gpu-specifications)
  - [2. MatmulXrow vs MatmulYrow (Float)](#2-matmulxrow-vs-matmulyrow-float)
  - [3. Float vs Integer Matrix Multiplication](#3-float-vs-integer-matrix-multiplication)
  - [4. Block Tiling Approach](#4-block-tiling-approach)
- [Part 2: Advanced Optimization Techniques](#part-2-advanced-optimization-techniques)
  - [1. Improved 1D Block Tiling](#1-improved-1d-block-tiling)
  - [2. Improved 2D Block Tiling with Register Blocking](#2-improved-2d-block-tiling-with-register-blocking)
  - [3. Vectorized Shared and Global Memory Accesses](#3-vectorized-shared-and-global-memory-accesses)
  - [4. Warp-Level Tiling](#4-warp-level-tiling)
- [Part 3: cuBLAS Implementation](#part-3-cublas-implementation)
  - [Standard cuBLAS Implementation (FP32)](#standard-cublas-implementation-fp32)
  - [Mixed-Precision cuBLAS](#mixed-precision-cublas)
- [TensorCore in NVIDIA SMs](#tensor-core-implementation)
  - [TensorCore Principle and Benefits](#principle-and-benefits-of-tensorcores)
  - [Tesor Core Implementation](#tensor-core-implementation)
  - [Performance Comparison](#performance-comparison)
- [Conclusion](#conclusion)

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

### 4. Warp-Level Tiling

#### Concept and Benefits

Warp-level tiling introduces an **intermediate level of granularity** in the matrix multiplication hierarchy, situated between block-level tiling and thread-level work. This approach recognizes and leverages the fundamental execution unit of NVIDIA GPUs: the **warp** (a group of 32 threads that execute in lockstep).

**Key Concept:**
While thread blocks and individual threads are explicit programming constructs in CUDA, warps are the actual **hardware execution units**. By organizing computation specifically around these 32-thread units, warp-level tiling aligns the algorithm with the GPU's physical execution model.

This creates a three-tiered hierarchy of parallelism:

1. **Block-level parallelism**: Different thread blocks execute independently across SMs (Streaming Multiprocessors)
2. **Warp-level parallelism**: Within a block, warps execute independently on warp schedulers
3. **Thread-level parallelism**: Within a warp, individual threads compute multiple elements using register reuse

#### Implementation Strategy

The warp-tiling approach structures the computation as follows:

- Each **thread block** is responsible for computing a large tile of the output matrix
- Within the block, computation is explicitly organized by **warps**, where each warp computes a sub-tile
- Each **thread** within a warp computes multiple elements of the output matrix

This organization leads to a natural memory access pattern:
1. All threads in a block cooperatively load data from global memory into shared memory
2. Warps then independently load elements from shared memory into registers
3. Each thread performs multiple multiply-accumulate operations using those register values

By structuring the work this way, warps can operate independently with minimal synchronization, reducing warp divergence and improving scheduler efficiency.

#### Performance Advantages

1. **Warp scheduler optimization**: Since the GPU hardware schedules execution at the warp level, organizing computation by warps leads to more efficient scheduling and fewer stalls

2. **Enhanced register locality**: Data loaded into registers by threads within a warp stays local to that warp, improving cache efficiency

3. **Reduced synchronization overhead**: By making warps the primary computational unit, we minimize the need for costly block-wide synchronization

4. **Better memory coalescing**: When threads within a warp access memory together, their access patterns naturally align with hardware memory transaction sizes

5. **More effective latency hiding**: With warps operating independently, the GPU can better hide memory latency by swapping between warps when one is waiting for memory

#### Results

Kernel Execution Performance: 9611.92 GFLOPS.

**This is 7.11× (611%) times faster than our baseline**

This improvement demonstrates the critical importance of understanding and aligning algorithms with the GPU's actual execution model rather than just its programming abstractions.

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

## Tensor Core Implementation

### 1. TensorCore in NVIDIA Streaming Multiprocessors

#### Principle and Benefits of TensorCores

TensorCores are specialized hardware units in modern NVIDIA GPUs (Volta architecture and later) designed to accelerate matrix multiplication operations. They provide dedicated matrix-multiply-and-accumulate (MMA) units that can perform mixed-precision matrix operations with exceptional throughput.

**Core Principles:**

1. **Matrix Math Acceleration**: TensorCores perform matrix operations in a single clock cycle that would take multiple cycles on traditional CUDA cores.

2. **Mixed Precision Computing**: TensorCores operate on lower precision inputs (FP16 or BF16) but accumulate results in higher precision (FP32), providing both performance and accuracy benefits.

3. **Specialized Matrix Instructions**: TensorCores use specialized instructions (WMMA - Warp Matrix Multiply-Accumulate) that operate on matrix fragments rather than individual values.

**Key Benefits:**

1. **Dramatically Higher Compute Throughput**: TensorCores can deliver up to 8× higher peak FLOPS compared to standard FP32 operations on the same GPU.

2. **Mixed Precision Without Accuracy Loss**: While inputs are FP16, accumulation in FP32 maintains numerical stability for most applications.

3. **Memory Bandwidth Efficiency**: Lower precision inputs (FP16) require half the memory bandwidth of FP32, effectively doubling the available bandwidth.

4. **Energy Efficiency**: TensorCores perform more calculations per watt than standard CUDA cores.

#### TensorCore Implementation

Our implementation uses NVIDIA's WMMA (Warp Matrix Multiply-Accumulate) API, which provides a high-level interface to program TensorCores. Here's how the implementation works:

**Host Program:**

```cpp
// Matrix dimensions (N×N matrices)
int N = 8192;
int M = N;
int K = N;

// Allocate host memory and initialize with FP16 data
vector<half> h_a(N * N);  // Input matrix A in FP16
vector<half> h_b(N * N);  // Input matrix B in FP16
vector<float> h_c(N * N); // Output matrix C in FP32

// Initialize matrices with FP16 values
for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);

// Allocate device memory
half* d_a, * d_b;  // Device FP16 input matrices
float* d_c;        // Device FP32 output matrix
cudaMalloc(&d_a, sizeof(half) * M * K);
cudaMalloc(&d_b, sizeof(half) * K * N);
cudaMalloc(&d_c, sizeof(float) * M * N);

// Copy data to the device
cudaMemcpy(d_a, h_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);

// Calculate grid and block dimensions based on TensorCore dimensions (16×16×16)
dim3 gridDim(ceil(static_cast<float>(M) / WMMA_M), 
             ceil(static_cast<float>(N) / WMMA_N), 1);
dim3 blockDim(32, 32, 1);  // 32 threads in x, 32 threads in y

// Warm-up run to amortize initialization overhead
wmmaMatrixMultiply<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
cudaDeviceSynchronize();

// Launch actual kernel
wmmaMatrixMultiply<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);

// Copy results back to host
cudaMemcpy(h_c.data(), d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
```

**Kernel Implementation:**

```cpp
__global__ void wmmaMatrixMultiply(half* A, half* B, float* C, int M, int N, int K) {
    // Calculate which warp in the grid this thread belongs to
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Declare matrix fragments for TensorCore operations
    // These fragments store portions of the input and output matrices
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                           nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                           nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, 
                           float> c_frag;
    
    // Initialize accumulator fragment to zeros
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop through K dimension in steps of TensorCore tile size (WMMA_K)
    for (int k = 0; k < K; k += WMMA_K) {
        // Calculate starting positions for loading matrix tiles
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;
        
        // Check if this warp's assigned work is within matrix bounds
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrix tiles into TensorCore fragments
            nvcuda::wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            nvcuda::wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform matrix multiplication using TensorCores
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Calculate output position for storing result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    // Store accumulated result back to global memory
    if (cRow < M && cCol < N) {
        nvcuda::wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, 
                                        nvcuda::wmma::mem_row_major);
    }
}
```

#### Performance Comparison

We compared the TensorCore implementation with our previously implemented Block Tiled 16×16 matrix multiplication algorithm, which uses the same block size but relies on standard CUDA cores rather than TensorCores:

| Implementation | Performance (GFLOPS) | Speedup vs Block Tiled 16×16 |
|----------------|---------------------|------------------------------|
| Block Tiled 16×16 | 1422.85 | 1.00× |
| TensorCore | 6268.35 | 4.41× |
| cuBLAS (FP32) | 12431.02 | 8.74× |
| cuBLAS (FP16/BF16) | 30923.96 | 21.73× |

#### Analysis of TensorCore Performance

The TensorCore implementation achieved a 4.41× speedup over the Block Tiled 16×16 implementation, despite both using the same general approach and block dimensions. This dramatic performance difference can be attributed to several key advantages of TensorCores:

1. **Hardware-Accelerated Matrix Operations**: TensorCores perform matrix operations in specialized hardware units that are optimized specifically for these operations, whereas CUDA cores are general-purpose compute units.

2. **Parallel Execution at Warp-Level**: TensorCores operate on matrix fragments in a synchronized manner across an entire warp, enabling higher efficiency than individual thread operations.

3. **Mixed Precision Advantage**: Using FP16 for inputs reduces memory bandwidth requirements while maintaining computational accuracy through FP32 accumulation. This mixed-precision approach effectively doubles available memory bandwidth.

4. **Reduced Instruction Count**: TensorCore operations replace numerous individual multiply-add operations with single warp-wide matrix operations, significantly reducing instruction overhead.

5. **Memory Access Efficiency**: The WMMA API optimizes the memory access patterns for TensorCore operations, further enhancing performance.

While our TensorCore implementation shows impressive performance, it still falls short of the cuBLAS library (approximately 50% of cuBLAS FP32 performance). This gap exists because:

- cuBLAS includes advanced memory access optimizations and heuristics developed over many years
- cuBLAS uses autotuning to select optimal parameters for a given GPU architecture
- Our TensorCore implementation is relatively simple and could benefit from additional optimizations

Nevertheless, the 4.41× speedup over our optimized Block Tiled implementation demonstrates the tremendous potential of TensorCores for accelerating matrix multiplication operations, even with relatively straightforward implementations.

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

4. **TensorCore acceleration** provides dramatic performance improvements:
   - 4.4× speedup over equivalent CUDA core implementation
   - Mixed precision operations enable significantly higher throughput
   - Hardware-specific accelerators like TensorCores represent the future of high-performance matrix operations

