
2025-04-24 14:35

Status:

Tags:

--- 
# TP CUDA
## Part 1
### 1. SPECS
===== GPU Device #0 =====
Name: NVIDIA GeForce RTX 3070 Laptop GPU
Architecture (Compute Capability): 8.6
Total Global Memory: 8191 MB
Multiprocessor Count (SMs): 40
--- Warp and Thread Info ---
Warp Size (Threads per Warp): 32
Max Threads per SM: 1536
Max Warps per SM: 48
--- Block/Thread Info ---
Max Threads per Block: 1024
Max Block Dimensions: [1024, 1024, 64]
Max Grid Dimensions: [2147483647, 65535, 65535]
--- Memory Info ---
Shared Memory per Block: 48 KB
Shared Memory per SM: 100 KB
Registers per Block: 65536
L2 Cache Size: 4096 KB
Memory Bus Width: 256 bits
Memory Clock Rate: 7001 MHz
--- Misc ---
Clock Rate: 1620 MHz
Concurrent Kernels Supported: Yes

#### some values found in internet:
| Attribute                           | Value                                |
| ----------------------------------- | ------------------------------------ |
| **Compute Capability**              | 8.6                                  |
| **Architecture**                    | Ampere (GA104)                       |
| **Registers per SM**                | **65,536 (64K 32-bit registers)**    |
| **Max Registers per Thread**        | 255                                  |
| **Register Allocation Granularity** | Warp (32 threads)                    |
| **Register Allocation Unit Size**   | 256 (32-bit registers)               |
| **Max Warps per SM**                | 48                                   |
| **Max Threads per SM**              | 48 warps × 32 threads = **1,536**    |
| **Max Registers per Block**         | 65,536                               |
| **Max Shared Memory per SM**        | 100 KB (configurable)                |
| **Max Shared Memory per Block**     | 48 KB (default), up to 100 KB opt-in |

### 2. MatmulXrow and MatmulYrow (Float)

results for X mult:

Time elapsed on Host To Device Transfer: 56.477886 ms.

Time elapsed on matrix multiplication on GPU: 7479.195312 ms.

Time elapsed on Device To Host Transfer: 45.755360 ms.

Total Time: 7581.428711 ms.

Kernel Execution Performance: 147.009354 GFLOPS.


results for Y mult:

Time elapsed on Host To Device Transfer: 56.021954 ms.

Time elapsed on matrix multiplication on GPU: 1035.420776 ms.

Time elapsed on Device To Host Transfer: 44.376480 ms.

Total Time: 1135.819214 ms.

Kernel Execution Performance: 1061.898315 GFLOPS.

#### Analysis:
we see that the matmulYrow is 7 times faster than the matmulXrow this is due to the fact that coalesced mem access is way faster than uncoalesced.

####  do you think that the implementation of the same kernel with integer Matrices?
No it would be atleast halfed since the current nvidia graphics card has only 16 int cuda cores comapred to the 32 float computing cuda cores.

### 3. Float vs int matmul

#### Float:

results for Xrow mult:

Time elapsed on Host To Device Transfer: 56.477886 ms.

Time elapsed on matrix multiplication on GPU: 7479.195312 ms.

Time elapsed on Device To Host Transfer: 45.755360 ms.

Total Time: 7581.428711 ms.

Kernel Execution Performance: 147.009354 GFLOPS.


results for Yrow mult:

Time elapsed on Host To Device Transfer: 56.021954 ms.

Time elapsed on matrix multiplication on GPU: 1035.420776 ms.

Time elapsed on Device To Host Transfer: 44.376480 ms.

Total Time: 1135.819214 ms.

Kernel Execution Performance: 1061.898315 GFLOPS.

#### int:

results for Xrow mult:

Time elapsed on Host To Device Transfer: 58.708576 ms.

Time elapsed on matrix multiplication on GPU: 7667.127930 ms.

Time elapsed on Device To Host Transfer: 44.044640 ms.

Total Time: 7769.880859 ms.

Kernel Execution Performance: 143.405930 GFLOPS.

results for Yrow mult:


Kernel Execution Performance: 1095.387695 GFLOPS.

#### Analysis:
We notice that even though in our current device there is twice the amount of fp32 cores compared to int cores the performance is identical.

#### Hypothesis:
The performance bottleneck isn't instruction throughput. So in a memory-bound kernel, **INT and FLOAT will perform similarly**, even if FP has more theoretical throughput.
(in the currentmatmul code there is 2 memory access and 1 store for each product and adition)

### 4. Block tiled aproach

results or block tiled for block size = 16:

Kernel Execution Performance: 1422.845459 GFLOPS.

results or block tiled for block size = 32:

Kernel Execution Performance: 1352.121094 GFLOPS.

#### Analysis:
The 16x16 block tiled approach is 5% faster than the 32x32, this is partially explained by the 100% warp concurrence utilization when using 16x16 (6 block could be loaded at the same time) and 66% warp concurrence utilization in the 32x32 approach (only one block loaded in the SM each time)


 compared to the matmul yrow the block tile approach is 20% faster 

based on calculations:
Mem Bandwidht = 448.06 GBs

Compute Bandwidth = 16.6 TFLOPS

Critical Arithmetic intensity = 34.5

Arithmetic intensity for Yrow = 0.48479

Arithmetic intensity for tiled = 7.9844

based on these calculations theoretically the blocktiled aproach should be 16.5 times faster than Yrow approach however based on real benchmarks there is only a 20% increase in performance. This could be due to:
- Hidden memory optimisations for the coalsed approach that reduce global memory access (this is logical since the actual theortical AI <<<< real observed AI = 1061/448 = 2.3683) 
- The overhead in the blocktiled approach since we are using shared memory and syncthreads means that the observed AI <<< theoretical AI

## Part 2
We will take the performance of the 32x32 blockTiled appproch as a baseline = 1352.121094 GFLOPS
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
In our case we took each block computes 64x64 block with BK = 8 ,TM = 8
effectively here we are calculating 8 results per thread

Kernel Execution Performance: 3974.385742 GFLOPS.

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

Kernel Execution Performance: 7140.942383 GFLOPS.


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

#### Results:
for:
BK = 8
TM = 8
TN = 8
BM = 128
BN = 128

Kernel Execution Performance: 8128.805664 GFLOPS.

we notice a 8.2 times (720%) increase compared to the baseline and a 1.24 times increase (24%) compared to the simple 2D blocktiling approach.

#### 4. Warp Tiling:

### 4. Warptiling in Matrix Multiplication (Matmul)

#### Technique Description

This optimization **introduces an extra level of tiling** in the matrix multiplication, called **warptiling**, between blocktiling and threadtiling.  
It **organizes the work inside a warp** (32 threads) more systematically to better match the GPU hardware capabilities, especially warp schedulers and register usage.

Unlike blocks and threads, **warps are not explicit in CUDA code** — they are a hardware scheduling unit. We simulate warp-level organization by calculating a thread’s `warpId` using:

`warpId = threadIdx.x % warpSize;`

where `warpSize = 32` .

By grouping threads into warp-sized tiles, **memory accesses** become more coalesced, **register usage** becomes more cache-friendly, and the **SM’s warp schedulers** can be better utilized.

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
        
        - Load each thread’s **sub-rows** of A and **sub-columns** of B into registers.
            
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
    
- **Prepares for tensor cores**: The warptiled organization closely matches the format needed for **tensor core** instructions (`mma.sync`), which accelerates matmul dramatically.
#### Results:
Kernel Execution Performance: 9611.924805 GFLOPS.

## Part3: Cublas implementation:
results for CUBLAS_COMPUTE_32F:

Kernel Execution Performance: 8298.698242 GFLOPS.

results for CUBLAS_COMPUTE_32F_FAST_16BF which is faster :

Kernel Execution Performance: 13367.097656 GFLOPS.


for our given best solution it is actuallly 15% faster than cublast fp32 and and 30% slower than the 16BF





# References
