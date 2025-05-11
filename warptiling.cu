#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;
typedef unsigned int uint;

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32;

namespace wt {
    template <const int BM, const int BN, const int BK, const int rowStrideA,
        const int rowStrideB>
    __device__ void loadFromGmem(int N, int K, const float* A, const float* B,
        float* As, float* Bs, int innerRowA, int innerColA,
        int innerRowB, int innerColB) {
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            // float4 tmp;
            // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
            //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
            //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
            // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
            //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
            //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
        }
    }

    template <const int BM, const int BN, const int BK, const int WM, const int WN,
        const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
        const int TM, const int TN>
    __device__ void
        processFromSmem(float* regM, float* regN, float* threadResults, const float* As,
            const float* Bs, const uint warpRow, const uint warpCol,
            const uint threadRowInWarp, const uint threadColInWarp) {
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // populate registers for whole warptile
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint i = 0; i < TM; ++i) {
                    regM[wSubRowIdx * TM + i] =
                        As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                        threadRowInWarp * TM + i];
                }
            }
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint i = 0; i < TN; ++i) {
                    regN[wSubColIdx * TN + i] =
                        Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                        threadColInWarp * TN + i];
                }
            }

            // execute warptile matmul
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN] +=
                                regM[wSubRowIdx * TM + resIdxM] *
                                regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
    }

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemmWarptiling(int M, int N, int K, float* A, float* B, float* C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/2=32
    constexpr uint WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = { 0.0 };
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = { 0.0 };
    float regN[WNITER * TN] = { 0.0 };

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
            TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                threadRowInWarp, threadColInWarp);
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    float4 tmp = reinterpret_cast<float4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    tmp.x = threadResults[i + 0] + tmp.x;
                    tmp.y = threadResults[i + 1] + tmp.y;
                    tmp.z = threadResults[i + 2] + tmp.z;
                    tmp.w = threadResults[i + 3] + tmp.w;
                    // write back
                    reinterpret_cast<float4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

void runSgemmWarptiling(int M, int N, int K, float* A, float* B, float* C) {
    // Settings for A6000
    const uint K10_NUM_THREADS = 128;
    const uint K10_BN = 128;
    const uint K10_BM = 128;
    const uint K10_BK = 16;
    const uint K10_WN = 64;
    const uint K10_WM = 64;
    const uint K10_WNITER = 4;
    const uint K10_TN = 4;
    const uint K10_TM = 8;
    dim3 blockDim(K10_NUM_THREADS);

    constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

    dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
    sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
        K10_TN, K10_NUM_THREADS>
        << <gridDim, blockDim >> > (M, N, K, A, B, C);
}

// Check result on the CPU
void verify_result(vector<float>& a, vector<float>& b, vector<float>& c, int N) {
    // For every row...
    for (int i = 0; i < N; i++) {
        // For every column...
        for (int j = 0; j < N; j++) {
            // For every element in the row-column pair
            float tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check against the CPU result
            assert(tmp == c[i * N + j]);
        }
    }
}

void verify_resultv1(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int N) {
    const float epsilon = 1e-3f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }

            if (std::abs(tmp - c[i * N + j]) > epsilon) {
                printf("Mismatch at (%d, %d): expected %.5f, got %.5f\n", i, j, tmp, c[i * N + j]);
                exit(1); // or return; or break;
            }
        }
    }
    printf("All results matched!\n");
}


int main() {
    // Matrix size of N x N;
    int N = 8192;

    // Size (in bytes) of matrix
    size_t bytes = N * N * sizeof(float);

    //Nbr of Floating Operations
    float Nbr_GFLOPS;
    Nbr_GFLOPS = 2 * N / 1000.0 * N / 1000.0 * N / 1000.0;

    // Host vectors
    vector<float> h_a(N * N);
    vector<float> h_b(N * N);
    vector<float> h_c(N * N);

    cout << "Step1 : h_a and h_b generation \n";

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    cout << "Step2 : Mem Allocation on host \n";
    // Allocate device memory
    float* d_a, * d_b;
    float* d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cout << "Step3 : Launch Event to measure Time \n";
    // --- start to count execution time of GPU version ---
    float Total_gpu_time, Host2Dev_time, Kernel_time, Dev2Host_time;
    // some events to count the execution time
    cudaEvent_t start, stop, Host2dev, KernelExec;

    cudaEventCreate(&start);
    cudaEventCreate(&Host2dev);
    cudaEventCreate(&KernelExec);
    cudaEventCreate(&stop);
    // --- execution time of GPU version ---

    cudaEventRecord(start, 0);


    // Copy data to the device
    cout << "Step3 : Copy Data To Device \n";
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(Host2dev, 0);

    // Launch kernel
    runSgemmWarptiling(N, N, N, d_a, d_b, d_c);

    // record time after kernel execution
    cudaEventRecord(KernelExec, 0);


    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);


    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&Total_gpu_time, start, stop);
    cudaEventElapsedTime(&Host2Dev_time, start, Host2dev);
    cudaEventElapsedTime(&Kernel_time, Host2dev, KernelExec);
    cudaEventElapsedTime(&Dev2Host_time, KernelExec, stop);



    printf("Time elapsed on Host To Device Transfer: %f ms.\n\n", Host2Dev_time);
    printf("Time elapsed on matrix multiplication on GPU: %f ms.\n\n", Kernel_time);
    printf("Time elapsed on Device To Host Transfer: %f ms.\n\n", Dev2Host_time);
    printf("Total Time: %f ms.\n\n", Total_gpu_time);


    float Perf_GFLOPS;
    Perf_GFLOPS = Nbr_GFLOPS * 1000 / Kernel_time;
    printf("Kernel Execution Performance: %f GFLOPS.\n\n", Perf_GFLOPS);


    // Check result
    verify_resultv1(h_a, h_b, h_c, N);


    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //wait for keyboard press
    int kml;
    scanf("%c", &kml);

    return 0;
}
