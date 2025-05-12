#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>  // For NVIDIA Tensor Core operations
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


void convertFp16ToFp32(float* out, half* in, int n, int m) {
    for (int idx = 0; idx < n * m; idx++) {
        out[idx] = __half2float(in[idx]);
    }
}


__global__ void wmmaMatrixMultiply(half* A, half* B, float* C, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            nvcuda::wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            nvcuda::wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        nvcuda::wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, nvcuda::wmma::mem_row_major);
    }
}



// Check result on the CPU
void verify_result(vector<float>& a, vector<float>& b, vector<float>& c, int N) {
    // For every row...
    const float epsilon = 1e-3f;
    for (int i = 0; i < N; i++) {
        // For every column...
        for (int j = 0; j < N; j++) {
            // For every element in the row-column pair
            float tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check against the CPU result with epsilon due to FP16 precision
            if (std::abs(tmp - c[i * N + j]) > epsilon) {
                printf("Mismatch at (%d, %d): expected %.5f, got %.5f\n", i, j, tmp, c[i * N + j]);
                return;
            }
        }
    }
    printf("All results matched within epsilon tolerance!\n");
}

int main() {
    // Matrix size of N x N
    int N = 8192;
    int M = N;
    int K = N;

    // Size (in bytes) of matrix
    size_t bytes = N * M * sizeof(float);

    //Nbr of Floating Operations
    float Nbr_GFLOPS;
    Nbr_GFLOPS = 2 * N / 1000.0 * N / 1000.0 * N / 1000.0;

    // Host vectors
    vector<half> h_a(N * N);
    vector<half> h_b(N * N);
    vector<float> h_c(N * N);

    cout << "Step1 : h_a and h_b generation \n";

    // Initialize matrices A and B (example: fill with 1.0)
    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);

    cout << "Step2 : Mem Allocation on host \n";
    // Allocate device memory
    half* d_a, * d_b;
    float* d_c;
    cudaMalloc(&d_a, sizeof(half) * M * K);
    cudaMalloc(&d_b, sizeof(half) * K * N);
    cudaMalloc(&d_c, bytes);

    cout << "Step3 : Launch Event to measure Time \n";
    // --- start to count execution time of GPU version ---
    float Total_gpu_time, Host2Dev_time, Kernel_time, Dev2Host_time;
    // some events to count the execution time
    cudaEvent_t start, stop, warmUp, Host2dev, KernelExec;

    cudaEventCreate(&start);
    cudaEventCreate(&Host2dev);
    cudaEventCreate(&warmUp);
    cudaEventCreate(&KernelExec);
    cudaEventCreate(&stop);
    // --- execution time of GPU version ---

    cudaEventRecord(start, 0);

    // Copy data to the device
    cout << "Step4 : Copy Data To Device \n";
    cudaMemcpy(d_a, h_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);

    cudaEventRecord(Host2dev, 0);

    // Warm-up run to amortize initialization overhead
    dim3 gridDim(ceil(static_cast<float>(M) / WMMA_M), ceil(static_cast<float>(N) / WMMA_N), 1);
    dim3 blockDim(32, 32, 1); // 32 threads in x, 32 threads in y    

    // Launch warm-up kernel
    wmmaMatrixMultiply << <gridDim, blockDim >> > (d_a, d_b, d_c, M, N, K);

    // Synchronize to ensure warm-up is complete
    cudaDeviceSynchronize();

    cudaEventRecord(warmUp, 0);

    // Launch actual kernel (Tensor Core implementation)
    wmmaMatrixMultiply << <gridDim, blockDim >> > (d_a, d_b, d_c, M, N, K);

    // Record time after kernel execution
    cudaEventRecord(KernelExec, 0);

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    // Time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute time elapse on GPU computing
    cudaEventElapsedTime(&Total_gpu_time, start, stop);
    cudaEventElapsedTime(&Host2Dev_time, start, Host2dev);
    cudaEventElapsedTime(&Kernel_time, warmUp, KernelExec);
    cudaEventElapsedTime(&Dev2Host_time, KernelExec, stop);

    printf("Time elapsed on Host To Device Transfer: %f ms.\n\n", Host2Dev_time);
    printf("Time elapsed on matrix multiplication on GPU: %f ms.\n\n", Kernel_time);
    printf("Time elapsed on Device To Host Transfer: %f ms.\n\n", Dev2Host_time);
    printf("Total Time: %f ms.\n\n", Total_gpu_time);

    float Perf_GFLOPS;
    Perf_GFLOPS = Nbr_GFLOPS * 1000 / Kernel_time;
    printf("Kernel Execution Performance: %f GFLOPS.\n\n", Perf_GFLOPS);

    // Check result with higher tolerance due to FP16 precision
    vector<float> f_a(N * N);
    vector<float> f_b(N * N);
    convertFp16ToFp32(f_a.data(), h_a.data(), M, K);
    convertFp16ToFp32(f_b.data(), h_b.data(), K, N);
    //verify_result(f_a, f_b, h_c, N);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Wait for keyboard press
    int kml;
    scanf("%c", &kml);

    return 0;
}