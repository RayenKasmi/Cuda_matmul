// This program computes a simple version of matrix multiplication

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

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

const int BM = 64;
const int BN = 64;
const int BK = 8;
const int TM = 8;
  
__global__ void sgemm1DBlocktiling(const float *A, const float *B, float *C, int N) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * N;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const int innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const int innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const int innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (int bkIdx = 0; bkIdx < N; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * N + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        threadResults[resIdx] +
        C[(threadRow * TM + resIdx) * N + threadCol];
  }
}
/* =========================================================================== */

// Check result on the CPU
void verify_result(vector<float>& a, vector<float>& b, vector<float>& c, int N) {
	// For every row...
	for (int i = 0; i < N; i++) {
		// For every column...
		for (int j = 0; j < N; j++) {
			// For every element in the row-column pair
			int tmp = 0;
			for (int k = 0; k < N; k++) {
				// Accumulate the partial results
				tmp += a[i * N + k] * b[k * N + j];
			}

			// Check against the CPU result
			assert(tmp == c[i * N + j]);
		}
	}
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
	float *d_c;
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


// Use dim3 structs for block  and grid dimensions
dim3 threads(BN*BM/TM);
dim3 blocks(N / BN, N / BM);

// Launch kernel
sgemm1DBlocktiling <<<blocks, threads >>> (d_a, d_b, d_c, N);

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
verify_result(h_a, h_b, h_c, N);

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
