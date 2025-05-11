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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int N, float* A, float* B, float* C) {
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	// BN/TN are the number of threads to span a column
	const int threadCol = threadIdx.x % (BN / TN);
	const int threadRow = threadIdx.x / (BN / TN);

	// allocate space for the current blocktile in smem
	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];

	// Move blocktile to beginning of A's row and B's column
	A += cRow * BM * N;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	// calculating the indices that this thread will load into SMEM
	// we'll load 128bit / 32bit = 4 elements per thread at each step
	const uint innerRowA = threadIdx.x / (BK / 4);
	const uint innerColA = threadIdx.x % (BK / 4);
	const uint innerRowB = threadIdx.x / (BN / 4);
	const uint innerColB = threadIdx.x % (BN / 4);

	// allocate thread-local cache for results in registerfile
	float threadResults[TM * TN] = { 0.0 };
	float regM[TM] = { 0.0 };
	float regN[TN] = { 0.0 };

	// outer-most loop over block tiles
	for (uint bkIdx = 0; bkIdx < N; bkIdx += BK) {
		// populate the SMEM caches
		// transpose A while loading it
		float4 tmp =
			reinterpret_cast<float4*>(&A[innerRowA * N + innerColA * 4])[0];
		As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
		As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
		As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
		As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

		reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
			reinterpret_cast<float4*>(&B[innerRowB * N + innerColB * 4])[0];
		__syncthreads();

		// advance blocktile
		A += BK;     // move BK columns to right
		B += BK * N; // move BK rows down

		// calculate per-thread results
		for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
			// block into registers
			for (uint i = 0; i < TM; ++i) {
				regM[i] = As[dotIdx * BM + threadRow * TM + i];
			}
			for (uint i = 0; i < TN; ++i) {
				regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
			}
			for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[resIdxM * TN + resIdxN] +=
						regM[resIdxM] * regN[resIdxN];
				}
			}
		}
		__syncthreads();
	}

	// write out the results
	for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
		for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
			// load C vector into registers
			float4 tmp = reinterpret_cast<float4*>(
				&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
			// perform GEMM update in reg
			tmp.x = threadResults[resIdxM * TN + resIdxN] + tmp.x;
			tmp.y = threadResults[resIdxM * TN + resIdxN + 1] + tmp.y;
			tmp.z = threadResults[resIdxM * TN + resIdxN + 2] + tmp.z;
			tmp.w = threadResults[resIdxM * TN + resIdxN + 3] + tmp.w;
			// write back
			reinterpret_cast<float4*>(
				&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
				tmp;
		}
	}
}

void runSgemmVectorize(int N, float* A, float* B, float* C) {
	const int BK = 8;
	const int TM = 8;
	const int TN = 8;

	const int BM = 128;
	const int BN = 128;
	dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(N, BM));
	dim3 blockDim((BM * BN) / (TM * TN));
	sgemmVectorize<BM, BN, BK, TM, TN>
		<< <gridDim, blockDim >> > (N, A, B, C);

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
	runSgemmVectorize(N, d_a, d_b, d_c);

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
