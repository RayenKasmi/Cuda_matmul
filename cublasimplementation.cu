#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

#define CHECK_CUBLAS(call) { cublasStatus_t stat = (call); if (stat != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS Error!" << std::endl; exit(EXIT_FAILURE); } }

/* =========================================================================== */

// Check result on the CPU
void verify_result(vector<float>& a, vector<float>& b, vector<float>& c, int N) {
	// For every row...
	float epsilon = 0.001;
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
			assert(abs(tmp - c[i * N + j]) < epsilon);
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


	// cuBLAS handle
	cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

	// Matrix multiplication using cuBLAS: C = alpha * A x B + beta * C
	const float alpha = 1.0f;
	const float beta = 0.0f;

	// warm up run
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_b, CUDA_R_32F,
		N, d_a, CUDA_R_32F, N, &beta, d_c, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP);

	cudaEventRecord(Host2dev, 0);
	//actual run
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_b, CUDA_R_32F,
		N, d_a, CUDA_R_32F, N, &beta, d_c, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP);

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

