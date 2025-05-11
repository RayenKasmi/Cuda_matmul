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

__global__ void matrixMulXrow(const float* a, const float* b, float* c, int N) {
	// Compute each thread's global row and column index
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	float tmp = 0.0f;
	// Iterate over row, and down column
	for (int k = 0; k < N; k++) {
		// Accumulate results for a single element
		tmp += a[row * N + k] * b[k * N + col];
	}
	c[row * N + col] = tmp;

}

/* =========================================================================== */

__global__ void matrixMulYrow(const float* a, const float* b, float* c, int N) {
	// Compute each thread's global row and column index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float tmp = 0.0f;
	// Iterate over row, and down column
	for (int k = 0; k < N; k++) {
		// Accumulate results for a single element
		tmp += a[row * N + k] * b[k * N + col];
	}
	c[row * N + col] = tmp;
}

const int TILE_SIZE = 32; // Must evenly divide N
const int SHMEM_SIZE = TILE_SIZE * TILE_SIZE;

__global__ void matrixMulTiled(const float* a, const float* b, float* c, int N) {
	// 2D thread index within block
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Global row and column this thread is responsible for
	int row = blockIdx.y * TILE_SIZE + ty;
	int col = blockIdx.x * TILE_SIZE + tx;

	// Accumulator for the result
	float tmp = 0.0f;

	// Statically allocated shared memory for tiles
	__shared__ float Asub[TILE_SIZE][TILE_SIZE];
	__shared__ float Bsub[TILE_SIZE][TILE_SIZE];

	// Loop over all tiles
	for (int t = 0; t < N / TILE_SIZE; ++t) {
		// Load tile elements into shared memory
		Asub[ty][tx] = a[row * N + t * TILE_SIZE + tx];
		Bsub[ty][tx] = b[(t * TILE_SIZE + ty) * N + col];

		__syncthreads(); // Wait for all threads to load

		// Multiply the two tiles together
		for (int k = 0; k < TILE_SIZE; ++k) {
			tmp += Asub[ty][k] * Bsub[k][tx];
		}

		__syncthreads(); // Wait before loading new tiles
	}

	// Write result to global memory
	c[row * N + col] = tmp;
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

// Threads per CTA dimension
int THREADS = 32;

// Blocks per grid dimension (assumes THREADS divides N evenly)
int BLOCKS = N / THREADS;

// Use dim3 structs for block  and grid dimensions
dim3 threads(THREADS, THREADS);
dim3 blocks(BLOCKS, BLOCKS);

// Launch kernel
matrixMulYrow <<<blocks, threads >>> (d_a, d_b, d_c, N);

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


/*
#include <cuda_runtime.h>
#include <iostream>

int main() {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		std::cout << "No CUDA devices found.\n";
		return 1;
	}

	for (int device = 0; device < deviceCount; ++device) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, device);

		std::cout << "===== GPU Device #" << device << " =====\n";
		std::cout << "Name: " << prop.name << "\n";
		std::cout << "Architecture (Compute Capability): " << prop.major << "." << prop.minor << "\n";
		std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
		std::cout << "Multiprocessor Count (SMs): " << prop.multiProcessorCount << "\n";

		std::cout << "--- Warp and Thread Info ---\n";
		std::cout << "Warp Size (Threads per Warp): " << prop.warpSize << "\n";
		std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
		std::cout << "Max Warps per SM: " << prop.maxThreadsPerMultiProcessor / prop.warpSize << "\n";


		std::cout << "--- Block/Thread Info ---\n";
		std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
		std::cout << "Max Block Dimensions: [" << prop.maxThreadsDim[0] << ", "
			<< prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
		std::cout << "Max Grid Dimensions: [" << prop.maxGridSize[0] << ", "
			<< prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";

		std::cout << "--- Memory Info ---\n";
		std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
		std::cout << "Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB\n";
		std::cout << "Registers per Block: " << prop.regsPerBlock << "\n";
		std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024.0 << " KB\n";
		std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
		std::cout << "Memory Clock Rate: " << (prop.memoryClockRate / 1000.0) << " MHz\n";

		std::cout << "--- Misc ---\n";
		std::cout << "Clock Rate: " << prop.clockRate / 1000.0 << " MHz\n";
		std::cout << "Concurrent Kernels Supported: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
		std::cout << "Compute Mode: " << prop.computeMode << "\n";

		std::cout << "========================================\n\n";
	}

	return 0;
}
*/

