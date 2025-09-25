#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

// Utility macros
#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


// Kernels

// 1 thread 1 element
__global__ void kernel_1t1e(float* A, const float* B, const float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int idx = row * N + col;
        A[idx] = B[idx] + C[idx];
    }
}

// 1 thread 1 row (each thread processes an entire row)
__global__ void kernel_1t1r(float* A, const float* B, const float* C, int N) {
    int threadRow = blockIdx.x * blockDim.x + threadIdx.x; // 1D mapping across rows
    if (threadRow < N) {
        int rowStart = threadRow * N;
        // iterate across columns
        for (int col = 0; col < N; ++col) {
            int idx = rowStart + col;
            A[idx] = B[idx] + C[idx];
        }
    }
}

// 1 thread 1 column (each thread processes an entire column)
__global__ void kernel_1t1c(float* A, const float* B, const float* C, int N) {
    int threadCol = blockIdx.x * blockDim.x + threadIdx.x; // 1D mapping across cols
    if (threadCol < N) {
        for (int row = 0; row < N; ++row) {
            int idx = row * N + threadCol;
            A[idx] = B[idx] + C[idx];
        }
    }
}


// Host helper functions

// CPU reference
void matrix_add_cpu(float* A, const float* B, const float* C, int N) {
    int total = N * N;
    for (int i = 0; i < total; ++i) A[i] = B[i] + C[i];
}

// verification
bool approx_equal(const float* a, const float* b, int N, float eps = 1e-3f) {
    int total = N * N;
    for (int i = 0; i < total; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > eps) {
            // print one mismatch for debugging
            int r = i / N, c = i % N;
            fprintf(stderr, "Mismatch at (%d,%d): gpu=%f cpu=%f diff=%f\n", r, c, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

// choose block/tile sizes using device properties (simple heuristic)
void choose_tile_dims(const cudaDeviceProp& prop, dim3& block1t1e) {
    // prefer 16x16 but don't exceed maxThreadsPerBlock
    int base = 16;
    int maxThreads = prop.maxThreadsPerBlock;
    if (base * base > maxThreads) {
        // pick tile = floor(sqrt(maxThreads))
        int tile = (int)floor(sqrt((double)maxThreads));
        if (tile < 1) tile = 1;
        block1t1e = dim3(tile, 1, 1); // fallback to 1D if necessary
    }
    else {
        block1t1e = dim3(base, base, 1);
    }
}

// Host wrapper: 1 thread per element
float hostMatrixAdd_1t1e(float* h_A, const float* h_B, const float* h_C, int N, const cudaDeviceProp& prop, int runs = 10) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    // choose block and grid
    dim3 block;
    choose_tile_dims(prop, block); // block.x, block.y
    if (block.y == 0) block.y = 1;
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // timing with events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms_total = 0.0f;

    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel_1t1e << <grid, block >> > (d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms_total += ms;
    }
    float avg_ms = ms_total / (float)runs;

    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return avg_ms;
}

// Host wrapper: 1 thread per row
float hostMatrixAdd_1t1r(float* h_A, const float* h_B, const float* h_C, int N, const cudaDeviceProp& prop, int runs = 10) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    int maxThreads = prop.maxThreadsPerBlock;
    int threadsPerBlock = 256;
    if (threadsPerBlock > maxThreads) threadsPerBlock = maxThreads;
    if (threadsPerBlock > N) threadsPerBlock = N;
    dim3 block(threadsPerBlock, 1, 1);
    dim3 grid((N + threadsPerBlock - 1) / threadsPerBlock, 1, 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms_total = 0.0f;

    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel_1t1r << <grid, block >> > (d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms_total += ms;
    }
    float avg_ms = ms_total / (float)runs;

    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return avg_ms;
}

// Host wrapper: 1 thread per column
float hostMatrixAdd_1t1c(float* h_A, const float* h_B, const float* h_C, int N, const cudaDeviceProp& prop, int runs = 10) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    int maxThreads = prop.maxThreadsPerBlock;
    int threadsPerBlock = 256;
    if (threadsPerBlock > maxThreads) threadsPerBlock = maxThreads;
    if (threadsPerBlock > N) threadsPerBlock = N;
    dim3 block(threadsPerBlock, 1, 1);
    dim3 grid((N + threadsPerBlock - 1) / threadsPerBlock, 1, 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms_total = 0.0f;

    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel_1t1c << <grid, block >> > (d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms_total += ms;
    }
    float avg_ms = ms_total / (float)runs;

    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return avg_ms;
}

// Print device properties
void printDeviceProperties(int dev) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %zu MB\n", (size_t)prop.totalGlobalMem / (1024 * 1024));
    printf("  Shared mem per block: %zu KB\n", (size_t)prop.sharedMemPerBlock / 1024);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max grid dimensions: x=%d y=%d z=%d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max threads dim: x=%d y=%d z=%d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  MultiProcessorCount (SMs): %d\n", prop.multiProcessorCount);
    printf("\n");
}

// Main 
int main(int argc, char** argv) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        printDeviceProperties(dev);
    }

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Default runs = 10 
    int runs = 10;
    if (argc >= 3) runs = atoi(argv[2]);

    std::vector<int> sizes;
    if (argc >= 2) {
        sizes.push_back(atoi(argv[1])); // user-specified N
    }
    else {
        sizes = { 512, 1024, 2048, 4096 };
    }

    for (int N : sizes) {
        printf("\nExperiment: N=%d, runs=%d \n", N, runs);

        size_t total = (size_t)N * N;
        size_t bytes = total * sizeof(float);

        float* h_B = (float*)malloc(bytes);
        float* h_C = (float*)malloc(bytes);
        float* h_A_gpu = (float*)malloc(bytes);
        float* h_A_ref = (float*)malloc(bytes);

        if (!h_B || !h_C || !h_A_gpu || !h_A_ref) {
            fprintf(stderr, "Host memory allocation failed\n");
            return 1;
        }

        // initialize B and C with pseudorandom floats in [0,100]
        srand(12345);
        for (size_t i = 0; i < total; ++i) {
            h_B[i] = (float)((rand() / (float)RAND_MAX) * 100.0);
            h_C[i] = (float)((rand() / (float)RAND_MAX) * 100.0);
        }

        // CPU reference
        auto t0 = std::chrono::high_resolution_clock::now();
        matrix_add_cpu(h_A_ref, h_B, h_C, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("CPU: computed reference in %.3f ms\n", cpu_ms);

        // Run 1t1e
        memset(h_A_gpu, 0, bytes);
        float ms_1t1e = hostMatrixAdd_1t1e(h_A_gpu, h_B, h_C, N, prop, runs);
        bool ok1 = approx_equal(h_A_gpu, h_A_ref, N);
        printf("kernel_1t1e: avg time %.3f ms (%s)\n", ms_1t1e, ok1 ? "OK" : "WRONG");

        // Run 1t1r
        memset(h_A_gpu, 0, bytes);
        float ms_1t1r = hostMatrixAdd_1t1r(h_A_gpu, h_B, h_C, N, prop, runs);
        bool ok2 = approx_equal(h_A_gpu, h_A_ref, N);
        printf("kernel_1t1r: avg time %.3f ms (%s)\n", ms_1t1r, ok2 ? "OK" : "WRONG");

        // Run 1t1c
        memset(h_A_gpu, 0, bytes);
        float ms_1t1c = hostMatrixAdd_1t1c(h_A_gpu, h_B, h_C, N, prop, runs);
        bool ok3 = approx_equal(h_A_gpu, h_A_ref, N);
        printf("kernel_1t1c: avg time %.3f ms (%s)\n", ms_1t1c, ok3 ? "OK" : "WRONG");

        // Results Summary
        printf("\nSummary (averaged over %d runs):\n", runs);
        printf("  CPU time:    %.3f ms (single run)\n", cpu_ms);
        printf("  1t1e (elem): %.3f ms\n", ms_1t1e);
        printf("  1t1r (row):  %.3f ms\n", ms_1t1r);
        printf("  1t1c (col):  %.3f ms\n", ms_1t1c);

        free(h_B);
        free(h_C);
        free(h_A_gpu);
        free(h_A_ref);
    }

    return 0;
}
