#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "kmeans.h"

#define CUDA_CHECK(err) { \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        fprintf(stderr, "Error Code: %d\n", err); \
        fprintf(stderr, "File: %s\n", __FILE__); \
        fprintf(stderr, "Line: %d\n", __LINE__); \
        fprintf(stderr, "Function: %s\n", __func__); \
        exit(EXIT_FAILURE); \
    } \
}

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_DIMS = 1024;

__global__
void assign_clusters(const float* __restrict__ data, 
                    const float* __restrict__ centroids,
                    int numPoints, int dims, int k, 
                    int* assignments)
{
    extern __shared__ float sharedCentroids[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = tid; i < k * dims; i += blockDim.x) {
        sharedCentroids[i] = centroids[i];
    }
    __syncthreads();

    if (gid < numPoints) {
        const float* point = &data[gid * dims];
        float minDist = INFINITY;
        int bestCluster = 0;

        #pragma unroll 4
        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            const float* centroid = &sharedCentroids[c * dims];
            
            for (int d = 0; d < dims; d++) {
                float diff = point[d] - centroid[d];
                dist += diff * diff;
            }

            if (dist < minDist) {
                minDist = dist;
                bestCluster = c;
            }
        }
        assignments[gid] = bestCluster;
    }
}

__global__
void update_centroids(const float* __restrict__ data,
                     const int* __restrict__ assignments,
                     float* centroidSums,
                     int* counts,
                     int numPoints,
                     int dims)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numPoints) {
        const int cluster = assignments[idx];
        const float* point = &data[idx * dims];
        
        atomicAdd(&counts[cluster], 1);
        
        for (int d = 0; d < dims; d++) {
            atomicAdd(&centroidSums[cluster * dims + d], point[d]);
        }
    }
}

void kmeans_cuda(const float* data, int numPoints, int dims, int k, int maxIter,
                 float tolerance, int* clusterAssignments, float* centroids)
{
    if (!data || !clusterAssignments || !centroids || 
        numPoints <= 0 || dims <= 0 || k <= 0 || k > numPoints) {
        fprintf(stderr, "Invalid input parameters\n");
        return;
    }

    const size_t dataSize = numPoints * dims * sizeof(float);
    const size_t centroidsSize = k * dims * sizeof(float);
    const size_t assignmentsSize = numPoints * sizeof(int);
    const size_t countsSize = k * sizeof(int);

    float *d_data = nullptr;
    float *d_centroids = nullptr;
    float *d_newCentroids = nullptr;
    int   *d_assignments = nullptr;
    int   *d_counts = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_data, dataSize));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, centroidsSize));
    CUDA_CHECK(cudaMalloc((void**)&d_newCentroids, centroidsSize));
    CUDA_CHECK(cudaMalloc((void**)&d_assignments, assignmentsSize));
    CUDA_CHECK(cudaMalloc((void**)&d_counts, countsSize));

    CUDA_CHECK(cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids, centroidsSize, cudaMemcpyHostToDevice));

    const int blocks = (numPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t sharedMemSize = k * dims * sizeof(float);

    float* h_newCentroids = (float*)malloc(centroidsSize);
    int* h_counts = (int*)malloc(countsSize);
    if (!h_newCentroids || !h_counts) {
        fprintf(stderr, "Failed to allocate host memory\n");
        goto cleanup;
    }

    for (int iter = 0; iter < maxIter; iter++) {
        assign_clusters<<<blocks, THREADS_PER_BLOCK, sharedMemSize>>>(
            d_data, d_centroids, numPoints, dims, k, d_assignments
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_newCentroids, 0, centroidsSize));
        CUDA_CHECK(cudaMemset(d_counts, 0, countsSize));

        update_centroids<<<blocks, THREADS_PER_BLOCK>>>(
            d_data, d_assignments, d_newCentroids, d_counts, numPoints, dims
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_newCentroids, d_newCentroids, centroidsSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts, d_counts, countsSize, cudaMemcpyDeviceToHost));

        bool converged = true;
        for (int c = 0; c < k; c++) {
            if (h_counts[c] == 0) continue;
            for (int d = 0; d < dims; d++) {
                float newVal = h_newCentroids[c * dims + d] / h_counts[c];
                if (fabs(newVal - centroids[c * dims + d]) > tolerance) {
                    converged = false;
                }
                centroids[c * dims + d] = newVal;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_centroids, centroids, centroidsSize, cudaMemcpyHostToDevice));

        if (converged) {
            printf("CUDA K-means converged at iteration %d\n", iter);
            break;
        }
    }

    CUDA_CHECK(cudaMemcpy(clusterAssignments, d_assignments, assignmentsSize, cudaMemcpyDeviceToHost));

cleanup:
    if (h_newCentroids) free(h_newCentroids);
    if (h_counts) free(h_counts);
    if (d_data) cudaFree(d_data);
    if (d_centroids) cudaFree(d_centroids);
    if (d_newCentroids) cudaFree(d_newCentroids);
    if (d_assignments) cudaFree(d_assignments);
    if (d_counts) cudaFree(d_counts);
}
