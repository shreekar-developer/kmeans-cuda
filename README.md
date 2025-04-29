# K-means Clustering with CUDA

This project implements the K-means clustering algorithm with both CPU and CUDA (GPU) versions for performance comparison. K-means is an unsupervised learning algorithm that partitions data points into K clusters based on their similarity.

## Features

- CPU implementation for reference
- CUDA-accelerated GPU implementation
- Performance comparison between CPU and GPU versions
- Support for multi-dimensional data points
- Configurable number of clusters and iterations
- Convergence checking with tolerance threshold

## Building

1. Ensure CUDA Toolkit is installed and properly configured
2. Navigate to the project directory
3. Compile using nvcc:

```bash
nvcc -o kmeans_cuda kmeans_cuda.cu kmeans_cpu.cpp main.cpp -std=c++11
```

## Usage

The program will automatically:
1. Generate random test data
2. Run the CPU implementation
3. Run the CUDA implementation
4. Compare and display results

Default parameters:
- Number of points: 10,000
- Dimensions: 3
- Number of clusters (K): 5
- Maximum iterations: 100
- Convergence tolerance: 1e-4

The program will output:
- Sample cluster assignments for the first 10 points
- Final centroids from both CPU and CUDA implementations
- Convergence information for both implementations

## Implementation Details

### CPU Version
- Standard iterative K-means implementation
- Uses Euclidean distance for point-to-centroid assignment
- Updates centroids as mean of assigned points
- Checks convergence based on centroid movement

### CUDA Version
- Parallel implementation using GPU
- Uses shared memory for centroid caching
- Implements coalesced memory access patterns
- Uses atomic operations for thread-safe updates
- Optimized thread organization for maximum parallelism

## Performance Considerations

The CUDA implementation is optimized for:
- Memory coalescing
- Shared memory usage
- Thread organization
- Atomic operations
- Divergent branching minimization
