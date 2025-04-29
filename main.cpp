#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "kmeans.h"

float randFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}

int main() {
    const int numPoints = 10000;
    const int dims = 3;
    const int k = 5;
    const int maxIter = 100;
    const float tolerance = 1e-4f;

    float* data = new float[numPoints * dims];
    float* centroids_cpu = new float[k * dims];
    float* centroids_gpu = new float[k * dims];
    int* assignments_cpu = new int[numPoints];
    int* assignments_gpu = new int[numPoints];

    srand(static_cast<unsigned>(time(NULL)));
    for (int i = 0; i < numPoints * dims; i++) {
        data[i] = randFloat();
    }

    for (int c = 0; c < k; c++) {
        int idx = rand() % numPoints;
        for (int d = 0; d < dims; d++) {
            centroids_cpu[c * dims + d] = data[idx * dims + d];
            centroids_gpu[c * dims + d] = data[idx * dims + d];
        }
    }

    printf("Running CPU K-means...\n");
    kmeans_cpu(data, numPoints, dims, k, maxIter, tolerance, assignments_cpu, centroids_cpu);

    printf("Running CUDA K-means...\n");
    kmeans_cuda(data, numPoints, dims, k, maxIter, tolerance, assignments_gpu, centroids_gpu);

    printf("\nSample cluster assignments (first 10 points):\n");
    for (int i = 0; i < 10; i++) {
        printf("Point %d: CPU=%d, CUDA=%d\n", i, assignments_cpu[i], assignments_gpu[i]);
    }

    printf("\nFinal centroids (CPU vs CUDA):\n");
    for (int c = 0; c < k; c++) {
        printf("Cluster %d: CPU: (", c);
        for (int d = 0; d < dims; d++) {
            printf("%f ", centroids_cpu[c * dims + d]);
        }
        printf(") | CUDA: (");
        for (int d = 0; d < dims; d++) {
            printf("%f ", centroids_gpu[c * dims + d]);
        }
        printf(")\n");
    }

    delete[] data;
    delete[] centroids_cpu;
    delete[] centroids_gpu;
    delete[] assignments_cpu;
    delete[] assignments_gpu;

    return 0;
}
