#ifndef KMEANS_H
#define KMEANS_H

#ifdef __cplusplus
extern "C" {
#endif

void kmeans_cuda(const float* data, int numPoints, int dims, int k, int maxIter,
                 float tolerance, int* clusterAssignments, float* centroids);

void kmeans_cpu(const float* data, int numPoints, int dims, int k, int maxIter,
                float tolerance, int* clusterAssignments, float* centroids);

#ifdef __cplusplus
}
#endif

#endif  // KMEANS_H
