#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "kmeans.h"

void kmeans_cpu(const float* data, int numPoints, int dims, int k, int maxIter,
                float tolerance, int* assignments, float* centroids)
{
    float* newCentroids = (float*)malloc(k * dims * sizeof(float));
    int* counts = (int*)malloc(k * sizeof(int));

    for (int iter = 0; iter < maxIter; iter++) {
        for (int i = 0; i < numPoints; i++) {
            const float* point = data + i * dims;
            float bestDist = 1e30f;
            int bestCluster = -1;
            for (int c = 0; c < k; c++) {
                float dist = 0.0f;
                for (int d = 0; d < dims; d++) {
                    float diff = point[d] - centroids[c * dims + d];
                    dist += diff * diff;
                }
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }
            }
            assignments[i] = bestCluster;
        }

        for (int c = 0; c < k; c++) {
            counts[c] = 0;
            for (int d = 0; d < dims; d++) {
                newCentroids[c * dims + d] = 0.0f;
            }
        }

        for (int i = 0; i < numPoints; i++) {
            int cluster = assignments[i];
            counts[cluster]++;
            for (int d = 0; d < dims; d++) {
                newCentroids[cluster * dims + d] += data[i * dims + d];
            }
        }

        bool converged = true;
        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) continue;
            for (int d = 0; d < dims; d++) {
                float newVal = newCentroids[c * dims + d] / counts[c];
                if (fabs(newVal - centroids[c * dims + d]) > tolerance) {
                    converged = false;
                }
                centroids[c * dims + d] = newVal;
            }
        }
        if (converged) {
            printf("CPU K-means converged at iteration %d\n", iter);
            break;
        }
    }

    free(newCentroids);
    free(counts);
}
