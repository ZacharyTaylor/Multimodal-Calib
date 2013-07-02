#ifndef KERNEL_H
#define KERNEL_H

#include "common.h"
#include "trace.h"

#define TEST_POINTS 30
#define TEST_LOCS 1000

//ugly but textures cannot be passed into functions so must be declared globally
texture<float, 2, cudaReadModeElementType> tex;

__global__ void transformKernel(float *outputData, int width, int height, float theta);

__global__ void generateOutputKernel(float* locs, float* vals, float* out, size_t width, size_t height, size_t depth, size_t numPoints, size_t dilate);

__global__ void AffineTransformKernel(const float* tform, const float* pointsIn, float* pointsOut, const size_t numPoints);

__global__ void CameraTransformKernel(const float* tform, const float* cam, const float* pointsIn, float* pointsOut, const size_t numPoints, const bool panoramic);

__global__ void GOMKernel(const float* A, const float* B, const size_t length, float* phaseOut, float* magOut);

__global__ void livValKernel(const float* A, const float* B, const size_t length, float* out);

__global__ void TestKernel(const float* locs, const float* A, const float* B, const size_t length, float* randNums, float* phaseOut, float* magOut);

void RunBSplineKernel(float* volume, size_t width, size_t height);

#endif //KERNEL_H