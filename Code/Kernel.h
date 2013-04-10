#ifndef KERNEL_H
#define KERNEL_H

#include "common.h"
#include "trace.h"

//ugly but textures cannot be passed into functions so must be declared globally
texture<float, 3, cudaReadModeElementType> tex;

__global__ void DenseImageInterpolateKernel(const size_t width, const size_t height, const float* locIn, float layer, float* valsOut, const size_t numPoints);

__global__ void AffineTransformKernel(const float* tform, const float* pointsIn, float* pointsOut, const size_t numPoints);

__global__ void CameraTransformKernel(const float* tform, const float* cam, const float* pointsIn, float* pointsOut, const size_t numPoints, const bool panoramic);

__global__ void GOMKernel(const float* A, const float* B, const size_t length, float* phaseOut, float* magOut);

void RunBSplineKernel(float* volume, size_t width, size_t height, size_t depth);

#endif //KERNEL_H