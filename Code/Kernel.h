#ifndef KERNEL_H
#define KERNEL_H

#include "common.h"
#include "trace.h"

//ugly but textures cannot be passed into functions so must be declared globally
texture<float, 2, cudaReadModeElementType> tex;

__global__ void DenseImageInterpolateKernel(const size_t width, const size_t height, const float* locIn, float* valsOut, const size_t numPoints);

__global__ void AffineTransformKernel(const float* tform, const float* pointsIn, float* pointsOut, const size_t numPoints);

__global__ void CameraTransformKernel(const float* tform, const float* cam, const float* pointsIn, float* pointsOut, const size_t numPoints, const bool panoramic);

#endif //KERNEL_H