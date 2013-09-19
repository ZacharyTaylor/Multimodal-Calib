#ifndef KERNEL_H
#define KERNEL_H

#include "common.h"

//ugly but textures cannot be passed into functions so must be declared globally
texture<float, 2, cudaReadModeElementType> tex;

__global__ void CameraTransformKernel(const float* const tforms,
									  const size_t* const tformIdx,
									  const float* const cams,
									  const bool* const pans,
									  const size_t* const camIdx,
									  const float* const xIn,
									  const float* const yIn,
									  const float* const zIn,
									  const float* const pointsIdx,
									  const size_t numScans,
									  float* const xOut,
									  float* const yOut,
									  const size_t numPoints,
									  const size_t offset);

__global__ void generateOutputKernel(float* locs, float* vals, float* out, size_t width, size_t height, size_t depth, size_t numPoints, size_t dilate);

__global__ void AffineTransformKernel(const float* tform, const float* pointsIn, float* pointsOut, const size_t numPoints);

__global__ void CameraTransformKernel(const float* tform, const float* cam, const float* pointsIn, float* pointsOut, const size_t numPoints, const bool panoramic);

__global__ void SSDKernel(const float* A, const float* B, const size_t length, float* out, float* zeroEl);

__global__ void GOMKernel(const float* A, const float* B, const size_t length, float* phaseOut, float* magOut);

__global__ void livValKernel(const float* A, const float* B, const float* Bavg, const size_t length, float* out);

void RunBSplineKernel(float* volume, size_t width, size_t height);

#endif //KERNEL_H