#ifndef KERNEL_H
#define KERNEL_H

#include "common.h"

//ugly but textures cannot be passed into functions so must be declared globally
texture<float, 2, cudaReadModeElementType> tex;

__global__ void CameraTransformKernel(const float* const tform,
									  const float* const cam,
									  const bool const pan,
									  const float* const xIn,
									  const float* const yIn,
									  const float* const zIn,
									  const size_t numPoints,
									  float* const xOut,
									  float* const yOut);

__global__ void LinearInterpolateKernel(const float* const imageIn,
										float* const out,
										const size_t height,
										const size_t width,
										const size_t depth,
										const float* const x,
										const float* const y);

__global__ void generateOutputKernel(float* locs, float* vals, float* out, size_t width, size_t height, size_t depth, size_t numPoints, size_t dilate);

__global__ void AffineTransformKernel(const float* tform, const float* pointsIn, float* pointsOut, const size_t numPoints);

__global__ void CameraTransformKernel(const float* tform, const float* cam, const float* pointsIn, float* pointsOut, const size_t numPoints, const bool panoramic);

__global__ void SSDKernel(const float* A, const float* B, const size_t length, float* out, float* zeroEl);

__global__ void GOMKernel(const float* A, const float* B, const size_t length, float* phaseOut, float* magOut);

__global__ void livValKernel(const float* A, const float* B, const float* Bavg, const size_t length, float* out);

void RunBSplineKernel(float* volume, size_t width, size_t height);

#endif //KERNEL_H