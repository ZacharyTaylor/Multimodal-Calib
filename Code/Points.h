#ifndef POINTS_H
#define POINTS_H

#include "trace.h"
#include "cuda.h"

class PointsList {
protected:
	const size_t numPoints_;
	bool onGpu_;

	const float* points_;
	void* d_points_;

public:
	PointsList(float* points, const size_t numPoints);
	PointsList(const size_t numPoints);
	~PointsList();
	size_t GetNumPoints();
	float* GetGpuPointer();
	bool GetOnGpu();
	void AllocateGpu(void);
	void ClearGpu(void);
	void GpuToCpu(void);
	void CpuToGpu(void);
}

class TextureList: public PointsList {
protected:
	const size_t height_;
	const size_t width_;
	const size_t depth_;
public:

	TextureList(float* points, const size_t height = 1, const size_t width = 1, const size_t depth = 1);
	TextureList(const size_t height = 1, const size_t width = 1, const size_t depth = 1);
	~TextureList();
	size_t GetHeight(void);
	size_t GetWidth(void);
	size_t GetDepth(void);
	cudaArray* GetGpuPointer();
	void AllocateGpu(void);
	void ClearGpu(void);
	void GpuToCpu(void);
	void CpuToGpu(void);
	void PrefilterTexture(void);
}

#endif //POINTS_H