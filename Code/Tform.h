#ifndef TFORM_H
#define TFORM_H

#include "common.h"
#include "Scan.h"

#define CAM_WIDTH 4
#define CAM_HEIGHT 3

#define AFFINE_DIM 2
#define CAM_DIM 3

class Tform {
protected:
	float* d_tform_;
	size_t sizeTform_;

public:
	virtual SparseScan transform(Scan in);
	Tform(size_t sizeTform);
	void setTform(float* tform);

}

class AffineTform: public Tform {
public:
	SparseScan transform(DenseScan in, float* tformMat);
}

class PinCameraTform: public Tform {
public:
	GpuSparseScan transform(GpuSparseScan in, float* tform, float* cam);

private:
	__global__ void pointsTransformKernel(float* tform, float* cam, const float* pointsIn, float* pointsOut, size_t* dimSize, const bool panoramic);
}

#endif //TFORM_H