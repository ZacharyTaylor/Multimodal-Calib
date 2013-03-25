#ifndef TFORM_H
#define TFORM_H

#include "common.h"
#include "Scan.h"

#define CAM_WIDTH 4
#define CAM_HEIGHT 3

#define AFFINE_DIM 2
#define CAM_DIM 3

class Camera {
private:
	
	float* d_camera_;
	const bool panoramic_;

public:

	Camera(bool panoramic);
	~Camera(void);
	void SetCam(float* cam);
	float* d_GetCam(void);
	bool IsPanoramic(void);
};

class Tform {
protected:

	float* d_tform_;
	size_t sizeTform_;

public:

	Tform(size_t sizeTform);
	~Tform(void);
	void SetTform(float* tform);
	float* d_GetTform(void);
	virtual SparseScan* d_Transform(Scan* in);
};

class CameraTform: public Tform {
public:

	CameraTform(Camera* cam);
	void d_Transform(SparseScan* in, SparseScan* out);

private:

	const Camera* d_cam_;

	__global__ void CameraTransformKernel(const float* tform, const float* cam, const float* pointsIn, float* pointsOut, const size_t numPoints, const bool panoramic);
};

class AffineTform: public Tform {
public:

	AffineTform(void);
	void d_Transform(SparseScan* in, SparseScan* out);

private:

	__global__ void AffineTransformKernel(const float* tform, const float* pointsIn, float* pointsOut, const size_t numPoints);
};

#endif //TFORM_H