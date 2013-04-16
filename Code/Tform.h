#ifndef TFORM_H
#define TFORM_H

#include "common.h"
#include "Kernel.h"
#include "Scan.h"
#include "trace.h"

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
	virtual void d_Transform(SparseScan* in, SparseScan* out) = 0;
};

class CameraTform: public Tform {
public:

	CameraTform(Camera* cam);
	~CameraTform(void);
	void d_Transform(SparseScan* in, SparseScan* out);

private:

	Camera* cam_;
};

class AffineTform: public Tform {
public:

	AffineTform(void);
	~AffineTform(void);
	void d_Transform(SparseScan* in, SparseScan* out);
};

#endif //TFORM_H