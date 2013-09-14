#ifndef TFORM_H
#define TFORM_H

#include "common.h"
//#include "Kernel.h"
#include "Scan.h"
#include "trace.h"

//! width of camera projection matrix
#define CAM_WIDTH 4
//! height of camera projection matrix
#define CAM_HEIGHT 3

//! dimensionality of data an affine transform can be used on
#define AFFINE_DIM 2
//! dimensionality of data a camera transform can be used on
#define CAM_DIM 3

//! Holds the properties of the virtual camera, contains the camera matrix and properties for setting it and the type of camera
class Camera {
private:
	//! The camera matrix stored in device memory
	float* d_camera_;
	//! True for panoramic camera, false otherwise
	const bool panoramic_;

public:
	//! Sets up camera, takes in if camera is panoramic
	Camera(bool panoramic);
	//! Destructor clears gpu memory
	~Camera(void);
	//! Sets a new camera matrix
	void SetCam(float* cam);
	//! Gets a pointer to the current camera matrix
	float* d_GetCam(void);
	//! Returns if the camera is panoramic
	bool IsPanoramic(void);
};

//! Holds the transform matrix and methods for applying it to the data
class Tform {
protected:
	//! The transform matrix stored in device memory
	float* d_tform_;
	//! The transform matrix is an n by n matrix, this is the value of n
	size_t sizeTform_;

public:
	//! Constructor, takes in size of matrix
	Tform(size_t sizeTform);
	//! Destructor clears gpu memory
	~Tform(void);
	//! Sets a new transformation matrix
	void SetTform(float* tform);
	//! Gets a pointer to the transformation matrix
	float* d_GetTform(void);
	//! Transforms a scans coordinates
	virtual void d_Transform(SparseScan* in, SparseScan** out, cudaStream_t* stream) = 0;
};

//! Places a virtual camera in the scan and projects the points through its lense onto a surface
class CameraTform: public Tform {
public:
	//! Creates a transform, specifies the camera it will be operating on
	CameraTform(Camera* cam);
	//! Destructor clears transform, leaves camera alone as may be used by multiple transforms
	~CameraTform(void);
	//! Performs the camera transform on a scan
	/*!
		\param in The input scan, must be 3D, this operation does not modify it
		\param out The output scan, of the same size as in, memory must be preallocated
	*/
	void d_Transform(SparseScan* in, SparseScan** out, cudaStream_t* stream);

private:
	//! Pointer to the camera being used
	Camera* cam_;
};

//! Performs a simple affine transform on 2D data
class AffineTform: public Tform {
public:
	//! Simple affine constructor
	AffineTform(void);
	//! Destructor
	~AffineTform(void);
	//! Performs the affine transform on a scan
	/*!
		\param in The input scan, must be 2D, this operation does not modify it
		\param out The output scan, of the same size as in, memory must be preallocated
	*/
	void d_Transform(SparseScan* in, SparseScan** out, cudaStream_t* stream);
};

#endif //TFORM_H
