#ifndef TFORM_H
#define TFORM_H

#include "common.h"
//#include "Kernel.h"
#include "New\Scans.h"

//! dimensionality of data an affine transform can be used on
#define AFFINE_DIM 2
//! dimensionality of data a camera transform can be used on
#define CAM_DIM 3

//! Holds the properties of the virtual camera, contains the camera matrix and properties for setting it and the type of camera
class Cameras {
private:

	//! structre holding infomation about each camera
	typedef struct {
		//! vector holding camera matrix
		thrust::device_vector<float> cam;

		//! flag for if camera is boolean
		boolean panoramic;
	} cam;

	//! Vector storing camera matrices
	std::vector<cam> camD;

public:
	//! Adds new camera matrices
	void addCams(thrust::device_vector<float> camDIn, boolean panoramic);
	//! Adds new camera matrices
	void addCams(thrust::host_vector<float> camDIn, boolean panoramic);
	//! Clears all the cameras from memory
	void removeAllCameras(void);
	//! Get a pointer to a camera matrix
	/*! /param index of matrix
	*/
	float* getCamP(size_t idx);
	//! Get if camera is panoramic
	/*! /param index of matrix
	*/
	float* getPanoramic(size_t idx);
};

//! Holds the transform matrix and methods for applying it to the data
class Tforms {
protected:
	//! The transform matrices data
	thrust::device_vector<float> tformD;
	const size_t tformSizeX_;
	const size_t tformSizeY_;

public:
	//! Constructs tform
	Tforms(size_t tformSizeX, size_t tformSizeY);
	//! Adds new transformation matricies
	void addTforms(thrust::device_vector<float> tformDIn);
	//! Adds new transformation matricies
	void addTforms(thrust::host_vector<float> tformDIn);
	//! Clear all the transforms
	void removeAllTforms(void);
	//! Gets a pointer to the transformation matrices
	float* getTformP(void);
	//! Get size of transform
	size_t getTformSize(void);
	
	//! Transforms the scans coordinates
	/*! \param in the original scans
		\param out generated output scans
		\param imageList holding transform indexs
		\param start index of first point to transform
		\param end index of last point to transform
	*/
	virtual void transform(ScanList* in, ScanList* out, ImageList* index, size_t start, size_t end);
};

//! Places a virtual camera in the scan and projects the points through its lense onto a surface
class CameraTforms: public Tforms {
public:
	//! Clear all the transforms
	void removeAllTforms(void);
	//! Transforms the scans coordinates
	/*! \param in the original scans
		\param out generated output scans
		\param start index of first point to transform
		\param end index of last point to transform
	*/
	void transform(ScanList* in, ScanList* out, ImageList* index, size_t start, size_t end);

private:
	//! Vector to index of camera to use
	thrust::device_vector<size_t> camIdx;
};

//! Performs a simple affine transform on 2D data
class AffineTforms: public Tforms {
public:
	//! Performs the affine transform on a scan
	/*! \param in the original scans
		\param out generated output scans
		\param start index of first point to transform
		\param end index of last point to transform
	*/
	void Transform(ScanList* in, ScanList* out, ImageList* index, size_t start, size_t end);
};

#endif //TFORM_H
