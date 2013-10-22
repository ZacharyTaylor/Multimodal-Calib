#ifndef TFORM_H
#define TFORM_H

#include "common.h"
#include "Cameras.h"
#include "ScanList.h"
#include "ImageList.h"

//! dimensionality of data an affine transform can be used on
#define AFFINE_DIM 2
//! dimensionality of data a camera transform can be used on
#define CAM_DIM 3

//! Holds the transform matrix and methods for applying it to the data
class Tforms {
protected:

	//! structre holding infomation about each camera
	typedef struct {
		//! vector holding transform matrix
		thrust::device_vector<float> tform;

		size_t tformSizeX;
		size_t tformSizeY;

	} tform;

	//! Vector storing transform matrices
	std::vector<tform> tformD;

public:
	//! Adds new transformation matricies
	void addTforms(thrust::device_vector<float> tformDIn, size_t tformSizeX, size_t tformSizeY);
	//! Adds new transformation matricies
	void addTforms(thrust::host_vector<float> tformDIn, size_t tformSizeX, size_t tformSizeY);
	//! Clear all the transforms
	void removeAllTforms(void);
	//! Gets a pointer to the transformation matrices
	/*! /param index of matrix
	*/
	float* getTformP(size_t idx);
	//! Get size of transform
	/*! /param index of matrix
	*/
	size_t getTformSize(size_t idx);
	
	//! Transforms the scans coordinates
	/*! \param scans the original scans
		\param cam holds cameras needed for transform
		\param tformIdx index of transform to use
		\param camIdx index of camera to use
		\param scanIdx index of scan to use
	*/
	virtual void transform(ScanList* scans, Cameras* cam, size_t tformIdx, size_t camIdx, size_t scanIdx);
};

//! Places a virtual camera in the scan and projects the points through its lense onto a surface
class CameraTforms: public Tforms {
public:
	void addTforms(thrust::device_vector<float> tformDIn);

	void addTforms(thrust::host_vector<float> tformDIn);

	//! Transforms the scans coordinates
	/*! \param scans the original scans
		\param cam holds cameras needed for transform
		\param tformIdx index of transform to use
		\param camIdx index of camera to use
		\param scanIdx index of scan to use
	*/
	void transform(ScanList* scans, Cameras* cam, size_t tformIdx, size_t camIdx, size_t scanIdx);
};

//! Performs a simple affine transform on 2D data
class AffineTforms: public Tforms {
public:

	void addTforms(thrust::device_vector<float> tformDIn);

	void addTforms(thrust::host_vector<float> tformDIn);

	//! Transforms the scans coordinates
	/*! \param scans the original scans
		\param cam holds cameras needed for transform
		\param tformIdx index of transform to use
		\param camIdx index of camera to use
		\param scanIdx index of scan to use
	*/
	void transform(ScanList* scans, Cameras* cam, size_t tformIdx, size_t camIdx, size_t scanIdx);
};

#endif //TFORM_H
