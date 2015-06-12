#ifndef TFORM_H
#define TFORM_H

#include "common.h"
#include "Cameras.h"
#include "ScanList.h"
#include "ImageList.h"
#include "GenList.h"

//! dimensionality of data an affine transform can be used on
#define AFFINE_DIM 2
//! dimensionality of data a camera transform can be used on
#define CAM_DIM 3

//! Holds the transform matrix and methods for applying it to the data
class Tforms {
protected:

	//! structre holding infomation about each camera
	typedef struct tformInit{
		//! vector holding transform matrix
		thrust::device_vector<float> tform;

		size_t tformSizeX;
		size_t tformSizeY;

		tformInit(){};

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
		\param gen holds generated scan values
		\param tformIdx index of transform to use
		\param camIdx index of camera to use
		\param scanIdx index of scan to use
		\param genIdx index of generated scan to use
	*/
	virtual void transform(ScanList* scans, Cameras* cam, GenList* gen, size_t tformIdx, size_t camIdx, size_t scanIdx, size_t genIdx);
};

//! Places a virtual camera in the scan and projects the points through its lense onto a surface
class CameraTforms: public Tforms {
public:
	void addTforms(thrust::device_vector<float> tformDIn);

	void addTforms(thrust::host_vector<float> tformDIn);

	//! Transforms the scans coordinates
	/*! \param scans the original scans
		\param cam holds cameras needed for transform
		\param gen holds generated scan values
		\param tformIdx index of transform to use
		\param camIdx index of camera to use
		\param scanIdx index of scan to use
		\param genIdx index of generated scan to use
	*/
	void transform(ScanList* scans, Cameras* cam, GenList* gen, size_t tformIdx, size_t camIdx, size_t scanIdx, size_t genIdx);
};

//! Performs a simple affine transform on 2D data
class AffineTforms: public Tforms {
public:

	void addTforms(thrust::device_vector<float> tformDIn);

	void addTforms(thrust::host_vector<float> tformDIn);

	//! Transforms the scans coordinates
	/*! \param scans the original scans
		\param cam holds cameras needed for transform
		\param gen holds generated scan values
		\param tformIdx index of transform to use
		\param camIdx index of camera to use
		\param scanIdx index of scan to use
		\param genIdx index of generated scan to use
	*/
	void transform(ScanList* scans, Cameras* cam, GenList* gen, size_t tformIdx, size_t camIdx, size_t scanIdx, size_t genIdx);
};

#endif //TFORM_H
