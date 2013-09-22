#ifndef TFORM_H
#define TFORM_H

#include "common.h"
#include "Cameras.h"

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

	//! Vector storing camera matrices
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
	/*! \param in the original scans
		\param out generated output scans
		\param imageList holding transform indexs
		\param start index of first point to transform
		\param end index of last point to transform
	*/
	virtual void transform(ScanList* scansIn, std::vector<float*> locOut, size_t tformIdx, size_t camIdx, size_t scanIdx, cudaStream_t streams);
};

//! Places a virtual camera in the scan and projects the points through its lense onto a surface
class CameraTforms: public Tforms {
public:
	
	Cameras cameraStore;

	size_t getCameraIdx(size_t idx);

	void addTforms(thrust::device_vector<float> tformDIn, size_t camIdx);

	void addTforms(thrust::host_vector<float> tformDIn, size_t camIdx);

	//! Transforms the scans coordinates
	/*! \param in the original scans
		\param out generated output scans
		\param start index of first point to transform
		\param end index of last point to transform
	*/
	void transform(ScanList* scansIn, std::vector<float*> locOut, size_t tformIdx, size_t camIdx, size_t scanIdx, cudaStream_t streams);

private:
	std::vector<size_t> cameraIdx;
};

//! Performs a simple affine transform on 2D data
class AffineTforms: public Tforms {
public:

	void addTforms(thrust::device_vector<float> tformDIn);

	void addTforms(thrust::host_vector<float> tformDIn);

	//! Performs the affine transform on a scan
	/*! \param in the original scans
		\param out generated output scans
		\param start index of first point to transform
		\param end index of last point to transform
	*/
	void transform(ScanList* scanIn, std::vector<float*> locOut, ImageList* index, size_t start, cudaStream_t streams);
};

#endif //TFORM_H
