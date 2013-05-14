#ifndef POINTS_H
#define POINTS_H

#include "common.h"
#include "trace.h"

//!	PointsList class holds the device and host arrays of points. It contains methods for moving the data to and from the gpu and keeps track of the number of elements.
class PointsList {
protected:
	//! number of elements
	const size_t numEntries_;

	//! array of points stored in host memory
	float* points_;
	//! array of points stored in device memory
	void* d_points_;

	//! creates and fills points_
	/*! \sa PointsList() and ~PointsList()
	*/
	static float* PointsSetup(float* points, const size_t numEntries, bool copy);

public:
	//!	Creates a PointsList using given points
	/*!	\param points an array of points
		\param numEntries the number of elements in the array
		\param copy	if true creates a deep copy of float (always use this if points was created in matlab or things will crash if matlab modifies points)
	*/
	PointsList(float* points, const size_t numEntries, bool copy);

	//!	Creates an empty PointsList of size numElements
	/*!	\param numEntries number of entries in PointList
		\sa PointsList() and ~PointsList
	*/
	PointsList(const size_t numEntries);

	//!	Destructor, clears arrays from device and host memory
	~PointsList();

	//!	Returns number of elements in PointsList
	size_t GetNumEntries();

	//! Returns the location of the array on device memory
	void* GetGpuPointer();

	//! Returns the location of the array on host memory
	float* GetCpuPointer();

	//! Returns true if array loaded into device memory
	bool IsOnGpu();

	//! Allocates memory on device
	void AllocateGpu(void);

	//! Clears array from device memory
	/*! Will generate a warning if no array is allocated
	*/
	void ClearGpu(void);

	//! Copies contents of array on device memory to the array on host memory
	/*! If no device memory is allocated will generate a warning and exit without copying anything
	*/
	void GpuToCpu(void);

	//! Copies contents of array on host memory to the array on device memory
	/*! If no memory is allocated will generate a warning before allocating memory and continuing
	*/
	void CpuToGpu(void);
};

//! Expands on PointsList to allow it to store and make use of Cuda textures and texture memory.
class TextureList: public PointsList {

private:
	//! True if currently holding textures in device memory, false if holding an array
	bool texInMem_;
	//! Converts array stored in device memory to textures
	void ArrayToTexture(void);
	//! Filters array so that bspline interpolation is more efficient
	void PrefilterArray(void);
	

protected:
	//! Height of texture
	const size_t height_;
	//! Width of texture
	const size_t width_;
	//! Depth of texture
	const size_t depth_;
public:
	//! Creates a TextureList
	/*! loads an array to device memory, filters it and converts it to a texture
	\param points array of points to load into texture
	\param copy true for a deep copy of points, if false points must have been allocated with new [] or destructor will fail
	\param width width of texture
	\param height height of texture
	\param depth depth of texture
	\remark points array must be of size width*height*depth
	\remark 3D arrays (depth > 1) are stored as depth number of 2D textures
	*/
	TextureList(float* points, bool copy, const size_t width = 1, const size_t height = 1, const size_t depth = 1);

	//! Destructor frees arrays on device and host memory
	~TextureList(void);

	//! Returns the height of the array
	size_t GetHeight(void);

	//! Returns the width of the array
	size_t GetWidth(void);

	//! Returns the depth of the array
	size_t GetDepth(void);

	//! Allocates device memory for the texture
	void AllocateGpu(void);

	//! Clears array from device memory
	/*! Will generate a warning if no array is allocated
	*/
	void ClearGpu(void);

	//! Copies contents of array on device memory to the array on host memory
	/*! If no device memory is allocated will generate a warning and exit without copying anything
	*/
	void GpuToCpu(void);

	//! Copies contents of array on host memory to the array on device memory
	/*! If no memory is allocated will generate a warning before allocating memory and continuing
	*/
	void CpuToGpu(void);
};

#endif //POINTS_H