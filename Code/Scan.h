#ifndef SCAN_H
#define SCAN_H

#include "Points.h"
#include "common.h"
#include "trace.h"

//! Number of dimensions a photo has
#define IMAGE_DIM 2

//! Holds the sensors scans and unifies the method for accessing them
class Scan {
protected:
	//! Number of dimensions scan has
	size_t numDim_;
	//! Number of channels of information assosiated with each point in scan
	size_t numCh_;

	//! Array of length numDim_ specifiying the scans size in each dimension
	size_t* dimSize_;
	//! Holds the points of the scan
	PointsList* points_;

public:
	//! Constructor creates an empty scan
	/*!
		\param numDim Number of dimensions scan has
		\param numCh Number of channels of information assosiated with each point in scan
		\param dimSize Array of length numDim_ specifiying the scans size in each dimension
	*/
	Scan(size_t numDim, size_t numCh,  size_t* dimSize);
	Scan(size_t numDim, size_t numCh,  size_t* dimSize, PointsList* points);
	~Scan(void);
	size_t getNumDim(void);
	size_t getNumCh(void);
	size_t getDimSize(size_t i);
	size_t getNumPoints(void);
	PointsList* getPoints(void);
	void setPoints(PointsList* points);
	void SetupGPU(void);
	void ClearGPU(void);
};

//sparse scans have location and intesity
class SparseScan: public Scan {
private:

	static size_t* setDimSize(const size_t numCh, const size_t numDim, const size_t numPoints);

protected:

	PointsList* location_;
	
public:

	static float* SparseScan::GenLocation(size_t numDim, size_t* dimSize);

	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints);
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, PointsList* points, PointsList* location);
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, float* pointsIn, float* locationIn);
	SparseScan(Scan in);
	SparseScan(Scan in, PointsList* location);
	~SparseScan(void);
	void changeNumCh(size_t numCh);
	size_t getNumPoints(void);
	PointsList* GetLocation(void);
};

//dense scan points stored in a little endien (changing first dimension first) grid
class DenseImage: public Scan {
public:

	DenseImage(const size_t width, const size_t height, const size_t numCh, TextureList* points);
	DenseImage(const size_t width, const size_t height, const size_t numCh, float* pointsIn);
	~DenseImage(void);
	TextureList* getPoints(void);
	void d_interpolate(SparseScan* scan);

private:

	static size_t* setDimSize(const size_t width, const size_t height, const size_t numCh);
};

#endif //SCAN_H