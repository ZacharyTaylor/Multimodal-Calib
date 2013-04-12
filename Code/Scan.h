#ifndef SCAN_H
#define SCAN_H

#include "Points.h"
#include "common.h"
#include "trace.h"

#define IMAGE_DIM 2

class Scan {
protected:
	const size_t numDim_;
	const size_t numCh_;

	const size_t* dimSize_;
	PointsList* points_;

public:

	Scan(const size_t numDim, const size_t numCh,  const size_t* dimSize);
	Scan(const size_t numDim, const size_t numCh,  const size_t* dimSize, PointsList* points);
	size_t getNumDim(void);
	size_t getNumCh(void);
	size_t getDimSize(size_t i);
	size_t getNumPoints(void);
	PointsList* getPoints(void);
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
	size_t getNumPoints(void);
	PointsList* GetLocation(void);
};

//dense scan points stored in a little endien (changing first dimension first) grid
class DenseImage: public Scan {
public:

	DenseImage(const size_t width, const size_t height, const size_t numCh, TextureList* points);
	DenseImage(const size_t width, const size_t height, const size_t numCh, float* pointsIn);
	TextureList* getPoints(void);
	void d_interpolate(SparseScan* scan);

private:

	static size_t* setDimSize(const size_t width, const size_t height, const size_t numCh);
};

#endif //SCAN_H