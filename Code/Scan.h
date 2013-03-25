#ifndef SCAN_H
#define SCAN_H

#include "Points.h"
#include "Kernel.h"
#include "common.h"

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
	size_t getNumPoints();
	void* getPointsPointer();
};

//sparse scans have location and intesity
class SparseScan: public Scan {
protected:

	PointsList* location_;
	size_t* setDimSize(const size_t numCh, const size_t numPoints);

public:

	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints);
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, PointsList* points, PointsList* location);
	SparseScan(Scan in);
	SparseScan(Scan in, PointsList* location);
	void* GetLocationPointer(void);
};

//dense scan points stored in a little endien (changing first dimension first) grid
class DenseImage: public Scan {
public:

	DenseImage(const size_t height, const size_t width, const size_t numCh = 1);
	DenseImage(const size_t height, const size_t width, const size_t numCh, TextureList* points);
	~DenseImage(void);
	void d_interpolate(SparseScan* scan);

private:

	size_t* setDimSize(const size_t width, const size_t height, const size_t numCh);
};

#endif //SCAN_H