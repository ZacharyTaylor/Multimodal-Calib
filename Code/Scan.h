#ifndef SCAN_H
#define SCAN_H

#include "Points.h"
#include "trace.h"

class Scan {
protected:
	const size_t numDim_;
	const size_t numCh_;

	const size_t* dimSize_;
	
	PointsList* points_;

public:
	Scan(const size_t numDim, const size_t numCh,  const size_t* dimSize);

	size_t getNumDim(void);
	size_t getNumCh(void);
	size_t getDimSize(size_t i);
	size_t getNumPoints();
	float* getPointsPointer();
};

//dense scan points stored in a little endien (changing first dimension first) grid
class DenseScan: public Scan {
public:
	DenseScan(const size_t numDim, const size_t numCh,  const size_t* dimSize);
	DenseScan(const size_t numDim, const size_t numCh,  const size_t* dimSize, float* points);
};

//sparse scans have location and intesity
class SparseScan: public Scan {
protected:

	PointsList* location_;
	size_t* setDimSize(const size_t numDim, const size_t numCh, const size_t numPoints);

public:
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints);
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, PointsList* points, PointsList* location);
	SparseScan(DenseScan in);
	SparseScan(DenseScan in, PointsList* location);
};

#endif //SCAN_H