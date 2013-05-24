#ifndef PAIR_H
#define PAIR_H

#include "Points.h"
#include "Scan.h"
#include "Tform.h"
#include "common.h"
#include "trace.h"

class Pair{
private:
	DenseImage* base_;
	SparseScan* move_;

	SparseScan* gen_;

	Tform* tform_;

public:

	void SetMove(const size_t numDim, const size_t numCh,  const size_t numPoints, float* pointsIn, float* locIn);
	void SetMove(const size_t numDim, const size_t numCh,  const size_t numPoints, float* pointsIn);
	void MoveSetupGpu(void);
	void SetupAffineTransform();
	void transform(float* tform);
};


#endif //PAIR_H