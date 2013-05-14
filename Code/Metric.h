#ifndef METRIC_H
#define METRIC_H

#include "common.h"
#include "Scan.h"
#include "trace.h"
#include "Kernel.h"

#define GOM_DEPTH 2 
#define MI_BINS 30

class Metric {
public:

	virtual float EvalMetric(SparseScan* A, SparseScan* B);
};

class MI: public Metric {
public:
	MI(size_t numBins);
	float EvalMetric(SparseScan* A, SparseScan* B);
private:
	const size_t bins_;
};

class GOM: public Metric {
public:
	GOM(void);
	float EvalMetric(SparseScan* A, SparseScan* B);
};

class LIV: public Metric {
public:
	LIV(float* avImg, size_t width, size_t height);
	~LIV();
	float EvalMetric(SparseScan* A, SparseScan* B);
private:
	PointsList* avImg_;
};

/**
	\brief options required for setting up mi histograms

	\param threads
		number of threads to use
	\param blocks
		number of blocks to use
*/
struct cudaHistOptions
{
	int threads, blocks;
};

#endif //METRIC_H