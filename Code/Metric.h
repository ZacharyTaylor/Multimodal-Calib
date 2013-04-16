#ifndef METRIC_H
#define METRIC_H

#include "common.h"
#include "Scan.h"
#include "trace.h"
#include "Kernel.h"

#define GOM_DEPTH 2 
#define MI_BINS 50

class Metric {
public:

	virtual float EvalMetric(SparseScan* A, SparseScan* B);
};

class MI: public Metric {
public:

	float EvalMetric(SparseScan* A, SparseScan* B);
};

class GOM: public Metric {
public:

	float EvalMetric(SparseScan* A, SparseScan* B);
};

class LIV: public Metric {
public:

	float EvalMetric(SparseScan* A, SparseScan* B);
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