#ifndef METRIC_H
#define METRIC_H

#include "common.h"
#include "ScanList.h"
#include "GenList.h"

//! Depth of GOM images as need both phase and magnitude
#define GOM_DEPTH 2 

//! Metric to use when comparing two scans
class Metric {
public:
	//! Evalutaes two scans A and B to give a measure of their match strength
	virtual float evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx);
};

//! Evaluates scans using the MI metric
class MI: public Metric {
public:
	//! Sets up metric, note for MI due to implementation number of bins must be less then 64
	MI(size_t numBins);
	//! Evaluates MI for two scans and gives result
	float evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx);
private:
	//! Number of bins to use when calculating MI
	const size_t bins_;
};

//! Evaluates scans using the MI metric
class NMI: public Metric {
public:
	//! Sets up metric, note for MI due to implementation number of bins must be less then 64
	NMI(size_t numBins);
	//! Evaluates MI for two scans and gives result
	float evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx);
private:
	//! Number of bins to use when calculating MI
	const size_t bins_;
};

//! Evaluates scans using the SSD metric
class SSD: public Metric {
public:
	//! Sets up metric
	SSD(void);
	//! Evaluates SSD for two scans and gives result
	float evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx);
};

//! Evaluate scans using the GOM metric
class GOM: public Metric {
public:
	//! Basic setup
	GOM(void);
	//! Evaluates GOM for two scans and gives result
	float evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx);
};

//! Evaluate scans using the GOMS metric
class GOMS: public Metric {
public:
	//! Basic setup
	GOMS(void);
	//! Evaluates GOMS for two scans and gives result
	float evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx);
};

//! Evaluates two scans using the Levinson method
class LEV: public Metric {
public:
	//! Basic setup
	LEV(void);
	//! Evaluates Levinson method for two scans and gives result
	float evalMetric(ScanList* scan, GenList* gen, size_t scanIdx, size_t genIdx);
};

#endif //METRIC_H
