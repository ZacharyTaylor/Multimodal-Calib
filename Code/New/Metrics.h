#ifndef METRIC_H
#define METRIC_H

#include "common.h"
#include "ScanList.h"

//! Depth of GOM images as need both phase and magnitude
#define GOM_DEPTH 2 

//! Metric to use when comparing two scans
class Metric {
public:
	//! Evalutaes two scans A and B to give a measure of their match strength
	virtual float evalMetric(std::vector<float*>& gen, ScanList scan, size_t index, cudaStream_t stream);
};

//! Evaluates scans using the MI metric
class MI: public Metric {
public:
	//! Sets up metric, note for MI due to implementation number of bins must be less then 64
	MI(size_t numBins);
	//! Evaluates MI for two scans and gives result
	float evalMetric(std::vector<float*>& gen, ScanList scan, size_t index, cudaStream_t stream);
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
	float evalMetric(std::vector<float*>& gen, ScanList scan, size_t index, cudaStream_t stream);
};

//! Evaluate scans using the GOM metric
class GOM: public Metric {
public:
	//! Basic setup
	GOM(void);
	//! Evaluates GOM for two scans and gives result
	float evalMetric(std::vector<float*>& gen, ScanList scan, size_t index, cudaStream_t stream);
};

//! Evaluates two scans using the Levinson method
class LIV: public Metric {
public:
	//! Setup takes an image that is the average of all base images used
	LIV(float* avImg, size_t width, size_t height);
	//! Destructor clears avImg_
	~LIV();
	//! Evaluates Levinson method for two scans and gives result
	float evalMetric(std::vector<float*>& gen, ScanList scan, size_t index, cudaStream_t stream);
private:
	//! Average of all base images
	//PointsList* avImg_;
};

#endif //METRIC_H
