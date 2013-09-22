#ifndef CALIB_H
#define CALIB_H

#include "common.h"
#include "ImageList.h"
#include "ScanList.h"
#include "Tforms.h"
#include "Setup.h"
#include "Kernels.h"

#define IMAGE_DIM 2

class Calib {
protected:
	ScanList* moveStore;
	ImageList* baseStore;

	Tforms* tformStore;

	size_t allocateGenMem(ScanList* points, ImageList* images, std::vector<std::vector<float*>> genL, std::vector<std::vector<float*>> genI, size_t startIdx);

public:
	//! Constructor. Sets up graphics card for CUDA, sets transform type, camera type and metric type.
	Calib::Calib(std::string tformType, std::string metricType);

	//! Clears all the scans, excluding generated ones
	void clearScans(void);
	//! Clears all the images
	void clearImages(void);
	//! Clears all the transforms
	void clearTforms(void);
	//! Clears all the scans, images transforms etc
	void clearEverything(void);
	
	//! Adds scan onto end of stored scans
	void addScan(std::vector<thrust::host_vector<float>> ScanLIn, std::vector<thrust::host_vector<float>> ScanIIn);
	//! Adds image onto end of stored images
	void addImage(thrust::host_vector<float> imageIn, size_t height, size_t width, size_t depth, size_t tformIdx, size_t scanIdx);
	//! Adds transform onto end of stored transforms
	void addTform(thrust::host_vector<float> tformIn);

	//! Calculates the metrics value for the given data
	virtual float evalMetric(void);
	
	//! Outputs a render of the current alignment
	thrust::host_vector<float> getRender(size_t imageIdx);
}


#endif CALIB_H