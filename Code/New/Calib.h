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

	ScanList* genStore;

	Tforms* tformStore;
public:
	//! Constructor. Sets up graphics card for CUDA, sets transform type, camera type and metric type.
	Calib::Calib(size_t imHeight, size_t imWidth, size_t imDepth)

	//! Clears all the scans, excluding generated ones
	void clearScans(void);
	//! Clears all the generated scans
	void clearGenerated(void);
	//! Clears all the images
	void clearImages(void);
	//! Clears all the transforms
	void clearTforms(void);
	//! Clears all the scans, images transforms etc
	void clearEverything(void);
	
	//! Adds scan onto end of stored scans
	void addScan(std::vector<thrust::host_vector<float>> ScanLIn, std::vector<thrust::host_vector<float>> ScanIIn);
	//! Adds image onto end of stored images
	void addImage(thrust::host_vector<float> imageIn);
	//! Adds transform onto end of stored transforms
	void addTform(thrust::host_vector<float> tformIn);

	//! Calculates the metrics value for the given data
	float evalMetric(void);
	
	//! Outputs a render of the current alignment
	thrust::host_vector<float> getRender(size_t imageIdx);

#endif CALIB_H