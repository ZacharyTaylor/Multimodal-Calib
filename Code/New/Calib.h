#ifndef CALIB_H
#define CALIB_H

#include "common.h"
#include "ImageList.h"
#include "ScanList.h"
#include "Tforms.h"
#include "Cameras.h"
#include "Setup.h"
#include "Kernels.h"
#include "Metrics.h"

#define IMAGE_DIM 2

class Calib {
protected:
	ScanList moveStore;
	ImageList baseStore;

	Tforms tformStore;

	Metric* metric;

	//! index of transform to use on image
	std::vector<size_t> tformIdx;
	//! index of scan assosiated with image
	std::vector<size_t> scanIdx;

	size_t allocateGenMem(ScanList points, ImageList images, std::vector<std::vector<float*>>& genL, std::vector<std::vector<float*>>& genI, size_t startIdx);

public:
	//! Constructor. Sets up graphics card for CUDA, sets transform type, camera type and metric type.
	Calib(std::string metricType);

	//! Clears all the scans, excluding generated ones
	void clearScans(void);
	//! Clears all the images
	void clearImages(void);
	//! Clears all the transforms
	void clearTforms(void);
	//! Clears any extra parts that may be setup by derived classes
	void clearExtras(void);
	//! Clears all the scans, images transforms etc
	void clearEverything(void);
	
	//! Adds a camera onto the end of stored cameras
	virtual void addCamera(thrust::host_vector<float>& cameraIn, boolean panoramic);
	//! Adds scan onto end of stored scans
	void addScan(std::vector<thrust::host_vector<float>>& ScanLIn, std::vector<thrust::host_vector<float>>& ScanIIn);
	//! Adds image onto end of stored images
	void addImage(thrust::host_vector<float>& imageIn, size_t height, size_t width, size_t depth);
	//! Adds transform onto end of stored transforms
	void addTform(thrust::host_vector<float>& tformIn, size_t tformSizeX, size_t tformSizeY);

	//! Adds the cameras indecies relating cameras to images
	virtual void addCameraIndices(std::vector<size_t>& cameraIdxIn);
	//! Adds the transforms indecies relating tforms to images
	void addTformIndices(std::vector<size_t>& tformsIdxIn);
	//! Adds the scans indecies relating scans to images
	void addScanIndices(std::vector<size_t>& scanIdxIn);

	//! Sets the metric to use SSD for evaluation
	void setSSDMetric(void);
	//! Sets the metric to use GOM for evaluation
	void setGOMMetric(void);

	//! Calculates the metrics value for the given data
	virtual float evalMetric(void);
	
	//! Outputs a render of the current alignment
	thrust::host_vector<float> getRender(size_t imageIdx);
};

class CameraCalib: public Calib {
protected:
	CameraTforms tformStore;
	Cameras cameraStore;

	std::vector<size_t> cameraIdx;

public:
	//! Constructor. Sets up graphics card for CUDA, sets transform type, camera type and metric type.
	CameraCalib(std::string metricType);

	//! Adds the cameras indecies relating cameras to images
	void addCameraIndices(std::vector<size_t>& cameraIdxIn);
	//! Adds a camera onto the end of stored cameras
	void addCamera(thrust::host_vector<float>& cameraIn, boolean panoramic);
	
	void addTform(thrust::host_vector<float>& tformIn);
	//! Calculates the metrics value for the given data
	float evalMetric(void);
};

#endif CALIB_H