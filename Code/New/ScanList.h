#ifndef SCANLIST_H
#define SCANLIST_H

#include "common.h"

#define MEM_LIMIT 0.5

//! Holds the sensors scans
class ScanList {
private:
	//! vector holding all the scans location data
	std::vector<thrust::device_vector<float>> scanL;
	
	//! vector holding all the scans intensity data
	std::vector<thrust::device_vector<float>> scanI;
	
	//! vector holding the index of scans in vectors
	thrust::device_vector<size_t> scanIdx;

public:
	//! Constructor creates an empty scan
	ScanList(void);

	//! Destructor
	~ScanList(void);

	//! Gets the number of dimensions the scans have
	size_t getNumDim(void);
	
	//! Gets the number of channels the scans have
	size_t getNumCh(void);
	
	//! Gets the number of points the scans have
	size_t getNumPoints(void);

	//! Gets the pointer of the location array
	/*! \param idx index of the dimension to return
	*/
	float* getLP(size_t idx);

	//! Gets the pointer of the intensity array
	/*! \param idx index of the intensity channel to return
	*/
	float* getIP(size_t idx);

	//! Gets the pointer of scans index
	size_t* getIdxP(void);

	//! Adds a scan to the list
	/*! \param scanLIn input scans location information
		\param scanIIn input scans intensity information
	*/
	void addScan(std::vector<thrust::device_vector<float>> scanLIn, std::vector<thrust::device_vector<float>> scanIIn);

	//! Adds a scan to the list
	/*! \param scanLIn input scans location information
		\param scanIIn input scans intensity information
	*/
	void addScan(std::vector<thrust::host_vector<float>> scanLIn, std::vector<thrust::host_vector<float>> scanIIn);

	//! Removes a scan from the list
	/*! \param idx index of scan to remove
	*/
	void removeScan(size_t idx);

	//! Removes the last scan on the list
	void removeLastScan();

	//! Removes all of the scans in the list
	void removeAllScans();

	//! Allocates memory until it is the same size as input or it hits capacity (may clear scans)
	size_t allocateMemory(size_t dims, size_t ch, size_t length);
};

#endif //SCANLIST_H
