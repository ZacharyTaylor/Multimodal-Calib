#ifndef SCANLIST_H
#define SCANLIST_H

#include "common.h"

#define MEM_LIMIT 0.5

//! Holds the sensors scans
class ScanList {
private:
	//! vector holding all the scans location data, indexed by [scan num][dimension num][point num]
	std::vector< std::vector< thrust::device_vector< float > > > scanL;
	
	//! vector holding all the scans intensity data, indexed by [scan num][channel num][point num]
	std::vector< std::vector< thrust::device_vector< float > > > scanI;
	
public:
	//! Constructor creates an empty scan
	ScanList(void);

	//! Destructor
	~ScanList(void);

	//! Gets the number of dimensions the specified scan has
	/*! \param idx index of the scan
	*/
	size_t getNumDim(size_t idx);
	
	//! Gets the number of channels the specified scan has
	/*! \param idx index of the scan
	*/
	size_t getNumCh(size_t idx);
	
	//! Gets the number of points the specified scan has
	/*! \param idx index of the scan
	*/
	size_t getNumPoints(size_t idx);

	//! Gets the number of scans
	size_t getNumScans(void);

	//! Gets the pointer of the location array
	/*! \param idx index of scan
		\param dim index of dimension
	*/
	float* getLP(size_t idx, size_t dim);

	//! Gets the pointer of the intensity array
	/*! \param idx index of the scan
		\param ch index of the intensity channel to return
	*/
	float* getIP(size_t idx, size_t ch);

	//! Adds a scan to the list
	/*! \param scanLIn input scans location information
		\param scanIIn input scans intensity information
	*/
	void addScan(std::vector<thrust::device_vector<float>>& scanLIn, std::vector<thrust::device_vector<float>>& scanIIn);

	//! Adds a scan to the list
	/*! \param scanLIn input scans location information
		\param scanIIn input scans intensity information
	*/
	void addScan(std::vector<thrust::host_vector<float>>& scanLIn, std::vector<thrust::host_vector<float>>& scanIIn);

	//! Removes a scan from the list
	/*! \param idx index of scan to remove
	*/
	void removeScan(size_t idx);

	//! Removes the last scan on the list
	void removeLastScan();

	//! Removes all of the scans in the list
	void removeAllScans();

	void generateImage(thrust::device_vector<float>& out, size_t idx, size_t width, size_t height, size_t dilate);

};

#endif //SCANLIST_H
