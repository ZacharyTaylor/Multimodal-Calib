#ifndef GENLIST_H
#define GENLIST_H

#include "common.h"

#define TEMP_MEM_SIZE 65535

//! Holds the sensors scans
class GenList {
private:
	//! vector holding generated location data, indexed by [scan num][dimension num][point num]
	std::vector< std::vector< thrust::device_vector< float > > > genL;
	//! vector holding generated intensity data, indexed by [scan num][channel num][point num]
	std::vector< std::vector< thrust::device_vector< float > > > genI;
	//! vector of stream used to perform operations on scan
	std::vector<cudaStream_t> streams;
	//! temporary memory used in reductions (declared out here as mallocs force gpu sync)
	std::vector< thrust::device_vector< float > > tempMemD;

	//! more reduction temp memory
	std::vector<float*> tempMemH;

public:
	//! Destructor
	~GenList(void);

	//! Constructor creates an empty scan
	void setupGenList(size_t numScans);

	//! Gets the number of generated scans that exist
	size_t getNumGen(void);

	//! Gets the stream of the corrosponding scan
	/*! \param idx index of the scan
	*/
	cudaStream_t getStream(size_t idx);

	//! Gets the pointer of the generated location array
	/*! \param idx index of scan
		\param dim index of dimension
		\param numPoints number of points that will be stored in generated array
	*/
	float* getGLP(size_t idx, size_t dim, size_t numPoints);

	//! Gets the pointer of the generated intensity array
	/*! \param idx index of the scan
		\param ch index of the intensity channel to return
		\param numPoints number of points that will be stored in generated array
	*/
	float* getGIP(size_t idx, size_t ch, size_t numPoints);

	//! Gets the pointer to the temporary device memory
	/*! \param idx index of the scan
	*/
	float* getTMPD(size_t idx);
	
	//! Gets the pointer to the temporary host memory
	/*! \param idx index of the scan
	*/
	float* getTMPH(size_t idx);
};

#endif //GENLIST_H
