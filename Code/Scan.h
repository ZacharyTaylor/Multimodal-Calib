#ifndef SCAN_H
#define SCAN_H

#include "Points.h"
#include "common.h"
#include "trace.h"

//! Number of dimensions a photo has
#define IMAGE_DIM 2

//! Holds the sensors scans and unifies the method for accessing them
class Scan {
protected:
	//! Number of dimensions scan has
	size_t numDim_;
	//! Number of channels of information assosiated with each point in scan
	size_t numCh_;

	//! Array of length numDim_ specifiying the scans size in each dimension
	size_t* dimSize_;
	//! Holds the points of the scan
	PointsList* points_;

public:
	//! Constructor creates an empty scan
	/*!
		\param numDim Number of dimensions scan has
		\param numCh Number of channels of information assosiated with each point in scan
		\param dimSize Array of length numDim_ specifiying the scans size in each dimension
	*/
	Scan(size_t numDim, size_t numCh,  size_t* dimSize);

	//! Constructs a scan that holds given points
	/*!
		\param numDim Number of dimensions scan has
		\param numCh Number of channels of information assosiated with each point in scan
		\param dimSize Array of length numDim_ specifiying the scans size in each dimension
		\param points PointsList of points that hold the location and intensity information of the sacn
	*/
	Scan(size_t numDim, size_t numCh,  size_t* dimSize, PointsList* points);

	//! Destructor clears points_ and dimSize_ from memory
	~Scan(void);

	//! Gets the number of dimensions scan has
	size_t getNumDim(void);
	
	//! Gets the number of channels a scan has
	size_t getNumCh(void);
	
	//! Gets the size of the ith dimension
	/*
		\param i the dimension you want the size of
	*/
	size_t getDimSize(size_t i);
	
	//! Gets the number of points
	size_t getNumPoints(void);
	
	//! Gets the pointer to the PointsList containing all the points
	PointsList* getPoints(void);

	//! Binds a new points list to the scan
	void setPoints(PointsList* points);

	//! Allocates and copies points to the GPU
	void SetupGPU(void);

	//! Clears points from GPU
	void ClearGPU(void);
};

//! Extends class Scan by storing a location for each scan point allowing sparse scans to be stored
class SparseScan: public Scan {
private:

	//! Sets up the number and size of the dimensions and channels of the scan
	static size_t* setDimSize(const size_t numCh, const size_t numDim, const size_t numPoints);

protected:

	//! Stores the location of each point may be n dimensional
	PointsList* location_;
	
public:

	//! Generates an array of locations in a dense grid.
	/*! For a given dimension space and size generates the location of points, assuiming resolution of scan is 1 and point are ordered by dimension (x,y,z,...) (used to convert images to sparse scans)
		\param numDim number of dimensions
		\param dimSize array of size numDim, containg the size of each dimension
	*/
	static float* SparseScan::GenLocation(size_t numDim, size_t* dimSize);

	//! Constructor, creates a SparseScan with no points
	/*!
		\param numDim the number of dimensions the scan has
		\param numCh number of channels of information assosiated with each point in scan
		\param numPoints number of points in the scan
	*/
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints);

	//! Constructor, creates a SparseScan with the points and locations given as PointsList
	/*!
		\param numDim the number of dimensions the scan has
		\param numCh number of channels of information assosiated with each point in scan
		\param numPoints number of points in the scan
		\param points the intensity information for each point (of size numCh*numPoints)
		\param location the location information for each point (of size numDim*numPoints)
	*/
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, PointsList* points, PointsList* location);

	//! Constructor, creates a SparseScan with the points and locations given as arrays
	/*!
		\param numDim the number of dimensions the scan has
		\param numCh number of channels of information assosiated with each point in scan
		\param numPoints number of points in the scan
		\param points the intensity information for each point (of size numCh*numPoints)
		\param location the location information for each point (of size numDim*numPoints)
	*/
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, float* pointsIn, float* locationIn);
	
	//! Constructs a sparse scan from a dense scan, locations generated using GenLocation
	/*!
		\param in the input Scan
	*/
	SparseScan(Scan in);

	//! Constructs a SparseScan from a Scan giving each point a provided location
	/*!
		\param in the input Scan
		\param location the location of each Point
	*/
	SparseScan(Scan in, PointsList* location);

	//! Destructor, clears the location_ and points_ PointsList as well as any other allocated memory
	~SparseScan(void);

	//! Changes the number of information channels
	/*! If this is used to set numCh to a value for which the size of points < numCh*numPoints the program will segfault as soon 
		as you try do anything with the data
		\param numCh number of channels of information points_ has
	*/
	void changeNumCh(size_t numCh);
	
	//! Gets the number of points in the Scan
	size_t getNumPoints(void);

	//! Gets the location of the points
	PointsList* GetLocation(void);
};

//! Extends Scan with dense points stored in a little endien (changing first dimension first) grid
/*! Adds support for using texture memory for images (only supports 2 dimensions at the moment)*/
class DenseImage: public Scan {
public:

	//! Constructor, creates a DenseImage with the points given as TextureList
	/*!
		\param width width of image
		\param height height of image
		\param numCh number of colour channels Image has
		\param points the intensity information for each point (of size numCh*numPoints)
	*/
	DenseImage(const size_t width, const size_t height, const size_t numCh, TextureList* points);
	
	//! Constructor, creates a DenseImage with the points given as an array
	/*!
		\param width width of image
		\param height height of image
		\param numCh number of colour channels Image has
		\param pointsIn the intensity information for each point (of size numCh*numPoints)
	*/
	DenseImage(const size_t width, const size_t height, const size_t numCh, float* pointsIn);
	
	//! Destructor clears points_ and other allocated memory
	~DenseImage(void);

	//! gets the pointer to the points stored in dense image
	TextureList* getPoints(void);

	//! interpolates a SparseScan to generate a DenseImage (very quick and gives poor results only used for quick visual checking of data)
	/*!
		\param scan Scan to interpolate
	*/
	void d_interpolate(SparseScan* scan);

private:

	/! Sets up the number and size of the dimensions and channels of the scan
	static size_t* setDimSize(const size_t width, const size_t height, const size_t numCh);
};

#endif //SCAN_H