#include "Pair.h"

void Pair::SetMove(const size_t numDim, const size_t numCh,  const size_t numPoints, float* pointsIn, float* locIn){

	PointsList* points = new PointsList(pointsIn, numPoints*numCh);
	PointsList* loc = new PointsList(locIn, numPoints*numDim);

	move_ = new SparseScan(numDim, numCh,  numPoints, points, loc);
}

void Pair::SetMove(const size_t numDim, const size_t numCh,  const size_t numPoints, float* pointsIn){

	PointsList* points = new PointsList(pointsIn, numPoints*numCh);
	move_ = new SparseScan(numDim, numCh,  numPoints, points);
}

void Pair::MoveSetupGpu(void){
	move_->GetLocation()->AllocateGpu();
	move_->GetLocation()->CpuToGpu();

	move_->getPoints()->AllocateGpu();
	move_->getPoints()->CpuToGpu();
}

void Pair::SetupAffineTransform(){
	tform_ = new AffineTform();

	gen_ = new SparseScan(move_->getNumDim(), move_->getNumCh(), move_->getNumPoints());
	gen_->GetLocation()->AllocateGpu();
	gen_->getPoints()->AllocateGpu();
}

void Pair::transform(float* tform){
	tform_->SetTform(tform);
	tform_->d_Transform(move_, gen_);
}