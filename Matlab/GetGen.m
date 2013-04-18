function [ gen ] = GetGen()
%GETGENERATED outputs moving image imNum from scan

%ensures the library is loaded
CheckLoaded();

% %get location
locs = calllib('LibCal','getGenLocs');

%get points
points = calllib('LibCal','getGenPoints');

%get size of pointers
numPoints = calllib('LibCal','getGenNumPoints');
numCh = calllib('LibCal','getGenNumCh');
numDim = calllib('LibCal','getGenNumDim');

setdatatype(locs,'singlePtr',numPoints,numDim);
setdatatype(points,'singlePtr',numPoints,numCh);

%get data
locsVal = get(locs);
pointsVal = get(points);

gen = single(zeros(numPoints, (numDim+numCh)));


gen(:,1:numDim) = locsVal.Value;
gen(:,numDim+1:end) = pointsVal.Value;

end

