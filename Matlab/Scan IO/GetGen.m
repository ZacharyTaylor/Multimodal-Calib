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
gen = single(zeros(numPoints, (numDim+numCh)));

if(numDim ~= 0)
    locsVal = get(locs);
    gen(:,1:numDim) = locsVal.Value;
end

if(numCh ~= 0)
    pointsVal = get(points);
    gen(:,numDim+1:end) = pointsVal.Value;
end

end

