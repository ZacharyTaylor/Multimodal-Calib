function [ gen ] = GetGen(idx)
%GETGENERATED outputs moving image imNum from scan

if((idx ~= round(idx)) || (idx < 0))
    TRACE_ERROR('idx must be a positive integer, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

% %get location
locs = calllib('LibCal','getGenLocs',idx);

%get points
points = calllib('LibCal','getGenPoints',idx);

%get size of pointers
numPoints = calllib('LibCal','getGenNumPoints',idx);
numCh = calllib('LibCal','getGenNumCh',idx);
numDim = calllib('LibCal','getGenNumDim',idx);

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

