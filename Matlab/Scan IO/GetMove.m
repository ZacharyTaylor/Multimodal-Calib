function [ move ] = GetMove(imgNum)
%GETMOVE outputs moving image imNum from scan

%check inputs
if((imgNum ~= round(imgNum)) || (imgNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning');
    return;
end

imTot = calllib('LibCal','getNumMove');

if(imTot <= imgNum)
    string = sprintf('requested image %i of %i',imgNum,imTot);
    TRACE_ERROR(string);
    return;
end

%ensures the library is loaded
CheckLoaded();

%get location
locs = calllib('LibCal','getMoveLocs', imgNum);

%get points
points = calllib('LibCal','getMovePoints', imgNum);

%get size of pointers
numPoints = calllib('LibCal','getMoveNumPoints', imgNum);
numCh = calllib('LibCal','getMoveNumCh', imgNum);
numDim = calllib('LibCal','getMoveNumDim', imgNum);

setdatatype(locs,'singlePtr',numPoints,numDim);
setdatatype(points,'singlePtr',numPoints,numCh);

%get data
locsVal = get(locs);
pointsVal = get(points);

move = single(zeros(numPoints, (numDim+numCh)));


move(:,1:numDim) = locsVal.Value;
move(:,numDim+1:end) = pointsVal.Value;

end

