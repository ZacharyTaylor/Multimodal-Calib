function [ baseOut ] = GetBase(imgNum)
%GETMOVE outputs base image imNum from scan

%check inputs
if((imgNum ~= round(imgNum)) || (imgNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning');
    return;
end

imTot = calllib('LibCal','getNumBase');

if(imTot <= imgNum)
    string = sprintf('requested image %i of %i',imgNum,imTot);
    TRACE_ERROR(string);
    return;
end

%ensures the library is loaded
CheckLoaded();

%get image
base = calllib('LibCal','getBaseImage', imgNum);

%get size of pointers
height = calllib('LibCal','getBaseDim', imgNum, 0);
width = calllib('LibCal','getBaseDim', imgNum, 1);
depth = calllib('LibCal','getBaseNumCh', imgNum);

setdatatype(base,'singlePtr',height*width*depth,1);

%get data
baseVal = get(base);

baseOut = baseVal.Value;
baseOut = reshape(baseOut,height,width,depth);

end

