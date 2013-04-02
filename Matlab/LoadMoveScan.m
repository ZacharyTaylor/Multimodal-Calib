function [] = LoadMoveScan(scanNum, scan, numDim)
%LOADBASEIMAGE creates a copy of the image for use with library

%check inputs
if((imgNum ~= round(imgNum)) || (scanNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning without setting');
    return;
end

if(~ismatrix(scan) || (ndims(scan) ~= 2))
    TRACE_ERROR('scan must be a matrix with 2 dimensions, returning without setting');
    return;
end

if(size(scan,2) - numDim < 0)
    TRACE_ERROR('scan cannot be %i dimensional as only %i wide, returning without setting',numDim, size(scan,2));
    return;
end

%ensures the library is loaded
CheckLoaded();

%setting base image
calllib('LibCal','setMoveScan', scanNum, numDim, (size(scan,2)-numDim), size(scan,1), scan);

end

