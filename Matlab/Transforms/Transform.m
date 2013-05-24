function [] = Transform(imgNum)
%TRANSFORM performs transform

%check inputs
if((imgNum ~= round(imgNum)) || (imgNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','transform', imgNum);

end