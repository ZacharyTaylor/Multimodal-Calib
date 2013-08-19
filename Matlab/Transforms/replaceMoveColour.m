function [] = replaceMoveColour(imgNum)
%REPLACEMOVECOLOUR replaces the colours of the images with those stored in
%the generated scan

%check inputs
if((imgNum ~= round(imgNum)) || (imgNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','replaceMovePoints', imgNum);

end