function [] = InterpolateBaseValues(baseNum )
%INTERPOLATEBASEVALUES interpolates the base image so that the colour of
%the points at all of the moving images positions is known

%check inputs
if((baseNum ~= round(baseNum)) || (baseNum < 0))
    TRACE_ERROR('number of base scans must be a positive integer, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','genBaseValues', baseNum);

end

