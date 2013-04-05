function [] = SetupCamera(panoramic)
%SETUPCAMERA sets up the camera
%0 for pinpoint
%otherwise panoramic

if(~isscalar(panoramic))
    TRACE_ERROR('panoramic must be a scalar variable,0 for pinpoint, not 0 for panoramic');
    return;
end

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','setupCamera', panoramic);

end