function [] = LoadBaseImage(imgNum, img)
%LOADBASEIMAGE creates a copy of the image for use with library

%check inputs
if((imgNum ~= round(imgNum))|| (imgNum < 0))
    TRACE_ERROR('number of base scans must be a positive integer, returning without setting');
    return;
end

if(~ismatrix(img) || (ndims(img) < 2))
    TRACE_ERROR('image must be a matrix with atleast 2 dimensions, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

%setting base image
calllib('LibCal','setBaseImage', imgNum, size(img,1), size(img,2), size(img,3), single(img));

end

