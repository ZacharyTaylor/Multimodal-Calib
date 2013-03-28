function [] = LoadBaseImage(imgNum, img)
%LOADBASEIMAGE creates a copy of the image for use with library

%check inputs
if(~isinteger(imgNum) || (imgNum < 0))
    TRACE_ERROR('number of base scans must be a positive integer, returning without setting');
    return;
end

if(~ismatrix(img) || (ndims(img) < 2))
    TRACE_ERROR('image must be a matrix with atleast 2 dimensions, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

%creating local copy of data that user can't mess with
numBase = calllib('LibCal','getNumBase');
persistent imgStore;
imgStore{numBase} = float(img);

%setting base image
calllib('LibCal','setBaseImage', imgNum, size(img,1), size(img,2), size(img,3), imgStore{numBase});

end

