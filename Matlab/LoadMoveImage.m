function [] = LoadMoveImage(imgNum, img)
%LOADBASEIMAGE creates a copy of the image for use with library

%check inputs
if(~isinteger(imgNum) || (imgNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning without setting');
    return;
end

if(~ismatrix(img) || (ndims(img) < 2))
    TRACE_ERROR('image must be a matrix with atleast 2 dimensions, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

%creating local copy of data that user can't mess with
numMove = calllib('LibCal','getNumMove');
persistent imgStore;
imgStore{numMove} = float(img);

%setting base image
calllib('LibCal','setMoveImage', imgNum, size(img,1), size(img,2), size(img,3), imgStore{numMove});

end

