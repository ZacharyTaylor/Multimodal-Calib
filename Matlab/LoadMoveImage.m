function [] = LoadMoveImage(imgNum, img)
%LOADBASEIMAGE creates a copy of the image for use with library

%check inputs
if((imgNum ~= round(imgNum)) || (imgNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning without setting');
    return;
end

if((ndims(img) < 2))
    TRACE_ERROR('image must be a matrix with atleast 2 dimensions, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

%setting move image (dim 2 before 1 because (y,x,z) is way too confusing
imgT = zeros([size(img,2),size(img,1),size(img,3)]);
for i = 1:size(img,3)
    imgT(:,:,i) = img(:,:,i)';
end

%system to use
calllib('LibCal','setMoveImage', imgNum, size(imgT,1), size(imgT,2), size(imgT,3), single(imgT));

end

