function [] = SetCameraMatrix(camMat)
%SETCAMERAMATRIX sets 3x4 cam matrix

%check input
if(~ismatrix(camMat) || (ndims(camMat) ~= 2) || (size(camMat,1) ~= 3) || (size(camMat,2) ~= 4))
    TRACE_ERROR('camera matrix must be a 3 by 4 matrix, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','setCameraMatrix', single(camMat));

end