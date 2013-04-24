function [] = SetTformMatrix(tMat)
%SETTFORMMATRIX sets tform matrix

%check input
if(~ismatrix(tMat) || (ndims(tMat) ~= 2) || (size(tMat,1) ~= size(tMat,2)) || (size(tMat,1) < 3) || (size(tMat,1) > 4))
    TRACE_ERROR('tform matrix must be either a 3 by 3 or 4 by 4 matrix, returning without setting');
    return;
end

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','setTformMatrix', single(tMat));

end