function [] = CheckCudaErrors()
%CHECKCUDAERRORS checks for errors that occured while running cuda kernels

%ensures the library is loaded
CheckLoaded();

%checks for errors
calllib('LibCal','checkCudaErrors');
end

