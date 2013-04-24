function [ proMat ] = cam2Pro( fcX, fcY, ccX, ccY)
%CAM2PRO converts camera parameters into projection matrix

proMat = [fcX 0 (ccX) 0;...
    0 fcY (ccY) 0;...
    0 0 1 0];


end

