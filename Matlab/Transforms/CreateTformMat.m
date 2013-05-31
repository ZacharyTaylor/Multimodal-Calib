function [ tformMat ] = CreateTformMat( tform )
%CREATETFORMMAT Takes in tform outputs tformMat

tformMat = zeros(4,4);
tformMat(1:3,1:3) = angle2dcm(tform(6), tform(5), tform(4));
tformMat(4,4) = 1;

tformMat(1:3,4) = tformMat(1:3,1:3)*tform(1:3)';

end

