function [ tformMat ] = CreateTformMat( tform )
%CREATETFORMMAT Takes in tform outputs tformMat
%Beyond convoluted and in need of a clean up but it gives results that 
%match snark and pyceptions outputs, not sure what is going wrong with 
%standard matlab approach

if(and(size(tform,1) == 4,size(tform,2) == 4))
    tformMat = tform;
    return;
end

tform = tform(:)';

roll = tform(4); pitch = tform(5); yaw = tform(6);

tformMat = angle2dcm(yaw,pitch,roll);

tformMat(4,4) = 1;

tformMat(1:3,4) = tform(1:3);
end

