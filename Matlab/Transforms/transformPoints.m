function [ out ] = transformPoints( tform, in, inverse )
%TRANSFORMPOINTS Summary of this function goes here
%   Detailed explanation goes here

tformMat = angle2dcm(tform(4),tform(5),tform(6),'XYZ');
tformMat(4,4) = 1;
tformMat(1,4) = tform(1);
tformMat(2,4) = tform(2);
tformMat(3,4) = tform(3);

temp = ones(size(in,1),4);
temp(:,1:3) = in(:,1:3);

if(inverse)
    temp = (tformMat\(temp'))';
else
    temp = (tformMat*(temp'))';
end
out = [temp(:,1:3) in(:,4:end)];

end

