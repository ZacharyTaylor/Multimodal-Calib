function [ out ] = transformPoints( tform, in, inverse )
%TRANSFORMPOINTS Summary of this function goes here
%   Detailed explanation goes here

tformMat = CreateTformMat(tform);

temp = ones(size(in,1),4);
temp(:,1:3) = in(:,1:3);

if(inverse)
    temp = (tformMat\(temp'))';
else
    temp = (tformMat*(temp'))';
end
out = [temp(:,1:3) in(:,4:end)];

end

