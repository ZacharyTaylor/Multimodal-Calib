function [ mag, phase ] = get2DGradient( in, tform )
%GETGRADIENT Summary of this function goes here
%   Detailed explanation goes here

%transform points
tform = double(tform);   
tformMat = angle2dcm(tform(6), tform(5), tform(4));
tformMat(4,4) = 1;
tformMat(1,4) = tform(1);
tformMat(2,4) = tform(2);
tformMat(3,4) = tform(3);

vals = in(:,4:size(in,2));

in = in(:,1:4);
in(:,4) = 1;

in = (tformMat*(in'))';

%project points onto sphere
sphere = zeros(size(in,1),2);
sphere(:,1) = atan2(in(:,1), in(:,2));
sphere(:,2) = atan(in(:,3)./ sqrt(in(:,1).^2 + in(:,2).^2));

kdTree = KDTreeSearcher(sphere(:,1:2),'distance','euclidean');

%get nearest neighbours
idx = knnsearch(kdTree,kdTree.X,'k',9);

%remove self
idx = idx(:,2:end);

dVals = repmat(vals,1,8);
dVals(:) = dVals(:) - vals(idx(:));

xLocs = kdTree.X(:,1);
dxLocs = repmat(xLocs,1,8);
dxLocs(:) = dxLocs(:) - xLocs(idx(:));

yLocs = kdTree.X(:,2);
dyLocs = repmat(yLocs,1,8);
dyLocs(:) = dyLocs(:) - yLocs(idx(:));

dxLocs = sum(dxLocs.*dVals,2) /8;
dyLocs = sum(dyLocs.*dVals,2) /8;

dVals = sum(abs(dVals)./sum(sqrt(dxLocs.^2+dyLocs.^2)),2) /8;

mag = dVals;
%get angle from 0 to 90 degrees
phase = abs(atan2d(dxLocs,dyLocs));

end

