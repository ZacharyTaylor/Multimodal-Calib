function [ mag, phase ] = getGradient( in, tform )
%GETGRADIENT Summary of this function goes here
%   Detailed explanation goes here

cloud = in(:,1:3);

cloud(:,4) = 1;

%transform points
tform = double(tform);   
tformMat = angle2dcm(tform(6), tform(5), tform(4));
tformMat(4,4) = 1;
tformMat(1,4) = tform(1);
tformMat(2,4) = tform(2);
tformMat(3,4) = tform(3);

vals = in(:,4);
cloud = (tformMat*(cloud'))';

%project points onto sphere
sphere = zeros(size(cloud,1),2);
sphere(:,1) = atan2(cloud(:,1), cloud(:,2));
sphere(:,2) = atan(cloud(:,3)./ sqrt(cloud(:,1).^2 + cloud(:,2).^2));

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

dVals = sum(abs(dVals),2) /8;

mag = dVals;
phase = atan2d(dxLocs,dyLocs);

end

