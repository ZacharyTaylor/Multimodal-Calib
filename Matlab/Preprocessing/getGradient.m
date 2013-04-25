function [ mag, phase ] = getGradient( in )
%GETGRADIENT Summary of this function goes here
%   Detailed explanation goes here

vals = in(:,4:size(in,2));
kdTree = KDTreeSearcher(in(:,1:3),'distance','euclidean');

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
%get angle from 0 to 90 degrees
phase = abs(atan2d(dxLocs,dyLocs));
phase = abs(phase-90);

end

