function [ out ] = thinVel( data )
%THINVEL Summary of this function goes here
%   Detailed explanation goes here

    data(:,4) = data(:,4) - min(data(:,4));
    data(:,4) = data(:,4) / max(data(:,4));

    filtSize = 51;    
    kdTree = KDTreeSearcher(data(:,1:3),'distance','euclidean');

    %get nearest neighbours
    idx = knnsearch(kdTree,kdTree.X,'k',filtSize);
    vals = reshape(data(idx,4),size(data,1),filtSize);
       
    ssd = abs(vals - repmat(mean(vals,2),1,filtSize));
    ssd = mean(ssd,2);

    ssd(isnan(ssd)) = 0;
    ssd = ssd - min(ssd);
    ssd = ssd / max(ssd);
    ssd = histeq(ssd);
    
    out = data(ssd > 0.9, :); 


end

