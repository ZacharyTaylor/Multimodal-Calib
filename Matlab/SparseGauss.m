function [ out ] = SparseGauss( kdTree, vals, sigma)
%SPARSEGAUSS gaussian blurs a list of values
%   kdTree - kdTree of positions
%   vals - values at each position
%   sigma- sd of gaussian blur

out = vals;

%only do 100,000 points at a time to save on memory
for j = 1:100000:size(kdTree.X,1)
    range = (j:min(j+100000,size(kdTree.X,1)));
    %getting all points within 2 standard deviations (95% of data)
    [idx,D]=rangesearch(kdTree,kdTree.X(range,:),2*sigma);

    for i = 1:size(idx,1)
        %gaussian blur
        f = (1/(sigma * sqrt(2*pi))) .* exp(-(D{i,:}.^2)/(2*(sigma.^2)));
        f = repmat(f',1,size(out,2));
        temp = out(idx{i,:},:).*f;
        out(j+i-1,:) = sum(temp,1)./sum(f,1);
    end
end

out(isnan(out)) = 0;


end

