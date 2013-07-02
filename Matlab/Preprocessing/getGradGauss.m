function [ Mag, Phase ] = getGradGauss( in, tform, sigma )
%GETGRADIENT Summary of this function goes here
%   Detailed explanation goes here

vals = in(:,4);

in = in(:,1:3);
in(:,4) = 1;

%transform points
tform = double(tform);   
tformMat = angle2dcm(tform(6), tform(5), tform(4));
tformMat(4,4) = 1;
tformMat(1,4) = tform(1);
tformMat(2,4) = tform(2);
tformMat(3,4) = tform(3);

in = (tformMat*(in'))';

%project points onto sphere
sphere = zeros(size(in,1),2);
sphere(:,1) = atan2(in(:,1), in(:,2));
sphere(:,2) = atan(in(:,3)./ sqrt(in(:,1).^2 + in(:,2).^2));

kdTree = KDTreeSearcher(sphere(:,1:2),'distance','euclidean');

for j = 1:10000:size(kdTree.X,1)
    range = (j:min(j+10000,size(kdTree.X,1)));
    
    %getting all points within 2 standard deviations (95% of data)
    [idx]=rangesearch(kdTree,kdTree.X(range,:),2*sigma);

    for i = 1:size(idx,1)

        Dx = kdTree.X(idx{i},1) - kdTree.X(idx{i}(1),1);
        Dy = kdTree.X(idx{i},2) - kdTree.X(idx{i}(1),2);

        %gaussian blur
        f = (1/((sigma^2) * 2 * pi)) .* exp(-(Dx.^2 + Dy.^2)/(2*(sigma.^2)));

        I = vals(idx{i,:}).*(f);
        vals(j+i-1) = sum(I)./sum(f);
    end
    j
end

tempX = zeros(size(vals));
tempY = zeros(size(vals));

for j = 1:10000:size(kdTree.X,1)
    range = (j:min(j+10000,size(kdTree.X(range,:),1)));
    
    %getting all points within 2 standard deviations (95% of data)
    [idx]=rangesearch(kdTree,kdTree.X(range,:),2*sigma);
    
    for i = 1:size(idx,1)

        Dx = kdTree.X(idx{i},1) - kdTree.X(idx{i}(1),1);
        Dy = kdTree.X(idx{i},2) - kdTree.X(idx{i}(1),2);

        %gaussian blur
        f = (1/((sigma^2) * 2 * pi)) .* exp(-(Dx.^2 + Dy.^2)/(2*(sigma.^2)));
        fx = (Dx/((sigma^4) * sqrt(2*pi))) .* exp(-(Dx.^2 + Dy.^2)/(2*(sigma.^2)));
        fy = (Dy/((sigma^4) * sqrt(2*pi))) .* exp(-(Dx.^2 + Dy.^2)/(2*(sigma.^2)));

        Ix = vals(idx{i,:}).*(fx);
        Iy = vals(idx{i,:}).*(fy);

        tempX(j+i-1) = sum(Ix)./sum(f);
        tempY(j+i-1) = sum(Iy)./sum(f);
    end
    j
end

Mag=sqrt(tempX .^ 2 + tempY .^ 2);
Magmax=max(Mag(:));
Mag=Mag/Magmax;

Phase = atan2d(tempY,tempX);
end

